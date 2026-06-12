"""Tests for :class:`CoverageRepository`.

The coverage queries decide which (location, date, variable) tuples a
satellite ingest job needs to fetch. They run *before* any API call, so a
buggy query that silently scans the whole warehouse can cost minutes per
station even when the location set is tiny. These tests lock in the
must-have filters — date, source, variable, and (when given) geohash5 —
in the generated SQL so a future refactor can't accidentally drop them.
"""

from __future__ import annotations

from datetime import date

from susse.warehouse_ops.population.coverage import CoverageRepository
from susse.warehouse_ops.population.types import DateRange, Source


class _RecordingBQ:
    """Captures every ``existing_keys`` call without executing anything."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def existing_keys(
        self,
        table_fqn: str,
        key_columns,
        *,
        where_filters=(),
    ) -> set[tuple]:
        self.calls.append(
            {
                "table_fqn": table_fqn,
                "key_columns": tuple(key_columns),
                "where_filters": tuple(where_filters),
            }
        )
        return set()  # pretend the warehouse is empty

    @property
    def last_call(self) -> dict:
        assert self.calls, "no BQ call recorded"
        return self.calls[-1]


def _date_range() -> DateRange:
    return DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))


class TestExistingLongKeys:
    """``existing_long_keys`` builds a scoped coverage query."""

    def test_includes_date_source_and_variable_filters(self) -> None:
        bq = _RecordingBQ()
        repo = CoverageRepository(bq)  # type: ignore[arg-type]
        repo.existing_long_keys(
            "p.d.nasa_long",
            date_range=_date_range(),
            source=Source.NASA_POWER,
            variable_ids=("temperature", "wind_speed"),
        )
        filters = bq.last_call["where_filters"]
        joined = " ".join(filters)
        assert "date BETWEEN DATE('2024-01-01') AND DATE('2024-01-31')" in joined
        assert "source = 'NASA'" in joined
        assert "variable_id IN ('temperature', 'wind_speed')" in joined
        # No geohash filter when geohash5s is not passed (preserves legacy
        # behaviour for grid plans where every grid point is in scope).
        assert not any("geohash5" in f for f in filters)

    def test_geohash5s_added_to_filter_when_provided(self) -> None:
        """The point of A6's coverage fix: a small location set scopes the scan.

        Without this filter, named-location plans whose date range overlaps
        the existing warehouse footprint trigger multi-million-row scans.
        """
        bq = _RecordingBQ()
        repo = CoverageRepository(bq)  # type: ignore[arg-type]
        repo.existing_long_keys(
            "p.d.nasa_long",
            date_range=_date_range(),
            source=Source.NASA_POWER,
            variable_ids=("temperature",),
            geohash5s=("s8p1v", "kzcw8"),
        )
        joined = " ".join(bq.last_call["where_filters"])
        assert "geohash5 IN ('s8p1v', 'kzcw8')" in joined

    def test_empty_geohash5s_skips_bq(self) -> None:
        """No locations to ingest → no rows can match, so don't even ask BQ."""
        bq = _RecordingBQ()
        repo = CoverageRepository(bq)  # type: ignore[arg-type]
        result = repo.existing_long_keys(
            "p.d.nasa_long",
            date_range=_date_range(),
            source=Source.NASA_POWER,
            variable_ids=("temperature",),
            geohash5s=(),
        )
        assert result == set()
        assert bq.calls == [], "empty geohash5s must short-circuit the BQ call"


class TestExistingIrradianceKeys:
    """``existing_irradiance_keys`` mirrors the long-keys contract."""

    def test_includes_date_and_source_filters(self) -> None:
        bq = _RecordingBQ()
        repo = CoverageRepository(bq)  # type: ignore[arg-type]
        repo.existing_irradiance_keys(
            "p.d.irr",
            date_range=_date_range(),
            source=Source.CAMS,
        )
        joined = " ".join(bq.last_call["where_filters"])
        assert "date BETWEEN DATE('2024-01-01') AND DATE('2024-01-31')" in joined
        assert "source = 'CAMS'" in joined
        assert not any("geohash5" in f for f in bq.last_call["where_filters"])

    def test_geohash5s_added_to_filter_when_provided(self) -> None:
        bq = _RecordingBQ()
        repo = CoverageRepository(bq)  # type: ignore[arg-type]
        repo.existing_irradiance_keys(
            "p.d.irr",
            date_range=_date_range(),
            source=Source.CAMS,
            geohash5s=("s8p1v",),
        )
        joined = " ".join(bq.last_call["where_filters"])
        assert "geohash5 IN ('s8p1v')" in joined

    def test_empty_geohash5s_skips_bq(self) -> None:
        bq = _RecordingBQ()
        repo = CoverageRepository(bq)  # type: ignore[arg-type]
        result = repo.existing_irradiance_keys(
            "p.d.irr",
            date_range=_date_range(),
            source=Source.CAMS,
            geohash5s=(),
        )
        assert result == set()
        assert bq.calls == []
