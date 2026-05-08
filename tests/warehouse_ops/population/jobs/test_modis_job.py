"""Tests for ModisJob — MODIS-specific contracts.

End-to-end ``run()`` against the live ORNL DAAC API is exercised from the
notebook. These tests cover the contracts that don't require a network
round-trip: variable-id parsing, cadence-aware coverage, plan validation,
and the per-location skip path.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from susse.warehouse_ops.population.jobs.modis_job import (
    ModisJob,
    _COVERAGE_RATIO_THRESHOLD,
    _split_variable_id,
)
from susse.warehouse_ops.population.types import (
    DateRange,
    GridPlan,
    GridSpec,
    BoundingBox,
    LocationSpec,
    NamedLocationsPlan,
    Source,
    VariableSpec,
)


def _modis_var(variable_id: str, api_code: str) -> VariableSpec:
    return VariableSpec(
        variable_id=variable_id, source=Source.MODIS, api_code=api_code,
        display_name=variable_id, unit="x", native_unit="x", description="x",
    )


class _StubBQ:
    @property
    def config(self):
        from susse.warehouse_ops.io.config import WarehouseConfig
        return WarehouseConfig()


class _StubFetcher:
    """Stub MODIS fetcher that records calls and returns canned rows."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.calls: list[dict] = []
        self._rows = rows if rows is not None else []

    def fetch_long_for_location(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        return pd.DataFrame(
            self._rows,
            columns=("date", "product_id", "band_id", "value"),
        )


class TestSplitVariableId:
    """``variable_id`` follows ``f"{api_code}_{band_id}"`` — recover both halves."""

    def test_simple_band(self) -> None:
        v = _modis_var("MCD43A4_Nadir_Reflectance_Band1", "MCD43A4")
        assert _split_variable_id(v) == ("MCD43A4", "Nadir_Reflectance_Band1")

    def test_band_containing_underscores(self) -> None:
        # The band id itself contains underscores; only the leading
        # api_code prefix should be stripped, leaving the band intact.
        v = _modis_var("MOD11A2_LST_Day_1km", "MOD11A2")
        assert _split_variable_id(v) == ("MOD11A2", "LST_Day_1km")

    def test_band_starting_with_digit(self) -> None:
        v = _modis_var("MOD13Q1_250m_16_days_NDVI", "MOD13Q1")
        assert _split_variable_id(v) == ("MOD13Q1", "250m_16_days_NDVI")

    def test_raises_when_variable_id_does_not_start_with_api_code(self) -> None:
        # Wrong catalog setup: variable_id doesn't carry the api_code prefix.
        # Must surface a clear remediation message rather than silently
        # producing a malformed (product, band) pair.
        v = _modis_var("Foo_Bar", "MCD43A4")
        with pytest.raises(ValueError, match="api_code prefix"):
            _split_variable_id(v)


class TestLocationFullyCached:
    """Coverage check uses cadence × ratio threshold per (product, band)."""

    @staticmethod
    def _job() -> ModisJob:
        return ModisJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]

    def test_returns_true_when_all_pairs_meet_threshold(self) -> None:
        # 16 days for MOD13Q1 (16-day cadence) → expected = 1 row;
        # threshold = 0.9 → need >=0.9 row → 1 actual row passes.
        existing = {
            (date(2024, 6, 1), "s8p1v", "MOD13Q1", "250m_16_days_NDVI", "MODIS"),
        }
        date_range = DateRange(start=date(2024, 6, 1), end=date(2024, 6, 16))
        cached = self._job()._location_fully_cached(
            "s8p1v",
            date_range=date_range,
            product_band_pairs=(("MOD13Q1", "250m_16_days_NDVI"),),
            existing_keys=existing,
        )
        assert cached is True

    def test_returns_false_when_pair_below_threshold(self) -> None:
        # MOD11A2 has 8-day cadence over 64 days → expected ~8 rows.
        # 5 actual rows = 0.625 ratio < 0.9 threshold → must refetch.
        existing = {
            (date(2024, 6, 1) + pd.Timedelta(days=8 * i), "s8p1v",
             "MOD11A2", "LST_Day_1km", "MODIS")
            for i in range(5)
        }
        date_range = DateRange(start=date(2024, 6, 1), end=date(2024, 8, 4))
        cached = self._job()._location_fully_cached(
            "s8p1v",
            date_range=date_range,
            product_band_pairs=(("MOD11A2", "LST_Day_1km"),),
            existing_keys=existing,
        )
        assert cached is False

    def test_returns_false_if_one_of_many_pairs_uncovered(self) -> None:
        # First pair is fully covered, second has no rows → must refetch.
        existing = {
            (date(2024, 6, 1), "s8p1v", "MOD13Q1", "250m_16_days_NDVI", "MODIS"),
        }
        date_range = DateRange(start=date(2024, 6, 1), end=date(2024, 6, 16))
        cached = self._job()._location_fully_cached(
            "s8p1v",
            date_range=date_range,
            product_band_pairs=(
                ("MOD13Q1", "250m_16_days_NDVI"),
                ("MOD11A2", "LST_Day_1km"),  # zero rows present
            ),
            existing_keys=existing,
        )
        assert cached is False

    def test_threshold_constant_is_under_one(self) -> None:
        # Pin the slack: composite end-dates align to fixed Julian-day
        # offsets, so an N-day window can yield slightly fewer rows than
        # the naïve N/cadence count. Must not be 1.0 or strict equality
        # would over-refetch on every re-run.
        assert 0.5 < _COVERAGE_RATIO_THRESHOLD < 1.0


class TestRunRejectsBadPlan:
    def test_raises_on_wrong_plan_type(self) -> None:
        job = ModisJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]

        class FakePlan:
            def describe(self) -> str:
                return "fake"

        with pytest.raises(TypeError, match="NamedLocationsPlan or GridPlan"):
            job.run(FakePlan())  # type: ignore[arg-type]

    def test_raises_on_source_mismatch(self) -> None:
        job = ModisJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.NASA_POWER,  # wrong source for MODIS job
            date_range=DateRange(start=date(2024, 6, 1), end=date(2024, 6, 1)),
            locations=(LocationSpec(name="x", lat=0.0, lon=0.0),),
            variables=(VariableSpec(
                variable_id="ghi", source=Source.NASA_POWER,
                api_code="ALLSKY_SFC_SW_DWN",
                display_name="GHI", unit="kWh/m^2/day",
                native_unit="kWh/m^2/day", description="x",
            ),),
        )
        with pytest.raises(ValueError, match="plan.source"):
            job.run(plan)


class _RecordingBQForCoverage:
    """Stub that returns canned coverage keys and records the queries.

    Uses 5-tuple keys ``(date, geohash5, product_id, band_id, source)``
    matching :data:`TableSchemas.MODIS_OBSERVATIONS.merge_keys`.
    """

    def __init__(self, existing_keys: set[tuple]) -> None:
        self._keys = existing_keys
        self.coverage_calls: list[dict] = []

    @property
    def config(self):
        from susse.warehouse_ops.io.config import WarehouseConfig
        return WarehouseConfig()

    def existing_keys(self, table_fqn, key_columns, *, where_filters=()):
        self.coverage_calls.append({
            "table_fqn": table_fqn,
            "key_columns": tuple(key_columns),
            "where_filters": tuple(where_filters),
        })
        return self._keys


class TestRunScopesCoverageByLocation:
    """``ModisJob.run`` must pass plan geohashes to the coverage query.

    Without this scope, named-location plans with date ranges overlapping
    the existing warehouse footprint pull millions of irrelevant rows
    back from BigQuery before the API is touched. Same regression-guard
    role as :class:`TestRunScopesCoverageByLocation` on BaseSatelliteJob.
    """

    def test_named_locations_plan_includes_geohash_filter(self) -> None:
        import pygeohash

        loc1 = LocationSpec(name="kampala", lat=0.333542, lon=32.568630)
        loc2 = LocationSpec(name="lira", lat=2.295190, lon=32.921370)
        gh1 = pygeohash.encode(loc1.lat, loc1.lon, precision=5)
        gh2 = pygeohash.encode(loc2.lat, loc2.lon, precision=5)
        date_range = DateRange(start=date(2024, 6, 1), end=date(2024, 6, 16))
        ndvi_var = _modis_var("MOD13Q1_250m_16_days_NDVI", "MOD13Q1")

        # Pre-populate a "warehouse" set that satisfies the cadence-based
        # threshold for both locations, so the run skips fetching.
        full_keys = {
            (date(2024, 6, 1), gh, "MOD13Q1", "250m_16_days_NDVI", "MODIS")
            for gh in (gh1, gh2)
        }
        bq = _RecordingBQForCoverage(full_keys)
        fetcher = _StubFetcher()

        job = ModisJob(bq, fetcher=fetcher)  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.MODIS,
            date_range=date_range,
            locations=(loc1, loc2),
            variables=(ndvi_var,),
        )
        result = job.run(plan)

        assert result.api_calls_made == 0
        assert result.extra["skipped_locations"] == 2
        assert fetcher.calls == [], "fully-cached locations must skip the fetcher"

        assert len(bq.coverage_calls) == 1
        joined = " ".join(bq.coverage_calls[0]["where_filters"])
        assert "geohash5 IN" in joined, (
            f"coverage call missing geohash5 scope: {bq.coverage_calls[0]!r}"
        )
        assert gh1 in joined and gh2 in joined

    def test_run_invokes_fetcher_per_uncached_location(self) -> None:
        # Empty warehouse → every location needs a fetch. Stub fetcher
        # returns an empty frame so we exercise the loop without needing
        # to mock the loader; the assertion is on call count.
        loc1 = LocationSpec(name="a", lat=0.0, lon=10.0)
        loc2 = LocationSpec(name="b", lat=1.0, lon=11.0)
        ndvi_var = _modis_var("MOD13Q1_250m_16_days_NDVI", "MOD13Q1")
        bq = _RecordingBQForCoverage(set())
        fetcher = _StubFetcher()

        job = ModisJob(bq, fetcher=fetcher)  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.MODIS,
            date_range=DateRange(start=date(2024, 6, 1), end=date(2024, 6, 16)),
            locations=(loc1, loc2),
            variables=(ndvi_var,),
        )
        result = job.run(plan)

        assert len(fetcher.calls) == 2
        assert result.api_calls_made == 2
        assert result.extra["skipped_locations"] == 0
        # Both calls must carry the (product, band) pair derived from the var.
        for call in fetcher.calls:
            assert call["products_and_bands"] == (
                ("MOD13Q1", "250m_16_days_NDVI"),
            )

    def test_grid_plan_resolves_locations_from_grid(self) -> None:
        # GridPlan: the job must enumerate via ``grid.iter_locations()`` —
        # a 1×1 grid yields a single point.
        ndvi_var = _modis_var("MOD13Q1_250m_16_days_NDVI", "MOD13Q1")
        grid = GridSpec(
            bbox=BoundingBox(min_lat=0.0, max_lat=0.0, min_lon=10.0, max_lon=10.0),
            lat_step=1.0, lon_step=1.0,
        )
        bq = _RecordingBQForCoverage(set())
        fetcher = _StubFetcher()
        job = ModisJob(bq, fetcher=fetcher)  # type: ignore[arg-type]
        plan = GridPlan(
            source=Source.MODIS,
            date_range=DateRange(start=date(2024, 6, 1), end=date(2024, 6, 16)),
            grid=grid,
            variables=(ndvi_var,),
        )
        result = job.run(plan)
        assert len(fetcher.calls) == 1
        assert result.api_calls_made == 1
