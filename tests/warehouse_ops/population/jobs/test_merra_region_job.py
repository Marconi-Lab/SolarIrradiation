"""Tests for MerraRegionJob — region-shaped MERRA-2 ingest.

End-to-end ``run()`` against the live GES DISC OPeNDAP endpoint is
exercised by the b9 migration. These tests cover the contracts that don't
require a network round-trip: plan validation, the single-fetch_region
fan-out shape, api_code → variable_id remapping, geohash5 enrichment,
and the existing-keys filter that spares re-runs from re-uploading rows.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from susse.warehouse_ops.population.jobs.merra_region_job import MerraRegionJob
from susse.warehouse_ops.population.types import (
    BoundingBox,
    DateRange,
    GridPlan,
    GridSpec,
    LocationSpec,
    NamedLocationsPlan,
    Source,
    VariableSpec,
)


def _merra_var(variable_id: str, api_code: str) -> VariableSpec:
    return VariableSpec(
        variable_id=variable_id, source=Source.MERRA_2, api_code=api_code,
        display_name=variable_id, unit="x", native_unit="x", description="x",
    )


class _StubBQ:
    @property
    def config(self):
        from susse.warehouse_ops.io.config import WarehouseConfig
        return WarehouseConfig()


class _StubFetcher:
    """Stub fetcher that records calls and returns a canned region frame.

    Returns rows keyed by ``api_code`` (matching the real fetcher contract);
    the job is responsible for mapping these to warehouse ``variable_id``.
    """

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.calls: list[dict] = []
        self._rows = rows if rows is not None else []

    def fetch_region(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        return pd.DataFrame(
            self._rows,
            columns=("date", "latitude", "longitude", "variable_id", "value"),
        )


class _RecordingBQForCoverage:
    """Stub that returns canned coverage keys and records the queries.

    Uses 3-tuple keys ``(date, geohash5, variable_id)`` matching
    :data:`TableSchemas.MERRA_DAILY_VARS_LONG.merge_keys` after the
    source/scoping filters are applied by the coverage layer.
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


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------


class TestRunRejectsBadPlan:
    def test_raises_on_wrong_plan_type(self) -> None:
        job = MerraRegionJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]

        class FakePlan:
            def describe(self) -> str:
                return "fake"

        with pytest.raises(TypeError, match="NamedLocationsPlan or GridPlan"):
            job.run(FakePlan())  # type: ignore[arg-type]

    def test_raises_on_source_mismatch(self) -> None:
        # Using a NASA-source plan against the MERRA job must surface a
        # clear contract violation rather than silently fetching the wrong
        # variables.
        job = MerraRegionJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.NASA_POWER,
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


# ---------------------------------------------------------------------------
# Single-call fan-out shape
# ---------------------------------------------------------------------------


class TestRunIssuesOneFetchRegionCall:
    """The whole point of MerraRegionJob is *one* fetch_region call per
    plan, regardless of location count. Pinning that shape protects the
    100× speedup over the legacy per-point pattern."""

    def test_named_locations_plan_calls_fetcher_once(self) -> None:
        loc1 = LocationSpec(name="kampala", lat=0.333542, lon=32.568630)
        loc2 = LocationSpec(name="lira", lat=2.295190, lon=32.921370)
        bq = _RecordingBQForCoverage(set())
        fetcher = _StubFetcher()
        job = MerraRegionJob(bq, fetcher=fetcher)  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.MERRA_2,
            date_range=DateRange(start=date(2024, 6, 1), end=date(2024, 6, 16)),
            locations=(loc1, loc2),
            variables=(_merra_var("aod_550_extinction", "TOTEXTTAU"),),
        )
        result = job.run(plan)

        assert len(fetcher.calls) == 1, (
            "MerraRegionJob must issue exactly one fetch_region call covering "
            "all locations — not one per location."
        )
        call = fetcher.calls[0]
        assert call["points"] == ((loc1.lat, loc1.lon), (loc2.lat, loc2.lon))
        assert call["api_codes"] == ("TOTEXTTAU",)
        assert call["date_start"] == date(2024, 6, 1)
        assert call["date_end"] == date(2024, 6, 16)
        assert result.api_calls_made == 1

    def test_grid_plan_iterates_grid_points(self) -> None:
        # GridPlan: the job must enumerate via ``grid.iter_locations()``
        # and pass all grid points into a single fetch_region call.
        var = _merra_var("aod_550_extinction", "TOTEXTTAU")
        grid = GridSpec(
            bbox=BoundingBox(min_lat=0.0, max_lat=1.0, min_lon=10.0, max_lon=11.0),
            lat_step=0.5, lon_step=0.5,
        )
        bq = _RecordingBQForCoverage(set())
        fetcher = _StubFetcher()
        job = MerraRegionJob(bq, fetcher=fetcher)  # type: ignore[arg-type]
        plan = GridPlan(
            source=Source.MERRA_2,
            date_range=DateRange(start=date(2024, 6, 1), end=date(2024, 6, 16)),
            grid=grid,
            variables=(var,),
        )
        job.run(plan)
        assert len(fetcher.calls) == 1
        # 3 lat × 3 lon = 9 grid points.
        assert len(fetcher.calls[0]["points"]) == 9


# ---------------------------------------------------------------------------
# Enrichment: api_code → variable_id, geohash5 attachment
# ---------------------------------------------------------------------------


class TestEnrichRemapsAndGeohashes:
    """The fetcher returns rows keyed by api_code; the job must remap to
    the warehouse variable_id and attach the geohash5 derived from each
    row's (lat, lon). Test these in isolation from coverage / load."""

    def test_remaps_api_code_and_attaches_geohash(self) -> None:
        import pygeohash

        loc = LocationSpec(name="kampala", lat=0.333542, lon=32.568630)
        expected_gh = pygeohash.encode(loc.lat, loc.lon, precision=5)
        var = _merra_var("aod_550_extinction", "TOTEXTTAU")

        fetcher_rows = [
            {"date": date(2024, 6, 21), "latitude": loc.lat, "longitude": loc.lon,
             "variable_id": "TOTEXTTAU", "value": 0.42},
        ]
        job = MerraRegionJob(_StubBQ(), fetcher=_StubFetcher(fetcher_rows))  # type: ignore[arg-type]

        df = pd.DataFrame(fetcher_rows)
        api_to_var = {var.api_code: var.variable_id}
        enriched = job._enrich(df, [loc], api_to_var)

        assert list(enriched["variable_id"]) == ["aod_550_extinction"]
        assert list(enriched["geohash5"]) == [expected_gh]
        assert list(enriched["source"]) == ["MERRA2"]

    def test_drops_rows_for_unknown_api_codes(self) -> None:
        # If the fetcher returns an api_code we didn't ask for (shouldn't
        # happen in practice but defensive), the row must be dropped
        # rather than written with a NaN variable_id.
        loc = LocationSpec(name="kampala", lat=0.333, lon=32.568)
        var = _merra_var("aod_550_extinction", "TOTEXTTAU")
        df = pd.DataFrame([
            {"date": date(2024, 6, 21), "latitude": loc.lat, "longitude": loc.lon,
             "variable_id": "TOTEXTTAU", "value": 0.42},
            {"date": date(2024, 6, 21), "latitude": loc.lat, "longitude": loc.lon,
             "variable_id": "GHOST_VAR", "value": 999.0},
        ])
        job = MerraRegionJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]
        enriched = job._enrich(df, [loc], {var.api_code: var.variable_id})

        assert list(enriched["variable_id"]) == ["aod_550_extinction"]
        assert 999.0 not in enriched["value"].values

    def test_raises_when_fetched_point_not_in_plan(self) -> None:
        # If somehow a row's (lat, lon) doesn't match any plan location,
        # the geohash merge would yield NaN. The job must surface the
        # invariant violation rather than silently writing NaN geohashes.
        loc = LocationSpec(name="kampala", lat=0.333, lon=32.568)
        var = _merra_var("aod_550_extinction", "TOTEXTTAU")
        df = pd.DataFrame([
            {"date": date(2024, 6, 21), "latitude": 99.0, "longitude": 99.0,
             "variable_id": "TOTEXTTAU", "value": 0.42},
        ])
        job = MerraRegionJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="out of sync"):
            job._enrich(df, [loc], {var.api_code: var.variable_id})


# ---------------------------------------------------------------------------
# Coverage filter: re-run optimization
# ---------------------------------------------------------------------------


class TestDropAlreadyCached:
    """Filter step that spares re-runs from re-uploading rows. The
    MergeLoader is idempotent on the merge keys so this is purely an
    optimization, but it's an important one for repeated runs over a
    largely-already-populated date range."""

    def test_drops_rows_in_existing_keys(self) -> None:
        df = pd.DataFrame([
            {"date": date(2024, 6, 1), "geohash5": "abc12",
             "variable_id": "v1", "value": 1.0},
            {"date": date(2024, 6, 2), "geohash5": "abc12",
             "variable_id": "v1", "value": 2.0},
        ])
        existing = {(date(2024, 6, 1), "abc12", "v1")}
        out = MerraRegionJob._drop_already_cached(df, existing)
        assert len(out) == 1
        assert out.iloc[0]["date"] == date(2024, 6, 2)

    def test_returns_input_when_existing_set_empty(self) -> None:
        df = pd.DataFrame([
            {"date": date(2024, 6, 1), "geohash5": "abc12",
             "variable_id": "v1", "value": 1.0},
        ])
        out = MerraRegionJob._drop_already_cached(df, set())
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Coverage scoping: same regression-guard role as ModisJob
# ---------------------------------------------------------------------------


class TestRunScopesCoverageByLocation:
    """``MerraRegionJob.run`` must pass plan geohashes to the coverage
    query. Without this scope, named-location plans whose date range
    overlaps the existing warehouse footprint pull millions of irrelevant
    rows back from BigQuery before the API is touched."""

    def test_named_locations_plan_includes_geohash_filter(self) -> None:
        import pygeohash

        loc1 = LocationSpec(name="kampala", lat=0.333542, lon=32.568630)
        loc2 = LocationSpec(name="lira", lat=2.295190, lon=32.921370)
        gh1 = pygeohash.encode(loc1.lat, loc1.lon, precision=5)
        gh2 = pygeohash.encode(loc2.lat, loc2.lon, precision=5)
        bq = _RecordingBQForCoverage(set())
        fetcher = _StubFetcher()
        job = MerraRegionJob(bq, fetcher=fetcher)  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.MERRA_2,
            date_range=DateRange(start=date(2024, 6, 1), end=date(2024, 6, 1)),
            locations=(loc1, loc2),
            variables=(_merra_var("aod_550_extinction", "TOTEXTTAU"),),
        )
        job.run(plan)

        assert len(bq.coverage_calls) == 1
        joined = " ".join(bq.coverage_calls[0]["where_filters"])
        assert "geohash5 IN" in joined, (
            f"coverage call missing geohash5 scope: {bq.coverage_calls[0]!r}"
        )
        assert gh1 in joined and gh2 in joined
