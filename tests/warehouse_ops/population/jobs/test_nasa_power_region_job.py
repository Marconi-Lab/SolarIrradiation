"""Tests for NasaPowerRegionJob — region-shaped NASA POWER ingest.

End-to-end ``run()`` against the live NASA POWER endpoint is exercised by
the a13 migration. These tests cover the contracts that don't need a
network round-trip: plan validation, the single fetch_region_long fan-out,
api_code → variable_id remapping with native-pixel geohash5 enrichment, the
irradiance/aux split, and the long→wide irradiance pivot.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pygeohash
import pytest

from susse.warehouse_ops.io.config import WarehouseConfig
from susse.warehouse_ops.population.jobs.nasa_power_region_job import NasaPowerRegionJob
from susse.warehouse_ops.population.types import (
    BoundingBox,
    DateRange,
    LocationSpec,
    NamedLocationsPlan,
    RegionPlan,
    Source,
    VariableSpec,
)


def _nasa_var(variable_id: str, api_code: str) -> VariableSpec:
    return VariableSpec(
        variable_id=variable_id,
        source=Source.NASA_POWER,
        api_code=api_code,
        display_name=variable_id,
        unit="x",
        native_unit="x",
        description="x",
    )


_GHI = _nasa_var("ghi", "ALLSKY_SFC_SW_DWN")
_DHI = _nasa_var("dhi", "ALLSKY_SFC_SW_DIFF")
_T2M = _nasa_var("temperature", "T2M")

_UGANDA_BBOX = BoundingBox(min_lat=-1.5, max_lat=4.5, min_lon=29.5, max_lon=35.05)
_DATES = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 2))


class _StubRegionalFetcher:
    """Records fetch_region_long calls and returns a canned native-pixel frame.

    Rows are keyed by ``api_code`` (the real fetcher's contract); the job
    maps these to warehouse ``variable_id``.
    """

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.calls: list[dict] = []
        self._rows = rows if rows is not None else []

    def fetch_region_long(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(kwargs)
        return pd.DataFrame(
            self._rows,
            columns=("date", "latitude", "longitude", "variable_id", "value"),
        )


class _FakeBQ:
    """Minimal BigQuery double: captures every staged DataFrame load.

    The MERGE DDL is a no-op; the MIN/MAX date probe returns a fixed span so
    MergeLoader runs a single un-chunked MERGE. ``MergeLoader.load`` then
    returns ``len(df)``, so the job's row counts reflect the split exactly.
    """

    def __init__(self) -> None:
        self.config = WarehouseConfig()
        self.loaded: list[tuple[str, pd.DataFrame]] = []

    def load_dataframe(
        self, df: pd.DataFrame, table_fqn: str, *, write_disposition: str
    ) -> None:
        self.loaded.append((table_fqn, df.copy()))

    def execute_ddl(self, sql: str) -> None:
        return None

    def query(self, sql: str) -> pd.DataFrame:
        return pd.DataFrame([{"min_d": date(2024, 1, 1), "max_d": date(2024, 1, 2)}])


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------


class TestRunRejectsBadPlan:
    def test_raises_on_wrong_plan_type(self) -> None:
        # A NamedLocationsPlan is a valid FetchPlan but the wrong shape for
        # the region job — it must be rejected, not silently mis-handled.
        job = NasaPowerRegionJob(_FakeBQ(), fetcher=_StubRegionalFetcher())  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.NASA_POWER,
            date_range=_DATES,
            locations=(LocationSpec(name="kampala", lat=0.33, lon=32.57),),
            variables=(_GHI,),
        )
        with pytest.raises(TypeError, match="RegionPlan"):
            job.run(plan)

    def test_raises_on_source_mismatch(self) -> None:
        job = NasaPowerRegionJob(_FakeBQ(), fetcher=_StubRegionalFetcher())  # type: ignore[arg-type]
        cams_var = VariableSpec(
            variable_id="ghi",
            source=Source.CAMS,
            api_code="GHI",
            display_name="GHI",
            unit="x",
            native_unit="x",
            description="x",
        )
        plan = RegionPlan(
            source=Source.CAMS,
            date_range=_DATES,
            bbox=_UGANDA_BBOX,
            variables=(cams_var,),
        )
        with pytest.raises(ValueError, match="plan.source"):
            job.run(plan)


# ---------------------------------------------------------------------------
# Single-call fan-out shape
# ---------------------------------------------------------------------------


class TestRunIssuesOneFetchCall:
    def test_one_fetch_region_long_call_with_bbox_and_all_api_codes(self) -> None:
        fetcher = _StubRegionalFetcher()
        job = NasaPowerRegionJob(_FakeBQ(), fetcher=fetcher)  # type: ignore[arg-type]
        plan = RegionPlan(
            source=Source.NASA_POWER,
            date_range=_DATES,
            bbox=_UGANDA_BBOX,
            variables=(_GHI, _T2M),
        )
        result = job.run(plan)

        assert len(fetcher.calls) == 1, (
            "NasaPowerRegionJob must issue exactly one fetch_region_long call "
            "for the whole bbox — the fetcher fans out into tiles internally."
        )
        call = fetcher.calls[0]
        assert call["min_lat"] == _UGANDA_BBOX.min_lat
        assert call["max_lon"] == _UGANDA_BBOX.max_lon
        assert call["date_start"] == date(2024, 1, 1)
        assert set(call["api_codes"]) == {"ALLSKY_SFC_SW_DWN", "T2M"}
        assert result.api_calls_made == 1
        # Empty fetch → nothing loaded, no rows added.
        assert result.rows_added == 0


# ---------------------------------------------------------------------------
# Enrichment: api_code → variable_id, native-pixel geohash5
# ---------------------------------------------------------------------------


class TestEnrich:
    def test_remaps_api_code_and_geohashes_each_native_pixel(self) -> None:
        job = NasaPowerRegionJob(_FakeBQ(), fetcher=_StubRegionalFetcher())  # type: ignore[arg-type]
        rows = [
            {
                "date": date(2024, 1, 1),
                "latitude": 0.5,
                "longitude": 32.5,
                "variable_id": "T2M",
                "value": 25.0,
            },
            {
                "date": date(2024, 1, 1),
                "latitude": 1.0,
                "longitude": 33.125,
                "variable_id": "T2M",
                "value": 26.0,
            },
        ]
        df = pd.DataFrame(rows)
        enriched = job._enrich(df, {"T2M": "temperature"})

        assert list(enriched["variable_id"]) == ["temperature", "temperature"]
        assert list(enriched["source"]) == ["NASA", "NASA"]
        # geohash5 is the native pixel centre's geohash — not a query point.
        assert enriched.iloc[0]["geohash5"] == pygeohash.encode(0.5, 32.5, 5)
        assert enriched.iloc[1]["geohash5"] == pygeohash.encode(1.0, 33.125, 5)

    def test_drops_unmapped_api_codes(self) -> None:
        # An api_code with no catalog mapping must be dropped, not stored
        # under a NaN variable_id.
        job = NasaPowerRegionJob(_FakeBQ(), fetcher=_StubRegionalFetcher())  # type: ignore[arg-type]
        df = pd.DataFrame(
            [
                {
                    "date": date(2024, 1, 1),
                    "latitude": 0.5,
                    "longitude": 32.5,
                    "variable_id": "UNKNOWN_CODE",
                    "value": 1.0,
                }
            ]
        )
        assert job._enrich(df, {"T2M": "temperature"}).empty


# ---------------------------------------------------------------------------
# Full run: irradiance / aux split + load routing
# ---------------------------------------------------------------------------


class TestRunSplitsIrradianceFromAux:
    def test_irradiance_and_aux_routed_to_their_tables(self) -> None:
        # ghi + dhi at one pixel over 2 days → 4 long rows → 2 wide rows.
        # T2M at two pixels over 2 days → 4 long aux rows.
        rows: list[dict] = []
        for day in (date(2024, 1, 1), date(2024, 1, 2)):
            rows.append(
                {
                    "date": day,
                    "latitude": 0.5,
                    "longitude": 32.5,
                    "variable_id": "ALLSKY_SFC_SW_DWN",
                    "value": 5.0,
                }
            )
            rows.append(
                {
                    "date": day,
                    "latitude": 0.5,
                    "longitude": 32.5,
                    "variable_id": "ALLSKY_SFC_SW_DIFF",
                    "value": 2.0,
                }
            )
            rows.append(
                {
                    "date": day,
                    "latitude": 0.5,
                    "longitude": 32.5,
                    "variable_id": "T2M",
                    "value": 25.0,
                }
            )
            rows.append(
                {
                    "date": day,
                    "latitude": 1.0,
                    "longitude": 33.125,
                    "variable_id": "T2M",
                    "value": 26.0,
                }
            )

        bq = _FakeBQ()
        job = NasaPowerRegionJob(bq, fetcher=_StubRegionalFetcher(rows))  # type: ignore[arg-type]
        plan = RegionPlan(
            source=Source.NASA_POWER,
            date_range=_DATES,
            bbox=_UGANDA_BBOX,
            variables=(_GHI, _DHI, _T2M),
        )
        result = job.run(plan)

        assert result.extra["rows_added_irradiance"] == 2
        assert result.extra["rows_added_long"] == 4
        assert result.rows_added == 6

        staged = {fqn: df for fqn, df in bq.loaded}
        irr_fqn = next(f for f in staged if "irradiance_daily_staging" in f)
        aux_fqn = next(f for f in staged if "nasa_daily_vars_long_staging" in f)
        assert set(staged[irr_fqn]["geohash5"]) == {pygeohash.encode(0.5, 32.5, 5)}
        assert list(staged[aux_fqn]["variable_id"].unique()) == ["temperature"]
