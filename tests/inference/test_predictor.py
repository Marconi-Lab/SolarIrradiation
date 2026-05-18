"""Predictor — happy path, cache-miss surfacing, and request validation.

The Predictor composes :class:`FeatureService` + :class:`Preprocessor`
+ :class:`TrainedBundle` end-to-end. The fixtures train a tiny RF
bundle on synthetic data and monkeypatch
:class:`SatelliteRepository`'s warehouse-query methods to return
canned DataFrames — so the test exercises the orchestration without
ever opening a BigQuery connection.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from susse.datasets import DatasetManifest, FeatureSelection, TrainingDataset
from susse.inference import PredictionRequest, Predictor
from susse.models import RandomForestParams
from susse.preprocessing import FeatureSpec, Preprocessor
from susse.training import SpatialBlockSplitter, Trainer
import pygeohash

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.repositories import SatelliteRepository
from susse.warehouse_ops.population.types import (
    IrradianceBand,
    Source,
    satellite_irradiance_column,
)

# ---------------------------------------------------------------------------
# Synthetic training data + bundle.
# ---------------------------------------------------------------------------

_FEATURE_COLUMNS: tuple[str, ...] = (
    "sat_ghi_nasa_kwh_m2_day",
    "sat_ghi_cams_kwh_m2_day",
    "nasa_temperature",
)
_TARGET_COLUMN: str = "y_ghi_kwh_m2_day"


@pytest.fixture
def fake_selection() -> FeatureSelection:
    """One NASA aux variable + GHI from both satellites — minimal but realistic."""
    return FeatureSelection(
        nasa_variable_ids=("temperature",),
        cams_variable_ids=(),
        include_satellite_irradiance=(Source.NASA_POWER, Source.CAMS),
        include_satellite_bands=(IrradianceBand.GHI,),
        qc_levels=("pass",),
    )


@pytest.fixture
def synthetic_training_frame(fake_selection: FeatureSelection) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    rows: list[dict] = []
    for station, lat, lon, gh in [
        ("sta_a", 0.4, 32.6, "abc12"),
        ("sta_b", -1.0, 36.8, "qrs45"),
    ]:
        for d in pd.date_range("2024-01-01", "2024-02-29", freq="D"):
            sat_nasa = float(rng.uniform(4.0, 7.0))
            sat_cams = sat_nasa + float(rng.normal(0, 0.3))
            temp = float(rng.uniform(20.0, 28.0))
            rows.append(
                {
                    "date": d.date(),
                    "location": station,
                    "geohash5": gh,
                    "lat": lat,
                    "lon": lon,
                    "sat_ghi_nasa_kwh_m2_day": sat_nasa,
                    "sat_ghi_cams_kwh_m2_day": sat_cams,
                    "nasa_temperature": temp,
                    "y_ghi_kwh_m2_day": 0.7 * sat_nasa
                    + 0.05 * temp
                    + rng.normal(0, 0.2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_bundle(
    tmp_path: Path,
    synthetic_training_frame: pd.DataFrame,
    fake_selection: FeatureSelection,
):
    """Tiny RF bundle saved to disk, ready for the Predictor to consume."""
    manifest = DatasetManifest(
        name="predictor_test",
        version="v0",
        created_at_utc=datetime(2026, 5, 12, tzinfo=timezone.utc).isoformat(),
        susse_version="test",
        git_sha=None,
        feature_selection=fake_selection,
        date_start=date(2024, 1, 1),
        date_end=date(2024, 2, 29),
        location_filter=None,
        warehouse_project="test",
        warehouse_dataset="test",
        warehouse_table_mods={},
        n_rows=len(synthetic_training_frame),
        n_cols=len(synthetic_training_frame.columns),
        column_names=tuple(synthetic_training_frame.columns),
        content_hash="predictor_test_hash",
    )
    spec = FeatureSpec(
        target_column=_TARGET_COLUMN,
        feature_columns=_FEATURE_COLUMNS,
        derived_features=(),
        id_columns=("date", "location", "geohash5", "lat", "lon"),
    )
    ds = TrainingDataset(df=synthetic_training_frame, manifest=manifest)
    processed = Preprocessor(spec).apply(ds)

    bundle_dest = tmp_path / "bundle"
    bundle = Trainer().train(
        processed=processed,
        params=RandomForestParams(n_estimators=10, max_depth=4, random_state=0),
        splitter=SpatialBlockSplitter(val_blocks=("sta_b",), block_column="location"),
        holdout_label="sta_b holdout",
        bundle_dest=bundle_dest,
    )
    return bundle


# ---------------------------------------------------------------------------
# Fake warehouse — monkeypatch SatelliteRepository methods.
# ---------------------------------------------------------------------------


def _make_fake_warehouse(
    irradiance_dates: list[date],
    geohash5s: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build canned irradiance + NASA-aux frames covering (geohash5 × date)."""
    rng = np.random.default_rng(seed=11)
    irr_rows: list[dict] = []
    aux_rows: list[dict] = []
    for gh in geohash5s:
        for d in irradiance_dates:
            sat_nasa = float(rng.uniform(4.5, 6.5))
            sat_cams = sat_nasa + float(rng.normal(0, 0.25))
            irr_rows.append(
                {
                    "date": d,
                    "geohash5": gh,
                    "sat_ghi_nasa_kwh_m2_day": sat_nasa,
                    "sat_ghi_cams_kwh_m2_day": sat_cams,
                }
            )
            aux_rows.append(
                {
                    "date": d,
                    "geohash5": gh,
                    "nasa_temperature": float(rng.uniform(22.0, 26.0)),
                }
            )
    return pd.DataFrame(irr_rows), pd.DataFrame(aux_rows)


@pytest.fixture
def patched_satellite_repo(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch SatelliteRepository so the Predictor pulls canned frames.

    Returns a setter the test can use to control the response.
    """
    state: dict[str, pd.DataFrame] = {
        "irradiance": pd.DataFrame(),
        "aux": pd.DataFrame(),
    }

    def fake_irradiance(self, start, end, *, sources, bands, geohash5s):
        # The real method projects one sat_<band>_<source> column per
        # requested source; FeatureService now calls it once per source,
        # so the fake must honour the `sources` filter too.
        df = state["irradiance"]
        if df.empty:
            return df.copy()
        keep = ["date", "geohash5"]
        for band in bands:
            for src in sources:
                col = satellite_irradiance_column(Source(src), band)
                if col in df.columns:
                    keep.append(col)
        return df[keep].copy()

    def fake_aux(
        self, *, table_fqn, column_prefix, start, end, variable_ids, geohash5s
    ):
        return state["aux"].copy()

    def fake_native_pixels(self, *, table_fqn, source=None):
        # The native-grid snapper is built from the warehouse's own pixels.
        # Derive one pixel per fake-warehouse geohash5, placed at the cell
        # centre, so a query coord inside that cell snaps back to it.
        df = state["irradiance"]
        cols = ["geohash5", "latitude", "longitude"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        for gh in df["geohash5"].unique():
            lat, lon = pygeohash.decode(gh)
            rows.append({"geohash5": gh, "latitude": lat, "longitude": lon})
        return pd.DataFrame(rows, columns=cols)

    monkeypatch.setattr(
        SatelliteRepository, "daily_irradiance_by_geohash", fake_irradiance
    )
    monkeypatch.setattr(SatelliteRepository, "long_aux_pivoted", fake_aux)
    monkeypatch.setattr(SatelliteRepository, "native_pixels", fake_native_pixels)

    def set_response(*, irradiance: pd.DataFrame, aux: pd.DataFrame) -> None:
        state["irradiance"] = irradiance
        state["aux"] = aux

    return set_response


@pytest.fixture
def fake_bq() -> MagicMock:
    """A MagicMock that satisfies the BigQueryClient interface superficially.

    FeatureService.__init__ accesses ``bq.config`` to derive TableRefs;
    everything else is mediated through the patched SatelliteRepository.
    """
    bq = MagicMock(spec=BigQueryClient)
    bq.config.project_id = "test-project"
    bq.config.dataset = "test_warehouse"
    return bq


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class TestPredictionRequest:
    def test_empty_coords_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            PredictionRequest(
                coords=(), start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)
            )

    def test_latitude_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Latitude"):
            PredictionRequest(
                coords=((200.0, 0.0),),
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 2),
            )

    def test_swapped_dates_raise_with_remediation(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            PredictionRequest(
                coords=((0.5, 33.0),),
                start_date=date(2024, 2, 1),
                end_date=date(2024, 1, 1),
            )


class TestPredictHappyPath:
    def test_single_coord_returns_one_row_per_date(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        patched_satellite_repo,
    ) -> None:
        start, end = date(2024, 1, 1), date(2024, 1, 7)  # 7 days
        irr, aux = _make_fake_warehouse(
            list(pd.date_range(start, end, freq="D").date),
            geohash5s=("s8p1v",),
        )
        patched_satellite_repo(irradiance=irr, aux=aux)

        predictor = Predictor(bundle=trained_bundle, bq=fake_bq)
        result = predictor.predict(
            coords=[(0.333542, 32.56863)],  # → s8p1v
            start_date=start,
            end_date=end,
        )
        assert len(result) == 7
        assert "y_pred_kwh_m2_day" in result.columns
        assert set(["lat", "lon", "geohash5", "date"]).issubset(result.columns)

    def test_multiple_coords_share_a_geohash_correctly(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        patched_satellite_repo,
    ) -> None:
        # Two coords; resolve to distinct geohash5s.
        start, end = date(2024, 1, 1), date(2024, 1, 3)
        gh_a = "s8p1v"  # Kampala-ish
        gh_b = "kzdre"  # eastern Kenya
        irr, aux = _make_fake_warehouse(
            list(pd.date_range(start, end, freq="D").date),
            geohash5s=(gh_a, gh_b),
        )
        patched_satellite_repo(irradiance=irr, aux=aux)

        predictor = Predictor(bundle=trained_bundle, bq=fake_bq)
        result = predictor.predict(
            coords=[(0.333542, 32.56863), (-1.491302, 37.052862)],
            start_date=start,
            end_date=end,
        )
        # 2 coords × 3 days = 6 rows.
        assert len(result) == 6
        assert result["y_pred_kwh_m2_day"].notna().all()

    def test_predictions_are_finite_floats(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        patched_satellite_repo,
    ) -> None:
        start, end = date(2024, 1, 1), date(2024, 1, 5)
        irr, aux = _make_fake_warehouse(
            list(pd.date_range(start, end, freq="D").date),
            geohash5s=("s8p1v",),
        )
        patched_satellite_repo(irradiance=irr, aux=aux)
        predictor = Predictor(bundle=trained_bundle, bq=fake_bq)
        result = predictor.predict(
            coords=[(0.333542, 32.56863)],
            start_date=start,
            end_date=end,
        )
        assert np.isfinite(result["y_pred_kwh_m2_day"]).all()


class TestCacheMiss:
    def test_raise_policy_surfaces_missing_dates(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        patched_satellite_repo,
    ) -> None:
        # Warehouse only has 3 days; we request 7. 4 missing.
        irr, aux = _make_fake_warehouse(
            [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            geohash5s=("s8p1v",),
        )
        patched_satellite_repo(irradiance=irr, aux=aux)
        predictor = Predictor(bundle=trained_bundle, bq=fake_bq)
        with pytest.raises(RuntimeError, match="missing"):
            predictor.predict(
                coords=[(0.333542, 32.56863)],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 7),
            )


class TestFetchModeCredentials:
    def test_missing_cams_email_raises_at_construction(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """on_cache_miss='fetch' without CAMS_EMAIL must fail loudly upfront."""
        monkeypatch.delenv("CAMS_EMAIL", raising=False)
        with pytest.raises(ValueError, match="CAMS_EMAIL"):
            Predictor(bundle=trained_bundle, bq=fake_bq, on_cache_miss="fetch")

    def test_present_cams_email_allows_construction(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CAMS_EMAIL", "you@example.com")
        predictor = Predictor(bundle=trained_bundle, bq=fake_bq, on_cache_miss="fetch")
        assert predictor.on_cache_miss == "fetch"


class TestFetchOnDemand:
    """The 'fetch' branch reuses NasaPowerSatelliteJob + CamsSatelliteJob.

    Mock both jobs' .run() — that's the seam between Predictor and the
    warehouse-ingest layer. After the mocked jobs return, we extend the
    fake-warehouse response so the re-query sees the freshly-written
    rows.
    """

    def test_fetch_runs_ingest_jobs_then_re_queries(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        patched_satellite_repo,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CAMS_EMAIL", "you@example.com")
        start, end = date(2024, 1, 1), date(2024, 1, 5)

        # Initial warehouse state: empty. Cache miss for every date.
        irr_full, aux_full = _make_fake_warehouse(
            list(pd.date_range(start, end, freq="D").date),
            geohash5s=("s8p1v",),
        )
        patched_satellite_repo(
            irradiance=pd.DataFrame(columns=irr_full.columns),
            aux=pd.DataFrame(columns=aux_full.columns),
        )

        # Mock the ingest jobs: instead of hitting the API, just refill
        # the patched warehouse so the re-query picks up the new rows.
        nasa_call = {"count": 0}
        cams_call = {"count": 0}

        def fake_nasa_run(self, plan):
            nasa_call["count"] += 1
            patched_satellite_repo(irradiance=irr_full, aux=aux_full)
            return MagicMock(rows_added_long=0, rows_added_irr=0)

        def fake_cams_run(self, plan):
            cams_call["count"] += 1
            return MagicMock(rows_added_long=0, rows_added_irr=0)

        from susse.warehouse_ops.population.jobs.satellite_job import (
            CamsSatelliteJob,
            NasaPowerSatelliteJob,
        )

        monkeypatch.setattr(NasaPowerSatelliteJob, "run", fake_nasa_run)
        monkeypatch.setattr(CamsSatelliteJob, "run", fake_cams_run)

        predictor = Predictor(bundle=trained_bundle, bq=fake_bq, on_cache_miss="fetch")
        result = predictor.predict(
            coords=[(0.333542, 32.56863)],
            start_date=start,
            end_date=end,
        )
        # Both jobs were invoked.
        assert nasa_call["count"] == 1
        assert cams_call["count"] == 1
        # And the result is a complete prediction frame.
        assert len(result) == 5
        assert result["y_pred_kwh_m2_day"].notna().all()

    def test_rate_limit_cap_rejects_huge_miss(
        self,
        trained_bundle,
        fake_bq: MagicMock,
        patched_satellite_repo,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """More than MAX_CAMS_CALLS_PER_PREDICT missing cells must raise."""
        from susse.inference.predictor import MAX_CAMS_CALLS_PER_PREDICT

        monkeypatch.setenv("CAMS_EMAIL", "you@example.com")
        # Build n = cap + 1 coordinates that all miss the cache.
        n = MAX_CAMS_CALLS_PER_PREDICT + 1
        coords = [(0.0 + 0.1 * i, 32.0 + 0.1 * i) for i in range(n)]

        patched_satellite_repo(
            irradiance=pd.DataFrame(
                columns=[
                    "date",
                    "geohash5",
                    "sat_ghi_nasa_kwh_m2_day",
                    "sat_ghi_cams_kwh_m2_day",
                ]
            ),
            aux=pd.DataFrame(columns=["date", "geohash5", "nasa_temperature"]),
        )
        predictor = Predictor(bundle=trained_bundle, bq=fake_bq, on_cache_miss="fetch")
        with pytest.raises(RuntimeError, match="exceeding the per-invocation cap"):
            predictor.predict(
                coords=coords,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
            )
