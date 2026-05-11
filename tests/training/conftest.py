"""Shared fixtures for training-package tests.

Provides a small, self-consistent :class:`PreprocessedDataset` with
two stations and a couple of months of synthetic-but-plausible data
so every test in this directory can construct a Trainer end-to-end
without a live warehouse.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from susse.datasets import DatasetManifest, FeatureSelection
from susse.preprocessing import FeatureSpec, PreprocessedDataset


@pytest.fixture
def toy_manifest() -> DatasetManifest:
    return DatasetManifest(
        name="toy_train",
        version="v0",
        created_at_utc=datetime(2026, 5, 10, tzinfo=timezone.utc).isoformat(),
        susse_version="test",
        git_sha=None,
        feature_selection=FeatureSelection(),
        date_start=date(2024, 1, 1),
        date_end=date(2024, 2, 29),
        location_filter=None,
        warehouse_project="test",
        warehouse_dataset="test",
        warehouse_table_mods={},
        n_rows=120,
        n_cols=8,
        column_names=(
            "date",
            "location",
            "geohash5",
            "y_ghi_kwh_m2_day",
            "feat_a",
            "feat_b",
            "sat_ghi_nasa_kwh_m2_day",
            "sat_ghi_cams_kwh_m2_day",
        ),
        content_hash="toy_hash_for_tests",
    )


@pytest.fixture
def toy_processed(toy_manifest: DatasetManifest) -> PreprocessedDataset:
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []
    for station in ("sta_a", "sta_b"):
        # Different intercept per station so station-LOSO has signal.
        intercept = 5.0 if station == "sta_a" else 4.0
        for d in pd.date_range("2024-01-01", "2024-02-29", freq="D"):
            feat_a = float(rng.uniform(0, 1))
            feat_b = float(rng.uniform(-1, 1))
            y = intercept + 1.5 * feat_a - 0.3 * feat_b + rng.normal(0, 0.2)
            rows.append(
                {
                    "date": d.date(),
                    "location": station,
                    "geohash5": "abc12",
                    "y_ghi_kwh_m2_day": y,
                    "feat_a": feat_a,
                    "feat_b": feat_b,
                    # Satellite "estimate" with a small fixed bias so the
                    # baseline-scoring path has a meaningful signal.
                    "sat_ghi_nasa_kwh_m2_day": y + 0.7,
                    "sat_ghi_cams_kwh_m2_day": y + 0.4,
                }
            )
    df = pd.DataFrame(rows).reset_index(drop=True)
    spec = FeatureSpec(
        target_column="y_ghi_kwh_m2_day",
        feature_columns=("feat_a", "feat_b"),
        derived_features=(),
        id_columns=("date", "location", "geohash5"),
    )
    return PreprocessedDataset(
        df=df,
        feature_columns=spec.output_feature_names,
        target_column=spec.target_column,
        feature_spec=spec,
        source_manifest=toy_manifest,
    )
