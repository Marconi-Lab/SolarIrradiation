"""Tests for the Preprocessor: TrainingDataset → PreprocessedDataset."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from susse.datasets import DatasetManifest, FeatureSelection, TrainingDataset
from susse.preprocessing import (
    ClearSkyIndexSpec,
    FeatureSpec,
    Preprocessor,
)


def _toy_dataset() -> TrainingDataset:
    df = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "location": ["a", "a", "a"],
        "geohash5": ["s8p1v", "s8p1v", "s8p1v"],
        "y_ghi_kwh_m2_day": [5.0, 6.0, 4.0],
        "sat_ghi_nasa_kwh_m2_day": [4.5, 5.5, 3.5],
        "nasa_ghi_clear": [6.0, 6.0, 6.0],
        "nasa_aod_550": [0.3, 0.25, 0.5],
    })
    manifest = DatasetManifest(
        name="toy", version="v0",
        created_at_utc=datetime(2026, 5, 9, tzinfo=timezone.utc).isoformat(),
        susse_version="test", git_sha=None,
        feature_selection=FeatureSelection(),
        date_start=date(2024, 1, 1), date_end=date(2024, 1, 3),
        location_filter=None,
        warehouse_project="test", warehouse_dataset="test",
        warehouse_table_mods={},
        n_rows=3, n_cols=len(df.columns),
        column_names=tuple(df.columns),
        content_hash="dummy_hash",
    )
    return TrainingDataset(df=df, manifest=manifest)


class TestApplyHappyPath:
    def test_kt_and_doy_features_appear_in_output(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            clear_sky_index_specs=(
                ClearSkyIndexSpec(
                    ghi_column="sat_ghi_nasa_kwh_m2_day",
                    ghi_clear_column="nasa_ghi_clear",
                    output_column="kt_nasa",
                ),
            ),
            include_cyclical_doy=True,
        )
        result = Preprocessor(spec).apply(_toy_dataset())
        assert result.feature_columns == (
            "nasa_aod_550", "kt_nasa", "doy_sin", "doy_cos",
        )
        # kt = 4.5/6.0 = 0.75 for the first row.
        assert result.df["kt_nasa"].iloc[0] == pytest.approx(0.75)
        assert result.target_column == "y_ghi_kwh_m2_day"
        # Lineage to source dataset is preserved.
        assert result.source_dataset_name == "toy"
        assert result.source_content_hash == "dummy_hash"

    def test_X_and_y_helpers_return_correct_slices(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            include_cyclical_doy=False,
        )
        result = Preprocessor(spec).apply(_toy_dataset())
        assert list(result.X().columns) == ["nasa_aod_550"]
        np.testing.assert_allclose(result.y().values, [5.0, 6.0, 4.0])


class TestNanHandling:
    def test_dropna_target_removes_rows_with_missing_y(self) -> None:
        ds = _toy_dataset()
        ds.df.loc[1, "y_ghi_kwh_m2_day"] = float("nan")
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            include_cyclical_doy=False,
            dropna_target=True,
        )
        result = Preprocessor(spec).apply(ds)
        assert result.n_rows == 2  # 3 input rows, 1 dropped

    def test_dropna_features_removes_rows_with_missing_feature(self) -> None:
        ds = _toy_dataset()
        ds.df.loc[2, "nasa_aod_550"] = float("nan")
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            include_cyclical_doy=False,
            dropna_target=False,
            dropna_features=True,
        )
        result = Preprocessor(spec).apply(ds)
        assert result.n_rows == 2

    def test_dropna_target_false_keeps_missing_target_rows(self) -> None:
        ds = _toy_dataset()
        ds.df.loc[0, "y_ghi_kwh_m2_day"] = float("nan")
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            include_cyclical_doy=False,
            dropna_target=False,
        )
        result = Preprocessor(spec).apply(ds)
        assert result.n_rows == 3
        assert pd.isna(result.df["y_ghi_kwh_m2_day"].iloc[0])


class TestColumnValidation:
    """Missing required columns surface as a single, named ValueError."""

    def test_missing_feature_column_raises(self) -> None:
        spec = FeatureSpec(
            feature_columns=("not_a_real_column",),
            include_cyclical_doy=False,
        )
        with pytest.raises(ValueError, match="not_a_real_column"):
            Preprocessor(spec).apply(_toy_dataset())

    def test_missing_kt_input_raises(self) -> None:
        spec = FeatureSpec(
            clear_sky_index_specs=(
                ClearSkyIndexSpec(
                    ghi_column="missing_ghi",
                    ghi_clear_column="missing_clear",
                    output_column="kt",
                ),
            ),
            include_cyclical_doy=False,
        )
        with pytest.raises(ValueError, match="missing_ghi"):
            Preprocessor(spec).apply(_toy_dataset())

    def test_missing_target_with_dropna_raises(self) -> None:
        ds = _toy_dataset()
        ds.df.drop(columns=["y_ghi_kwh_m2_day"], inplace=True)
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            include_cyclical_doy=False,
            dropna_target=True,
        )
        with pytest.raises(ValueError, match="y_ghi_kwh_m2_day"):
            Preprocessor(spec).apply(ds)


class TestStateless:
    """Re-applying the same preprocessor must produce identical output.

    Pins the design choice that no state leaks across applications.
    """

    def test_two_applications_produce_equal_output(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            clear_sky_index_specs=(
                ClearSkyIndexSpec(
                    ghi_column="sat_ghi_nasa_kwh_m2_day",
                    ghi_clear_column="nasa_ghi_clear",
                    output_column="kt_nasa",
                ),
            ),
            include_cyclical_doy=True,
        )
        pre = Preprocessor(spec)
        a = pre.apply(_toy_dataset())
        b = pre.apply(_toy_dataset())
        pd.testing.assert_frame_equal(a.df, b.df)
