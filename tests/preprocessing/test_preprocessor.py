"""Tests for the Preprocessor: TrainingDataset → PreprocessedDataset."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from susse.datasets import DatasetManifest, FeatureSelection, TrainingDataset
from susse.preprocessing import (
    AltitudeFeature,
    ClearSkyIndexFeature,
    CyclicalDayOfYearFeature,
    DataCleaner,
    FeatureSpec,
    GhiUpperBoundCleaner,
    Preprocessor,
)
from susse.preprocessing.cleaning import CleanerKind


def _toy_dataset(*, with_coords: bool = False) -> TrainingDataset:
    df = pd.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "location": ["a", "a", "a"],
            "geohash5": ["s8p1v", "s8p1v", "s8p1v"],
            "y_ghi_kwh_m2_day": [5.0, 6.0, 4.0],
            "sat_ghi_nasa_kwh_m2_day": [4.5, 5.5, 3.5],
            "nasa_ghi_clear": [6.0, 6.0, 6.0],
            "nasa_aod_550": [0.3, 0.25, 0.5],
        }
    )
    if with_coords:
        df["lat"] = 0.5179
        df["lon"] = 32.4715
    manifest = DatasetManifest(
        name="toy",
        version="v0",
        created_at_utc=datetime(2026, 5, 9, tzinfo=timezone.utc).isoformat(),
        susse_version="test",
        git_sha=None,
        feature_selection=FeatureSelection(),
        date_start=date(2024, 1, 1),
        date_end=date(2024, 1, 3),
        location_filter=None,
        warehouse_project="test",
        warehouse_dataset="test",
        warehouse_table_mods={},
        n_rows=3,
        n_cols=len(df.columns),
        column_names=tuple(df.columns),
        content_hash="dummy_hash",
    )
    return TrainingDataset(df=df, manifest=manifest)


def _kt() -> ClearSkyIndexFeature:
    return ClearSkyIndexFeature(
        ghi_column="sat_ghi_nasa_kwh_m2_day",
        ghi_clear_column="nasa_ghi_clear",
        output_column="kt_nasa",
    )


class TestApplyHappyPath:
    def test_kt_and_doy_features_appear_in_output(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            derived_features=(_kt(), CyclicalDayOfYearFeature()),
        )
        result = Preprocessor(spec).apply(_toy_dataset())
        assert result.feature_columns == (
            "nasa_aod_550",
            "kt_nasa",
            "doy_sin",
            "doy_cos",
        )
        # kt = 4.5/6.0 = 0.75 for the first row.
        assert result.df["kt_nasa"].iloc[0] == pytest.approx(0.75)
        assert result.target_column == "y_ghi_kwh_m2_day"
        # Lineage to source dataset is preserved.
        assert result.source_dataset_name == "toy"
        assert result.source_content_hash == "dummy_hash"

    def test_X_and_y_helpers_return_correct_slices(self) -> None:
        spec = FeatureSpec(feature_columns=("nasa_aod_550",))
        result = Preprocessor(spec).apply(_toy_dataset())
        assert list(result.X().columns) == ["nasa_aod_550"]
        np.testing.assert_allclose(result.y().values, [5.0, 6.0, 4.0])


class TestNanHandling:
    def test_dropna_target_removes_rows_with_missing_y(self) -> None:
        ds = _toy_dataset()
        ds.df.loc[1, "y_ghi_kwh_m2_day"] = float("nan")
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            dropna_target=True,
        )
        result = Preprocessor(spec).apply(ds)
        assert result.n_rows == 2  # 3 input rows, 1 dropped

    def test_dropna_features_removes_rows_with_missing_feature(self) -> None:
        ds = _toy_dataset()
        ds.df.loc[2, "nasa_aod_550"] = float("nan")
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
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
            dropna_target=False,
        )
        result = Preprocessor(spec).apply(ds)
        assert result.n_rows == 3
        assert pd.isna(result.df["y_ghi_kwh_m2_day"].iloc[0])


class TestColumnValidation:
    """Missing required columns surface as a single, named ValueError."""

    def test_missing_feature_column_raises(self) -> None:
        spec = FeatureSpec(feature_columns=("not_a_real_column",))
        with pytest.raises(ValueError, match="not_a_real_column"):
            Preprocessor(spec).apply(_toy_dataset())

    def test_missing_kt_input_raises(self) -> None:
        spec = FeatureSpec(
            derived_features=(
                ClearSkyIndexFeature(
                    ghi_column="missing_ghi",
                    ghi_clear_column="missing_clear",
                    output_column="kt",
                ),
            ),
        )
        with pytest.raises(ValueError, match="missing_ghi"):
            Preprocessor(spec).apply(_toy_dataset())

    def test_missing_target_with_dropna_raises(self) -> None:
        ds = _toy_dataset()
        ds.df.drop(columns=["y_ghi_kwh_m2_day"], inplace=True)
        spec = FeatureSpec(feature_columns=("nasa_aod_550",))
        with pytest.raises(ValueError, match="y_ghi_kwh_m2_day"):
            Preprocessor(spec).apply(ds)

    def test_altitude_without_lat_lon_raises(self) -> None:
        # AltitudeFeature declares lat/lon as required input; the
        # Preprocessor's column validation must surface that with a
        # clear error rather than letting compute() fail with a KeyError.
        spec = FeatureSpec(
            derived_features=(AltitudeFeature(provider=lambda lat, lon: 1000.0),),
        )
        with pytest.raises(ValueError, match="lat"):
            Preprocessor(spec).apply(_toy_dataset(with_coords=False))


class TestDerivedFeaturesIntegration:
    """End-to-end: each DerivedFeature emits its declared output columns.

    Smoke-tests the unifier — Preprocessor only knows about a generic
    DerivedFeature loop, not about kt vs doy vs altitude specifically.
    """

    def test_altitude_feature_emits_column(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            derived_features=(AltitudeFeature(provider=lambda lat, lon: 1200.0),),
        )
        result = Preprocessor(spec).apply(_toy_dataset(with_coords=True))
        assert "altitude_m" in result.df.columns
        assert (result.df["altitude_m"] == 1200.0).all()
        assert result.feature_columns == ("nasa_aod_550", "altitude_m")

    def test_altitude_provider_called_once_per_unique_pair(self) -> None:
        # Three rows at the same (lat, lon) must invoke the provider
        # exactly once. This is the dedupe contract that makes a
        # network-backed provider practical for batch training.
        calls: list[tuple[float, float]] = []

        def counting_provider(lat: float, lon: float) -> float:
            calls.append((lat, lon))
            return 1234.0

        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            derived_features=(AltitudeFeature(provider=counting_provider),),
        )
        Preprocessor(spec).apply(_toy_dataset(with_coords=True))
        assert len(calls) == 1

    def test_three_features_emit_six_columns(self) -> None:
        # Pin the unifier: kt + doy + altitude should produce four
        # derived columns total (kt_nasa, doy_sin, doy_cos, altitude_m)
        # plus one pass-through, in spec order.
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            derived_features=(
                _kt(),
                CyclicalDayOfYearFeature(),
                AltitudeFeature(provider=lambda lat, lon: 1500.0),
            ),
        )
        result = Preprocessor(spec).apply(_toy_dataset(with_coords=True))
        assert result.feature_columns == (
            "nasa_aod_550",
            "kt_nasa",
            "doy_sin",
            "doy_cos",
            "altitude_m",
        )


class TestStateless:
    """Re-applying the same preprocessor must produce identical output."""

    def test_two_applications_produce_equal_output(self) -> None:
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            derived_features=(_kt(), CyclicalDayOfYearFeature()),
        )
        pre = Preprocessor(spec)
        a = pre.apply(_toy_dataset())
        b = pre.apply(_toy_dataset())
        pd.testing.assert_frame_equal(a.df, b.df)


class _RecordingCleaner(DataCleaner):
    """Pass-through cleaner that records the order it ran in.

    Used by :class:`TestCleanerOrdering` to pin the contract that
    cleaners run in declared order *and* before any DerivedFeature.
    """

    _calls: list[str]
    _name: str

    def __init__(self, name: str, calls: list[str]) -> None:
        # Not a frozen dataclass — this stub mutates ``calls`` to record
        # invocation order, which a frozen-dataclass cleaner couldn't.
        self._name = name
        self._calls = calls

    @property
    def kind(self) -> CleanerKind:
        return CleanerKind.GHI_UPPER_BOUND  # arbitrary — only used for repr

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return ()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._calls.append(self._name)
        return df

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "name": self._name}

    @classmethod
    def _from_dict(cls, d: dict, *, providers: dict) -> "_RecordingCleaner":
        # Test stub — never round-tripped through JSON in this suite.
        raise NotImplementedError(
            "_RecordingCleaner is a test stub; JSON roundtrip is out of scope."
        )


class _RecordingFeature(ClearSkyIndexFeature):
    """ClearSkyIndexFeature that records when it ran. Subclassing keeps
    the existing input/output contract intact while the test only cares
    about ordering."""

    _calls: list[str]

    def __init__(self, name: str, calls: list[str], **kwargs):
        super().__init__(**kwargs)
        # Frozen dataclasses don't permit normal attribute assignment;
        # use object.__setattr__ to attach the ordering hooks.
        object.__setattr__(self, "_calls", calls)
        object.__setattr__(self, "_name", name)

    def compute(self, df):
        self._calls.append(self._name)
        return super().compute(df)


class TestCleanerOrdering:
    """Cleaners run in declared order, *before* any derived feature.

    Pinning this contract guards the speedup-from-skipping invariant
    (a feature shouldn't be computed on rows a downstream cleaner
    drops) and the "no flag-based dispatch in Preprocessor" invariant
    (the order is encoded in the spec, not in if-branches in apply()).
    """

    def test_cleaners_run_before_features_and_in_order(self) -> None:
        calls: list[str] = []
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            cleaners=(
                _RecordingCleaner("clean_a", calls),
                _RecordingCleaner("clean_b", calls),
            ),
            derived_features=(
                _RecordingFeature(
                    "derive_kt",
                    calls,
                    ghi_column="sat_ghi_nasa_kwh_m2_day",
                    ghi_clear_column="nasa_ghi_clear",
                    output_column="kt_nasa",
                ),
            ),
        )
        Preprocessor(spec).apply(_toy_dataset())
        assert calls == ["clean_a", "clean_b", "derive_kt"]

    def test_cleaner_filters_rows_before_feature_computes(self) -> None:
        # A cleaner that drops rows above sat_ghi=5.0 leaves 2 of 3 rows
        # (toy dataset has [4.5, 5.5, 3.5]). The downstream kt feature
        # therefore computes on 2 rows, and the preprocessed output
        # carries only those two — pinning that the cleaner's effect
        # propagates through to feature computation, not just to the
        # frame that the cleaner returned.
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            cleaners=(
                GhiUpperBoundCleaner(
                    column="sat_ghi_nasa_kwh_m2_day",
                    threshold=5.0,
                ),
            ),
            derived_features=(_kt(),),
        )
        result = Preprocessor(spec).apply(_toy_dataset())
        assert len(result.df) == 2
        # kt = sat_ghi / ghi_clear; the surviving rows had sat_ghi 4.5 and 3.5
        # against ghi_clear=6.0 → kt ∈ {0.75, 0.583...}. Pin the kt values
        # so a regression in the cleaner-then-feature ordering is caught.
        kt_values = sorted(result.df["kt_nasa"].round(4).tolist())
        assert kt_values == [
            pytest.approx(0.5833, abs=1e-4),
            pytest.approx(0.75, abs=1e-4),
        ]

    def test_validate_columns_catches_cleaner_input_missing(self) -> None:
        # If a cleaner needs a column that's not in the input frame,
        # the up-front validation must surface it before any cleaner runs.
        spec = FeatureSpec(
            feature_columns=("nasa_aod_550",),
            cleaners=(
                GhiUpperBoundCleaner(
                    column="not_in_frame",
                    threshold=5.0,
                ),
            ),
            derived_features=(_kt(),),
        )
        with pytest.raises(ValueError, match="not_in_frame"):
            Preprocessor(spec).apply(_toy_dataset())
