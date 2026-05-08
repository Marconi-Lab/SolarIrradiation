"""Tests for typed model params + dispatch enum."""

from __future__ import annotations

import pytest

from susse.models import (
    LinearParams, MeanBaselineParams, ModelKind, RandomForestParams,
    params_from_dict,
)


class TestModelKindDispatch:
    """ModelKind methods own the kind → params/model class mapping."""

    @pytest.mark.parametrize(
        ("kind", "expected_params_cls"),
        [
            (ModelKind.MEAN_BASELINE, MeanBaselineParams),
            (ModelKind.RANDOM_FOREST, RandomForestParams),
            (ModelKind.LINEAR, LinearParams),
        ],
    )
    def test_params_type_returns_matching_dataclass(
        self, kind: ModelKind, expected_params_cls: type
    ) -> None:
        assert kind.params_type() is expected_params_cls

    def test_model_class_returns_concrete_subclass(self) -> None:
        # Just verify the call resolves and returns a class — concrete-class
        # tests live in their own modules.
        for kind in ModelKind:
            cls = kind.model_class()
            assert isinstance(cls, type)


class TestParamsValidation:
    def test_meanbaseline_rejects_unknown_statistic(self) -> None:
        with pytest.raises(ValueError, match="mean.+median"):
            MeanBaselineParams(statistic="mode")

    def test_random_forest_rejects_zero_estimators(self) -> None:
        with pytest.raises(ValueError, match="n_estimators"):
            RandomForestParams(n_estimators=0)

    def test_random_forest_rejects_zero_max_depth(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            RandomForestParams(max_depth=0)

    def test_random_forest_allows_none_max_depth(self) -> None:
        # None means unlimited depth — must not be rejected.
        p = RandomForestParams(max_depth=None)
        assert p.max_depth is None


class TestKindProperty:
    def test_each_params_dataclass_reports_its_kind(self) -> None:
        assert MeanBaselineParams().kind is ModelKind.MEAN_BASELINE
        assert RandomForestParams().kind is ModelKind.RANDOM_FOREST
        assert LinearParams().kind is ModelKind.LINEAR


class TestJsonRoundtrip:
    """to_dict + params_from_dict roundtrip preserves params, even with
    non-default values."""

    @pytest.mark.parametrize(
        "original",
        [
            MeanBaselineParams(statistic="median"),
            RandomForestParams(n_estimators=50, max_depth=8, random_state=7),
            LinearParams(with_scaling=False, fit_intercept=False),
        ],
    )
    def test_roundtrip(self, original) -> None:
        d = original.to_dict()
        recovered = params_from_dict(d)
        assert recovered == original
        assert recovered.kind is original.kind

    def test_missing_kind_tag_raises(self) -> None:
        # to_dict always sets `kind`; params_from_dict expects it. A
        # caller building the dict by hand without it should get a
        # clear error, not silently default to one model type.
        with pytest.raises(ValueError, match="kind"):
            params_from_dict({"n_estimators": 100})
