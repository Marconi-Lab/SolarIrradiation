"""Tests for MeanBaselineRegressor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from susse.models import (
    MeanBaselineParams,
    MeanBaselineRegressor,
    ModelFactory,
    load_regressor,
)


def _toy_data(n: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.normal(loc=5.0, scale=1.0, size=n))
    return X, y


class TestPredictionBehavior:
    def test_predicts_training_mean_for_every_row(self) -> None:
        X, y = _toy_data()
        model = MeanBaselineRegressor(MeanBaselineParams()).fit(X, y)
        preds = model.predict(X)
        # All predictions equal training mean.
        assert preds.nunique() == 1
        assert preds.iloc[0] == pytest.approx(y.mean())

    def test_median_statistic_predicts_median(self) -> None:
        X, y = _toy_data()
        model = MeanBaselineRegressor(MeanBaselineParams(statistic="median")).fit(X, y)
        assert model.predict(X).iloc[0] == pytest.approx(y.median())

    def test_predict_aligns_to_input_index(self) -> None:
        # Non-default index must round-trip to predictions.
        X, y = _toy_data()
        X_test = X.iloc[:5].set_axis([100, 200, 300, 400, 500])
        model = MeanBaselineRegressor(MeanBaselineParams()).fit(X, y)
        preds = model.predict(X_test)
        assert list(preds.index) == [100, 200, 300, 400, 500]


class TestLifecycle:
    def test_predict_before_fit_raises(self) -> None:
        model = MeanBaselineRegressor(MeanBaselineParams())
        with pytest.raises(RuntimeError, match="before .fit"):
            model.predict(pd.DataFrame({"a": [1.0]}))

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        model = MeanBaselineRegressor(MeanBaselineParams())
        with pytest.raises(RuntimeError, match="unfitted"):
            model.save(tmp_path)

    def test_empty_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            MeanBaselineRegressor(MeanBaselineParams()).fit(
                pd.DataFrame({"a": []}), pd.Series([], dtype=float)
            )


class TestSaveLoadRoundtrip:
    def test_predictions_unchanged_after_roundtrip(self, tmp_path: Path) -> None:
        X, y = _toy_data()
        original = MeanBaselineRegressor(MeanBaselineParams()).fit(X, y)
        original.save(tmp_path)
        restored = load_regressor(tmp_path)
        # The dispatched class must be the right concrete one.
        assert isinstance(restored, MeanBaselineRegressor)
        # Predictions must be byte-identical.
        pd.testing.assert_series_equal(restored.predict(X), original.predict(X))


class TestFactory:
    def test_factory_dispatches_to_correct_class(self) -> None:
        model = ModelFactory.create(MeanBaselineParams())
        assert isinstance(model, MeanBaselineRegressor)
