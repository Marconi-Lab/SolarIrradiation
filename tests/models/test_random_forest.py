"""Tests for RandomForestRegressor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from susse.models import (
    ModelFactory,
    RandomForestParams,
    RandomForestRegressor,
    load_regressor,
)


def _learnable_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Toy regression where y is a simple function of X (with light noise).

    A correctly-fitted RF should reproduce this with much lower error
    than the standard deviation of y.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.uniform(-1, 1, n),
            "b": rng.uniform(-1, 1, n),
        }
    )
    y = pd.Series(2.0 * X["a"] - 3.0 * X["b"] + rng.normal(scale=0.1, size=n))
    return X, y


class TestFit:
    def test_random_seed_makes_predictions_deterministic(self) -> None:
        X, y = _learnable_data()
        params = RandomForestParams(n_estimators=20, random_state=123, n_jobs=1)
        a = RandomForestRegressor(params).fit(X, y).predict(X)
        b = RandomForestRegressor(params).fit(X, y).predict(X)
        pd.testing.assert_series_equal(a, b)

    def test_predictions_better_than_target_mean_baseline(self) -> None:
        # Sanity check: an RF with enough trees on a learnable signal
        # should have substantially lower MAE than predicting the mean.
        X, y = _learnable_data()
        model = RandomForestRegressor(
            RandomForestParams(n_estimators=50, random_state=42, n_jobs=1)
        ).fit(X, y)
        preds = model.predict(X)
        rf_mae = (preds - y).abs().mean()
        baseline_mae = (y - y.mean()).abs().mean()
        # Generous threshold so this isn't sensitive to sklearn internals;
        # the RF should still be much better than the baseline.
        assert rf_mae < 0.5 * baseline_mae


class TestLifecycle:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before .fit"):
            RandomForestRegressor(RandomForestParams()).predict(
                pd.DataFrame({"a": [1.0], "b": [2.0]})
            )


class TestSaveLoadRoundtrip:
    def test_predictions_unchanged_after_roundtrip(self, tmp_path: Path) -> None:
        X, y = _learnable_data()
        original = RandomForestRegressor(
            RandomForestParams(n_estimators=10, random_state=7, n_jobs=1)
        ).fit(X, y)
        original.save(tmp_path)
        restored = load_regressor(tmp_path)
        assert isinstance(restored, RandomForestRegressor)
        # The reloaded params must equal the originals.
        assert restored.params == original.params
        pd.testing.assert_series_equal(restored.predict(X), original.predict(X))


class TestFactory:
    def test_factory_dispatches_to_correct_class(self) -> None:
        model = ModelFactory.create(RandomForestParams(n_estimators=5))
        assert isinstance(model, RandomForestRegressor)
        assert model.params.n_estimators == 5
