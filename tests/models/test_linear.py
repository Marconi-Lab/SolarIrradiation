"""Tests for LinearRegressor — covers the in-wrapper scaling contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from susse.models import LinearParams, LinearRegressor, ModelFactory, load_regressor


def _learnable_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """y = 2a - 3b + noise. Linear is the right model class here."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.uniform(-1, 1, n),
            "b": rng.uniform(-10, 10, n),  # very different scale from `a`
        }
    )
    y = pd.Series(2.0 * X["a"] - 3.0 * X["b"] + rng.normal(scale=0.1, size=n))
    return X, y


class TestPredictionAccuracy:
    def test_low_error_on_linear_signal_with_scaling(self) -> None:
        X, y = _learnable_data()
        model = LinearRegressor(LinearParams(with_scaling=True)).fit(X, y)
        mae = (model.predict(X) - y).abs().mean()
        # Noise std is 0.1; perfect fit would have MAE ≈ 0.08.
        assert mae < 0.15

    def test_low_error_on_linear_signal_without_scaling(self) -> None:
        # Scaling shouldn't matter for a closed-form OLS fit, but
        # numerically with very different feature scales it can; this
        # pins that the un-scaled path also produces reasonable
        # coefficients.
        X, y = _learnable_data()
        model = LinearRegressor(LinearParams(with_scaling=False)).fit(X, y)
        mae = (model.predict(X) - y).abs().mean()
        assert mae < 0.15


class TestScalingContract:
    """When scaling is on, the scaler must be fitted on the X passed to fit
    (not on every X seen at predict). Pinning this protects against a
    regression where someone added per-call scaler.fit_transform."""

    def test_predict_uses_train_fold_scaler_only(self) -> None:
        # Train on data A; predict on data B with very different
        # distribution. If predict refit the scaler on B, the scaled
        # values would be near-standard-normal, defeating the purpose.
        # Instead, B's scaling must reflect A's mean/std.
        rng = np.random.default_rng(42)
        X_train = pd.DataFrame(
            {
                "a": rng.normal(loc=0.0, scale=1.0, size=200),
            }
        )
        y_train = pd.Series(2.0 * X_train["a"] + rng.normal(scale=0.05, size=200))

        # B has a wildly shifted mean — if scaler refit on B, predictions
        # would be biased toward the mean of y_train.
        X_test = pd.DataFrame({"a": [100.0]})

        model = LinearRegressor(LinearParams(with_scaling=True)).fit(X_train, y_train)
        prediction = model.predict(X_test).iloc[0]

        # With scaler fitted on A: a=100 is ~100σ above A's mean →
        # the linear extrapolation should give a very large positive
        # prediction (~200, consistent with y = 2a slope).
        # With scaler refit on B (single point): the scaled value would
        # be 0 → prediction near intercept (~0).
        assert prediction > 100.0, (
            f"prediction={prediction} suggests the scaler was refit at "
            f"predict time. Expected scaler to be fit on the training "
            f"fold only."
        )


class TestSaveLoadRoundtrip:
    def test_with_scaling_roundtrip(self, tmp_path: Path) -> None:
        X, y = _learnable_data()
        original = LinearRegressor(LinearParams(with_scaling=True)).fit(X, y)
        original.save(tmp_path)
        restored = load_regressor(tmp_path)
        assert isinstance(restored, LinearRegressor)
        pd.testing.assert_series_equal(restored.predict(X), original.predict(X))

    def test_without_scaling_roundtrip(self, tmp_path: Path) -> None:
        X, y = _learnable_data()
        original = LinearRegressor(LinearParams(with_scaling=False)).fit(X, y)
        original.save(tmp_path)
        restored = load_regressor(tmp_path)
        pd.testing.assert_series_equal(restored.predict(X), original.predict(X))


class TestFactory:
    def test_factory_dispatches_to_correct_class(self) -> None:
        model = ModelFactory.create(LinearParams(with_scaling=False))
        assert isinstance(model, LinearRegressor)
        assert model.params.with_scaling is False


class TestLifecycle:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before .fit"):
            LinearRegressor(LinearParams()).predict(pd.DataFrame({"a": [1.0]}))
