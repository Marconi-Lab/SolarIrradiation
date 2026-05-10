"""Tests for the paper-style evaluation metrics in :mod:`susse.metrics`.

Each metric is pinned against an analytical reference (perfect
prediction, constant offset, etc.) rather than against an arbitrary
numeric value — the contract is what the formula computes, not what we
got the first time we ran it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from susse.metrics import (
    index_of_agreement,
    mean_bias_error,
    normalised_mae,
    normalised_rmse,
)


# ---------------------------------------------------------------------------
# Index of Agreement (Willmott 1981)
# ---------------------------------------------------------------------------


class TestIndexOfAgreement:
    def test_perfect_prediction_is_one(self) -> None:
        y_true = pd.Series([4.0, 5.0, 6.0, 4.5])
        ioa = index_of_agreement(y_true, y_true.copy())
        assert ioa == pytest.approx(1.0, abs=1e-12)

    def test_constant_prediction_at_observed_mean_is_zero(self) -> None:
        # When the model collapses to predicting mean(O), the IOA formula
        # has numerator = denominator = Σ(O - mean(O))², so IOA = 0.
        # This is the "no skill" baseline against which any non-constant
        # predictor must improve to register IOA > 0.
        y_true = pd.Series([3.0, 4.0, 5.0, 6.0])
        y_pred = pd.Series([4.5, 4.5, 4.5, 4.5])  # constant = mean(y_true)
        ioa = index_of_agreement(y_true, y_pred)
        assert ioa == pytest.approx(0.0, abs=1e-12)

    def test_handles_nan_rows(self) -> None:
        # Rows where either side is NaN must be dropped, not propagated.
        y_true = pd.Series([4.0, np.nan, 6.0, 4.5])
        y_pred = pd.Series([4.0, 5.0, 6.0, np.nan])
        ioa = index_of_agreement(y_true, y_pred)
        # The two surviving rows are perfect matches → IOA = 1.0.
        assert ioa == pytest.approx(1.0, abs=1e-12)

    def test_returns_nan_when_perfectly_constant_match(self) -> None:
        # The genuinely undefined case: both series collapse to the same
        # constant, so numerator AND denominator are zero. IOA should be
        # NaN, not a division-by-zero or a spurious 1.0.
        y_true = pd.Series([5.0, 5.0, 5.0])
        y_pred = pd.Series([5.0, 5.0, 5.0])
        ioa = index_of_agreement(y_true, y_pred)
        assert math.isnan(ioa)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            index_of_agreement(pd.Series([1, 2, 3]), pd.Series([1, 2]))


# ---------------------------------------------------------------------------
# Mean Bias Error
# ---------------------------------------------------------------------------


class TestMeanBiasError:
    def test_constant_offset_recovered(self) -> None:
        # MBE = mean(P - O); if the model's offset is +0.5, MBE must
        # equal +0.5 (this is the formula's whole point — it surfaces
        # systematic bias that MAE/RMSE absolute-value away).
        y_true = pd.Series([4.0, 5.0, 6.0, 4.5])
        y_pred = y_true + 0.5
        assert mean_bias_error(y_true, y_pred) == pytest.approx(0.5, abs=1e-12)

    def test_negative_offset(self) -> None:
        y_true = pd.Series([4.0, 5.0, 6.0, 4.5])
        y_pred = y_true - 0.7
        assert mean_bias_error(y_true, y_pred) == pytest.approx(-0.7, abs=1e-12)

    def test_zero_when_perfect(self) -> None:
        y_true = pd.Series([4.0, 5.0, 6.0])
        assert mean_bias_error(y_true, y_true.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_returns_nan_on_empty_after_filter(self) -> None:
        y_true = pd.Series([np.nan, np.nan])
        y_pred = pd.Series([1.0, 2.0])
        assert math.isnan(mean_bias_error(y_true, y_pred))


# ---------------------------------------------------------------------------
# Normalised RMSE / MAE
# ---------------------------------------------------------------------------


class TestNormalisedRmseAndMae:
    def test_nrmse_perfect_prediction_is_zero(self) -> None:
        y_true = pd.Series([4.0, 5.0, 6.0])
        assert normalised_rmse(y_true, y_true.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_nmae_perfect_prediction_is_zero(self) -> None:
        y_true = pd.Series([4.0, 5.0, 6.0])
        assert normalised_mae(y_true, y_true.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_nrmse_matches_formula(self) -> None:
        # Compute manually: RMSE = sqrt(mean((P-O)^2)); nRMSE = 100*RMSE/mean(O).
        y_true = pd.Series([4.0, 5.0, 6.0])
        y_pred = pd.Series([5.0, 5.0, 5.0])  # constant prediction
        rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
        expected = 100.0 * rmse / y_true.mean()
        assert normalised_rmse(y_true, y_pred) == pytest.approx(expected, abs=1e-12)

    def test_nmae_matches_formula(self) -> None:
        y_true = pd.Series([4.0, 5.0, 6.0])
        y_pred = pd.Series([5.0, 5.0, 5.0])
        mae = float(np.abs(y_pred - y_true).mean())
        expected = 100.0 * mae / y_true.mean()
        assert normalised_mae(y_true, y_pred) == pytest.approx(expected, abs=1e-12)

    def test_returns_nan_when_observed_mean_is_zero(self) -> None:
        # nRMSE / nMAE are undefined when the denominator is zero;
        # surfacing NaN is more honest than dividing by zero.
        y_true = pd.Series([-1.0, 0.0, 1.0])  # mean = 0
        y_pred = pd.Series([0.0, 0.0, 0.0])
        assert math.isnan(normalised_rmse(y_true, y_pred))
        assert math.isnan(normalised_mae(y_true, y_pred))


# ---------------------------------------------------------------------------
# Cross-metric consistency on the paper's-style signal
# ---------------------------------------------------------------------------


class TestPaperShapedExample:
    """Sanity check the metrics against a synthesised "satellite has bias,
    RF corrects it" signal. Values are picked so that the metrics line
    up with the *direction* of the paper's Table IV (RF beats CAMS on
    every metric)."""

    def test_corrected_predictor_beats_biased_one(self) -> None:
        rng = np.random.default_rng(0)
        # 50 monthly observations, "true" GHI around 4.5.
        y_true = pd.Series(rng.normal(loc=4.5, scale=0.4, size=50))
        # Biased "satellite" predictor: +1.0 systematic offset, similar
        # spread to the observations.
        y_satellite = y_true + 1.0 + rng.normal(scale=0.2, size=50)
        # Bias-corrected predictor: residual offset near zero, smaller
        # noise.
        y_corrected = y_true + rng.normal(scale=0.15, size=50)

        # MBE should expose the satellite's positive bias and the
        # corrected one's near-zero bias.
        assert mean_bias_error(y_true, y_satellite) > 0.5
        assert abs(mean_bias_error(y_true, y_corrected)) < 0.1

        # IOA should be higher for the corrected predictor.
        ioa_sat = index_of_agreement(y_true, y_satellite)
        ioa_cor = index_of_agreement(y_true, y_corrected)
        assert ioa_cor > ioa_sat

        # nRMSE / nMAE: corrected wins on both.
        assert normalised_rmse(y_true, y_corrected) < normalised_rmse(y_true, y_satellite)
        assert normalised_mae(y_true, y_corrected) < normalised_mae(y_true, y_satellite)
