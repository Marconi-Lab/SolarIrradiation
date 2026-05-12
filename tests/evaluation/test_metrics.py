"""Metric ABC subclasses — pin every metric against analytical references.

The contract: each metric's :meth:`compute` matches its formula on
analytical examples (perfect prediction → 0, constant offset → MBE
recovers the offset, etc.), drops NaN-bearing rows, and returns NaN
on the pathological cases (division by zero, empty after filter).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from susse.evaluation import (
    DEFAULT_METRICS,
    MAE,
    R2,
    RMSE,
    IndexOfAgreement,
    MeanBiasError,
    Metric,
    NormalisedMAE,
    NormalisedRMSE,
)

# ---------------------------------------------------------------------------
# Shape mismatch handling — common to every metric via the shared helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        RMSE(),
        MAE(),
        R2(),
        IndexOfAgreement(),
        MeanBiasError(),
        NormalisedRMSE(),
        NormalisedMAE(),
    ],
)
class TestSharedContract:
    def test_shape_mismatch_raises(self, metric: Metric) -> None:
        with pytest.raises(ValueError, match="shape"):
            metric.compute(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0]))

    def test_nan_rows_are_dropped(self, metric: Metric) -> None:
        # Two NaN-paired rows + two perfectly-matched rows. The metric
        # should compute as if only the matched rows existed.
        yt = pd.Series([4.0, np.nan, 6.0, np.nan])
        yp = pd.Series([4.0, 5.0, 6.0, 7.0])
        score = metric.compute(yt, yp)
        # For every "perfect prediction" metric, all-perfect inputs
        # give a known outcome:
        #   RMSE, MAE, MBE, NormalisedRMSE, NormalisedMAE → 0
        #   R², IndexOfAgreement                            → 1 (or NaN)
        if isinstance(
            metric, (RMSE, MAE, MeanBiasError, NormalisedRMSE, NormalisedMAE)
        ):
            assert score == pytest.approx(0.0, abs=1e-12)
        elif isinstance(metric, (R2, IndexOfAgreement)):
            # 2 perfect rows: R² formula reduces to 1.0 if variance > 0,
            # else NaN. In our case yt=[4, 6] has variance > 0 → 1.0.
            assert score == pytest.approx(1.0, abs=1e-12)

    def test_empty_after_filter_returns_nan(self, metric: Metric) -> None:
        yt = pd.Series([np.nan, np.nan])
        yp = pd.Series([1.0, 2.0])
        assert math.isnan(metric.compute(yt, yp))


# ---------------------------------------------------------------------------
# RMSE / MAE / R²
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_perfect_prediction_is_zero(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0])
        assert RMSE().compute(yt, yt.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_matches_formula(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0])
        yp = pd.Series([5.0, 5.0, 5.0])  # constant prediction
        expected = float(np.sqrt(((yp - yt) ** 2).mean()))
        assert RMSE().compute(yt, yp) == pytest.approx(expected, abs=1e-12)

    def test_name(self) -> None:
        assert RMSE().name == "RMSE"


class TestMAE:
    def test_perfect_prediction_is_zero(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0])
        assert MAE().compute(yt, yt.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_matches_formula(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0])
        yp = pd.Series([5.0, 5.0, 5.0])
        expected = float(np.abs(yp - yt).mean())
        assert MAE().compute(yt, yp) == pytest.approx(expected, abs=1e-12)

    def test_name(self) -> None:
        assert MAE().name == "MAE"


class TestR2:
    def test_perfect_prediction_is_one(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0])
        assert R2().compute(yt, yt.copy()) == pytest.approx(1.0, abs=1e-12)

    def test_predicting_mean_is_zero(self) -> None:
        # When prediction = mean(obs), SS_res = SS_tot → R² = 0.
        yt = pd.Series([3.0, 4.0, 5.0, 6.0])
        yp = pd.Series([4.5, 4.5, 4.5, 4.5])
        assert R2().compute(yt, yp) == pytest.approx(0.0, abs=1e-12)

    def test_constant_target_returns_nan(self) -> None:
        # Zero variance in target → undefined.
        yt = pd.Series([5.0, 5.0, 5.0])
        yp = pd.Series([5.0, 5.0, 5.0])
        assert math.isnan(R2().compute(yt, yp))


# ---------------------------------------------------------------------------
# Index of Agreement (Willmott 1981)
# ---------------------------------------------------------------------------


class TestIndexOfAgreement:
    def test_perfect_prediction_is_one(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0, 4.5])
        assert IndexOfAgreement().compute(yt, yt.copy()) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_predicting_observed_mean_is_zero(self) -> None:
        # When the model collapses to mean(O), IOA formula has
        # numerator == denominator → 0.
        yt = pd.Series([3.0, 4.0, 5.0, 6.0])
        yp = pd.Series([4.5, 4.5, 4.5, 4.5])
        assert IndexOfAgreement().compute(yt, yp) == pytest.approx(0.0, abs=1e-12)

    def test_constant_match_returns_nan(self) -> None:
        # numerator == denominator == 0 — IOA must be NaN, not 1.0
        # (no division-by-zero, no spurious perfect score).
        yt = pd.Series([5.0, 5.0, 5.0])
        yp = pd.Series([5.0, 5.0, 5.0])
        assert math.isnan(IndexOfAgreement().compute(yt, yp))


# ---------------------------------------------------------------------------
# Mean Bias Error
# ---------------------------------------------------------------------------


class TestMeanBiasError:
    def test_positive_offset_recovered(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0, 4.5])
        yp = yt + 0.5
        assert MeanBiasError().compute(yt, yp) == pytest.approx(0.5, abs=1e-12)

    def test_negative_offset_recovered(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0, 4.5])
        yp = yt - 0.7
        assert MeanBiasError().compute(yt, yp) == pytest.approx(-0.7, abs=1e-12)


# ---------------------------------------------------------------------------
# Normalised RMSE / MAE
# ---------------------------------------------------------------------------


class TestNormalisedRMSE:
    def test_matches_formula(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0])
        yp = pd.Series([5.0, 5.0, 5.0])
        rmse = float(np.sqrt(((yp - yt) ** 2).mean()))
        expected = 100.0 * rmse / yt.mean()
        assert NormalisedRMSE().compute(yt, yp) == pytest.approx(expected, abs=1e-12)

    def test_zero_mean_observed_returns_nan(self) -> None:
        yt = pd.Series([-1.0, 0.0, 1.0])  # mean = 0
        yp = pd.Series([0.0, 0.0, 0.0])
        assert math.isnan(NormalisedRMSE().compute(yt, yp))


class TestNormalisedMAE:
    def test_matches_formula(self) -> None:
        yt = pd.Series([4.0, 5.0, 6.0])
        yp = pd.Series([5.0, 5.0, 5.0])
        mae = float(np.abs(yp - yt).mean())
        expected = 100.0 * mae / yt.mean()
        assert NormalisedMAE().compute(yt, yp) == pytest.approx(expected, abs=1e-12)

    def test_zero_mean_observed_returns_nan(self) -> None:
        yt = pd.Series([-1.0, 0.0, 1.0])
        yp = pd.Series([0.0, 0.0, 0.0])
        assert math.isnan(NormalisedMAE().compute(yt, yp))


# ---------------------------------------------------------------------------
# Cross-metric consistency on the paper's signal shape
# ---------------------------------------------------------------------------


class TestPaperShapedExample:
    """Synthesised "satellite has bias, RF corrects it" signal —
    the *direction* of every metric must agree with the paper's
    relative-improvement claim (RF beats biased satellite)."""

    def test_corrected_predictor_beats_biased_one(self) -> None:
        rng = np.random.default_rng(0)
        y_true = pd.Series(rng.normal(loc=4.5, scale=0.4, size=50))
        y_satellite = y_true + 1.0 + rng.normal(scale=0.2, size=50)
        y_corrected = y_true + rng.normal(scale=0.15, size=50)

        # MBE: biased satellite shows the positive offset; corrected
        # is near zero.
        assert MeanBiasError().compute(y_true, y_satellite) > 0.5
        assert abs(MeanBiasError().compute(y_true, y_corrected)) < 0.1

        # IOA: corrected wins.
        assert IndexOfAgreement().compute(y_true, y_corrected) > (
            IndexOfAgreement().compute(y_true, y_satellite)
        )

        # Normalised RMSE / MAE: corrected wins.
        assert NormalisedRMSE().compute(y_true, y_corrected) < (
            NormalisedRMSE().compute(y_true, y_satellite)
        )
        assert NormalisedMAE().compute(y_true, y_corrected) < (
            NormalisedMAE().compute(y_true, y_satellite)
        )


# ---------------------------------------------------------------------------
# DEFAULT_METRICS bundle
# ---------------------------------------------------------------------------


class TestDefaultMetrics:
    def test_default_tuple_has_seven_metrics_in_paper_order(self) -> None:
        names = [m.name for m in DEFAULT_METRICS]
        assert names == ["RMSE", "nRMSE_%", "MAE", "nMAE_%", "MBE", "R²", "IOA"]
