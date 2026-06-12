"""Evaluator — multi-prediction × multi-split scoring tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from susse.evaluation import DEFAULT_METRICS, RMSE, Evaluator, MeanBiasError


@pytest.fixture
def comparison_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 60
    obs = pd.Series(rng.normal(loc=4.5, scale=0.4, size=n))
    return pd.DataFrame(
        {
            "obs": obs,
            "pred_good": obs + rng.normal(scale=0.1, size=n),
            "pred_biased": obs + 0.8 + rng.normal(scale=0.1, size=n),
            "group": ["A"] * 30 + ["B"] * 30,
        }
    )


class TestEvaluatorOutput:
    def test_one_row_per_prediction_when_no_splits(
        self, comparison_frame: pd.DataFrame
    ) -> None:
        result = Evaluator().score(
            observed=comparison_frame["obs"],
            predictions={
                "good": comparison_frame["pred_good"],
                "biased": comparison_frame["pred_biased"],
            },
        )
        assert len(result) == 2
        assert list(result["split"].unique()) == ["all"]
        assert sorted(result["prediction"]) == ["biased", "good"]

    def test_row_count_is_splits_times_predictions(
        self, comparison_frame: pd.DataFrame
    ) -> None:
        splits = {
            "A only": comparison_frame.index[comparison_frame["group"] == "A"],
            "B only": comparison_frame.index[comparison_frame["group"] == "B"],
        }
        result = Evaluator().score(
            observed=comparison_frame["obs"],
            predictions={
                "good": comparison_frame["pred_good"],
                "biased": comparison_frame["pred_biased"],
            },
            splits=splits,
        )
        assert len(result) == 4

    def test_columns_include_every_metric(self, comparison_frame: pd.DataFrame) -> None:
        result = Evaluator().score(
            observed=comparison_frame["obs"],
            predictions={"good": comparison_frame["pred_good"]},
        )
        expected_cols = ["split", "prediction", "n"] + [m.name for m in DEFAULT_METRICS]
        assert list(result.columns) == expected_cols

    def test_custom_metric_tuple_controls_columns(
        self, comparison_frame: pd.DataFrame
    ) -> None:
        result = Evaluator(metrics=(RMSE(), MeanBiasError())).score(
            observed=comparison_frame["obs"],
            predictions={"good": comparison_frame["pred_good"]},
        )
        assert list(result.columns) == ["split", "prediction", "n", "RMSE", "MBE"]


class TestEvaluatorSemantics:
    def test_biased_prediction_has_higher_rmse_than_good(
        self, comparison_frame: pd.DataFrame
    ) -> None:
        result = Evaluator().score(
            observed=comparison_frame["obs"],
            predictions={
                "good": comparison_frame["pred_good"],
                "biased": comparison_frame["pred_biased"],
            },
        )
        rmse_good = result.loc[result["prediction"] == "good", "RMSE"].iloc[0]
        rmse_biased = result.loc[result["prediction"] == "biased", "RMSE"].iloc[0]
        assert rmse_biased > rmse_good

    def test_n_counts_only_non_nan_rows(self, comparison_frame: pd.DataFrame) -> None:
        # Introduce two NaN-paired rows on the prediction side.
        pred = comparison_frame["pred_good"].copy()
        pred.iloc[:2] = math.nan
        result = Evaluator().score(
            observed=comparison_frame["obs"],
            predictions={"good": pred},
        )
        assert result["n"].iloc[0] == len(comparison_frame) - 2


class TestEvaluatorErrors:
    def test_empty_predictions_raises(self, comparison_frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="empty"):
            Evaluator().score(observed=comparison_frame["obs"], predictions={})

    def test_empty_metrics_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one Metric"):
            Evaluator(metrics=())

    def test_split_index_outside_observed_raises(
        self, comparison_frame: pd.DataFrame
    ) -> None:
        bad = pd.Index([999, 1000])
        with pytest.raises(ValueError, match="absent from"):
            Evaluator().score(
                observed=comparison_frame["obs"],
                predictions={"good": comparison_frame["pred_good"]},
                splits={"out of range": bad},
            )
