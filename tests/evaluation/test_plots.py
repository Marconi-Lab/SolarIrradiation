"""Smoke tests for the generic evaluation plot helpers.

Plots are visual; we don't assert on pixel content. The contract we
DO assert: every helper accepts a plausibly-shaped input frame and
runs without raising. A regression in one of the helpers (column name
typo, division by zero, etc.) surfaces here before it lands in a
notebook.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — no GUI windows during tests

import numpy as np
import pandas as pd
import pytest

from susse.evaluation.plots import (
    plot_covariate_shift_kde,
    plot_feature_correlation_heatmap,
    plot_pca_by_station,
    plot_per_station_metric,
    plot_predicted_vs_observed_panels,
    plot_predictor_vs_observed_scatter,
    plot_target_distribution_per_station,
    plot_training_fit_scatter,
    plot_training_fit_timeseries,
)


@pytest.fixture
def eval_frame() -> pd.DataFrame:
    """One year of daily data across three stations, with predictions
    + two satellite columns + a categorical highlight flag."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows: list[dict] = []
    for loc in ("kampala", "soroti", "wadelai"):
        intercept = 4.5 + rng.uniform(-0.3, 0.3)
        for d in dates:
            obs = intercept + rng.normal(scale=0.3)
            rows.append(
                {
                    "date": d,
                    "location": loc,
                    "y_obs": obs,
                    "y_pred": obs + rng.normal(scale=0.1),
                    "sat_ghi_nasa_kwh_m2_day": obs + 0.5 + rng.normal(scale=0.2),
                    "sat_ghi_cams_kwh_m2_day": obs + 0.3 + rng.normal(scale=0.2),
                    "feat_a": float(rng.uniform(0, 1)),
                    "feat_b": float(rng.uniform(-1, 1)),
                    "seen_in_training": loc == "kampala",
                }
            )
    return pd.DataFrame(rows)


def test_plot_target_distribution_per_station(eval_frame: pd.DataFrame) -> None:
    plot_target_distribution_per_station(
        eval_frame,
        target_column="y_obs",
        location_column="location",
    )


def test_plot_predictor_vs_observed_scatter(eval_frame: pd.DataFrame) -> None:
    plot_predictor_vs_observed_scatter(
        eval_frame,
        observed_column="y_obs",
        predictor_columns={
            "NASA": "sat_ghi_nasa_kwh_m2_day",
            "CAMS": "sat_ghi_cams_kwh_m2_day",
        },
    )


def test_plot_feature_correlation_heatmap(eval_frame: pd.DataFrame) -> None:
    plot_feature_correlation_heatmap(
        eval_frame,
        feature_columns=("feat_a", "feat_b", "sat_ghi_nasa_kwh_m2_day"),
        target_column="y_obs",
        top_n=3,
    )


def test_plot_pca_by_station(eval_frame: pd.DataFrame) -> None:
    plot_pca_by_station(
        eval_frame,
        feature_columns=(
            "feat_a",
            "feat_b",
            "sat_ghi_nasa_kwh_m2_day",
            "sat_ghi_cams_kwh_m2_day",
        ),
    )


def test_plot_training_fit_scatter(eval_frame: pd.DataFrame) -> None:
    plot_training_fit_scatter(
        eval_frame,
        observed_column="y_obs",
        predicted_column="y_pred",
    )


def test_plot_training_fit_timeseries(eval_frame: pd.DataFrame) -> None:
    plot_training_fit_timeseries(
        eval_frame,
        observed_column="y_obs",
        prediction_series={"Model": "y_pred"},
        reference_series={
            "NASA": "sat_ghi_nasa_kwh_m2_day",
            "CAMS": "sat_ghi_cams_kwh_m2_day",
        },
        n_stations=2,
    )


def test_plot_training_fit_timeseries_no_references(
    eval_frame: pd.DataFrame,
) -> None:
    """The reference_series arg is optional — predictions alone must work."""
    plot_training_fit_timeseries(
        eval_frame,
        observed_column="y_obs",
        prediction_series={"Model": "y_pred"},
        n_stations=2,
    )


def test_plot_predicted_vs_observed_panels(eval_frame: pd.DataFrame) -> None:
    plot_predicted_vs_observed_panels(
        eval_frame,
        observed_column="y_obs",
        predictions={
            "Model": "y_pred",
            "NASA": "sat_ghi_nasa_kwh_m2_day",
            "CAMS": "sat_ghi_cams_kwh_m2_day",
        },
    )


def test_plot_per_station_metric_with_highlight(eval_frame: pd.DataFrame) -> None:
    plot_per_station_metric(
        eval_frame,
        observed_column="y_obs",
        predicted_column="y_pred",
        highlight_column="seen_in_training",
    )


def test_plot_covariate_shift_kde(eval_frame: pd.DataFrame) -> None:
    plot_covariate_shift_kde(
        eval_frame.iloc[:300],
        eval_frame.iloc[300:],
        features=("feat_a", "feat_b"),
    )
