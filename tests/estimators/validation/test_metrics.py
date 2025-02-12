import numpy as np
import pytest

from susse.estimators.validation.metrics import StatisticalMetrics


def test_mean_bias_deviation():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.mean_bias_deviation(obs, pred) == pytest.approx(
        -3.33, rel=1e-2
    )


def test_mean_absolute_deviation():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.mean_absolute_deviation(obs, pred) == pytest.approx(
        10, rel=1e-2
    )


def test_root_mean_square_deviation():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.root_mean_square_deviation(obs, pred) == pytest.approx(
        10, rel=1e-2
    )


def test_relative_rmsd():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.relative_rmsd(obs, pred) == pytest.approx(5, rel=1e-2)


def test_pearson_correlation():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.pearson_correlation(obs, pred) == pytest.approx(
        1, rel=1e-2
    )


def test_nash_sutcliffe_efficiency():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.nash_sutcliffe_efficiency(obs, pred) == pytest.approx(
        0.9, rel=1e-2
    )


def test_index_of_agreement():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.index_of_agreement(obs, pred) == pytest.approx(
        0.98, rel=1e-2
    )


def test_mean_absolute_percentage_error():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190, 290])
    assert StatisticalMetrics.mean_absolute_percentage_error(
        obs, pred
    ) == pytest.approx(5, rel=1e-2)


def test_empty_arrays():
    obs = np.array([])
    pred = np.array([])
    with pytest.raises(ValueError, match="Arrays cannot be empty"):
        StatisticalMetrics.mean_bias_deviation(obs, pred)


def test_mismatched_shapes():
    obs = np.array([100, 200, 300])
    pred = np.array([110, 190])
    with pytest.raises(ValueError, match="Arrays must have same shape"):
        StatisticalMetrics.mean_absolute_deviation(obs, pred)


def test_invalid_types():
    obs = [100, 200, 300]  # Not a NumPy array
    pred = np.array([110, 190, 290])
    with pytest.raises(TypeError, match="Inputs must be numpy arrays"):
        StatisticalMetrics.root_mean_square_deviation(obs, pred)
