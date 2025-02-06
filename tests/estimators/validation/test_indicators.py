import numpy as np
import pytest

from susse.estimators.validation.indicators import StatisticalIndicators


class TestStatisticalIndicators:

    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data"""
        observations = np.array([10, 20, 30, 40, 50])
        predictions = np.array([12, 21, 29, 42, 48])
        return observations, predictions

    def test_initialization(self, sample_data):
        observations, predictions = sample_data
        stats = StatisticalIndicators(observations, predictions)
        assert np.array_equal(stats.observations, observations)
        assert np.array_equal(stats.predictions, predictions)
        assert np.array_equal(stats.deviations, predictions - observations)

    def test_mean_bias_deviation(self, sample_data):
        observations, predictions = sample_data
        stats = StatisticalIndicators(observations, predictions)
        expected_mbd = 0.4  # (2 + 1 - 1 + 2 -2) / 5
        assert np.isclose(stats.mean_bias_deviation(), expected_mbd)

    def test_mean_absolute_deviation(self, sample_data):
        observations, predictions = sample_data
        stats = StatisticalIndicators(observations, predictions)
        expected_mad = 1.6  # (2 + 1 + 1 + 2 + 2) / 5
        assert np.isclose(stats.mean_absolute_deviation(), expected_mad)

    def test_root_mean_square_deviation(self, sample_data):
        observations, predictions = sample_data
        stats = StatisticalIndicators(observations, predictions)
        expected_rmsd = np.sqrt(2)  # sqrt((4 + 1 + 1 + 4 + 4 ) / 5)
        assert np.isclose(stats.root_mean_square_deviation(), expected_rmsd)

    def test_relative_root_mean_square_deviation(self, sample_data):
        observations, predictions = sample_data
        stats = StatisticalIndicators(observations, predictions)
        rmsd = np.sqrt(2)
        expected_rrmsd = (rmsd / np.mean(observations)) * 100
        assert np.isclose(stats.relative_root_mean_square_deviation(), expected_rrmsd)

    def test_empty_arrays(self):
        with pytest.raises(ValueError):
            StatisticalIndicators([], [])

    def test_unequal_arrays(self):
        with pytest.raises(ValueError):
            StatisticalIndicators([1, 2, 3], [1, 2])

    def test_zero_mean_observations(self):
        with pytest.raises(ZeroDivisionError):
            stats = StatisticalIndicators([0, 0, 0], [1, 1, 1])
            stats.relative_root_mean_square_deviation()
