from functools import wraps
from typing import Tuple

import numpy as np


def validate_inputs(func):
    @wraps(func)
    def wrapper(observations: np.ndarray, predictions: np.ndarray) -> float:
        observations = np.asarray(observations) if not isinstance(observations, np.ndarray) else observations
        predictions = np.asarray(predictions) if not isinstance(predictions, np.ndarray) else predictions
        if not isinstance(observations, np.ndarray) or not isinstance(predictions, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        if observations.shape != predictions.shape:
            raise ValueError("Arrays must have same shape")
        if len(observations) == 0:
            raise ValueError("Arrays cannot be empty")
        return func(observations, predictions)

    return wrapper


class StatisticalMetrics:
    """Static methods for computing statistical metrics between observations and predictions"""

    @staticmethod
    @validate_inputs
    def mean_bias_deviation(observations: np.ndarray, predictions: np.ndarray) -> float:
        """Mean Bias Deviation (MBD)"""
        return float(np.mean(predictions - observations))

    @staticmethod
    @validate_inputs
    def mean_absolute_deviation(
        observations: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Mean Absolute Deviation (MAD)"""
        return float(np.mean(np.abs(predictions - observations)))

    @staticmethod
    @validate_inputs
    def root_mean_square_deviation(
        observations: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Root Mean Square Deviation (RMSD)"""
        return float(np.sqrt(np.mean(np.square(predictions - observations))))

    @staticmethod
    @validate_inputs
    def relative_rmsd(observations: np.ndarray, predictions: np.ndarray) -> float:
        """Relative Root Mean Square Deviation (rRMSD) in percentage"""
        rmsd = StatisticalMetrics.root_mean_square_deviation(observations, predictions)
        return float((rmsd / np.mean(observations)) * 100)

    @staticmethod
    @validate_inputs
    def pearson_correlation(observations: np.ndarray, predictions: np.ndarray) -> float:
        """Pearson correlation coefficient"""
        return float(np.corrcoef(observations, predictions)[0, 1])

    @staticmethod
    @validate_inputs
    def nash_sutcliffe_efficiency(
        observations: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Nash-Sutcliffe Efficiency (NSE)"""
        numerator = np.sum(np.square(predictions - observations))
        denominator = np.sum(np.square(observations - np.mean(observations)))
        return float(1 - (numerator / denominator))

    @staticmethod
    @validate_inputs
    def index_of_agreement(observations: np.ndarray, predictions: np.ndarray) -> float:
        """Willmott's Index of Agreement (IoA)"""
        numerator = np.sum(np.square(predictions - observations))
        denominator = np.sum(
            np.square(
                np.abs(predictions - np.mean(observations))
                + np.abs(observations - np.mean(observations))
            )
        )
        return float(1 - (numerator / denominator))

    @staticmethod
    @validate_inputs
    def mean_absolute_percentage_error(
        observations: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Mean Absolute Percentage Error (MAPE)"""
        return float(np.mean(np.abs((observations - predictions) / observations)) * 100)
