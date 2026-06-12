"""Constant-prediction baseline.

Predicts the training set's mean (or median) for every input row. Almost
useless as a model — useful as a sanity baseline: any "real" model that
scores worse than this on a held-out set has a bug.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseRegressor
from .params import MeanBaselineParams


class MeanBaselineRegressor(BaseRegressor[MeanBaselineParams]):
    """Predicts the training-y central tendency for all inputs."""

    def __init__(self, params: MeanBaselineParams) -> None:
        self._params = params
        self._prediction: Optional[float] = None

    @property
    def params(self) -> MeanBaselineParams:
        return self._params

    @property
    def is_fitted(self) -> bool:
        return self._prediction is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanBaselineRegressor":
        if y.empty:
            raise ValueError(
                "MeanBaselineRegressor.fit received an empty target series. "
                "Pass at least one training row."
            )
        if self._params.statistic == "mean":
            self._prediction = float(y.mean())
        else:  # "median" — validated by params.__post_init__
            self._prediction = float(y.median())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._prediction is None:
            raise RuntimeError("MeanBaselineRegressor.predict called before .fit().")
        return pd.Series(np.full(len(X), self._prediction, dtype=float), index=X.index)

    def _state(self) -> object:
        return {"prediction": self._prediction}

    @classmethod
    def _from_state(
        cls, params: MeanBaselineParams, state: object
    ) -> "MeanBaselineRegressor":
        if not isinstance(state, dict) or "prediction" not in state:
            raise ValueError(
                f"MeanBaselineRegressor._from_state expected "
                f"{{'prediction': float}}, got {type(state).__name__}."
            )
        instance = cls(params)
        instance._prediction = float(state["prediction"])
        return instance
