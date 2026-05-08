"""Linear regressor with optional in-wrapper scaling.

The scaler is fitted on whatever ``X`` is passed to :meth:`fit` —
leakage protection is the caller's responsibility (only train-fold rows
should reach :meth:`fit`). This mirrors how a future NN wrapper would
own its own scaling, and keeps :class:`Preprocessor` stateless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as _SkLinear
from sklearn.preprocessing import StandardScaler

from .base import BaseRegressor
from .params import LinearParams


@dataclass
class _LinearState:
    """Internal joblib-able state for :class:`LinearRegressor`."""

    estimator: _SkLinear
    scaler: Optional[StandardScaler]


class LinearRegressor(BaseRegressor[LinearParams]):
    """Sklearn linear regression with optional StandardScaler."""

    def __init__(self, params: LinearParams) -> None:
        self._params = params
        self._state_obj: Optional[_LinearState] = None

    @property
    def params(self) -> LinearParams:
        return self._params

    @property
    def is_fitted(self) -> bool:
        return self._state_obj is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressor":
        X_array = X.values
        if self._params.with_scaling:
            scaler = StandardScaler()
            X_array = scaler.fit_transform(X_array)
        else:
            scaler = None
        estimator = _SkLinear(fit_intercept=self._params.fit_intercept)
        estimator.fit(X_array, np.asarray(y.values, dtype=float))
        self._state_obj = _LinearState(estimator=estimator, scaler=scaler)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._state_obj is None:
            raise RuntimeError("LinearRegressor.predict called before .fit().")
        X_array = X.values
        if self._state_obj.scaler is not None:
            X_array = self._state_obj.scaler.transform(X_array)
        preds = self._state_obj.estimator.predict(X_array)
        return pd.Series(np.asarray(preds, dtype=float), index=X.index)

    def _state(self) -> object:
        return self._state_obj

    @classmethod
    def _from_state(
        cls, params: LinearParams, state: object
    ) -> "LinearRegressor":
        if not isinstance(state, _LinearState):
            raise ValueError(
                f"LinearRegressor._from_state expected a _LinearState, got "
                f"{type(state).__name__}."
            )
        instance = cls(params)
        instance._state_obj = state
        return instance
