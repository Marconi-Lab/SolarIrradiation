"""Random-forest regressor wrapping sklearn's :class:`RandomForestRegressor`.

Tree-based; no scaling needed (CLAUDE.md / NB 03 decision: scaling lives
in the model wrapper, RF wrappers skip it).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as _SkRF

from .base import BaseRegressor
from .params import RandomForestParams


class RandomForestRegressor(BaseRegressor[RandomForestParams]):
    """Sklearn random forest with the typed-params interface."""

    def __init__(self, params: RandomForestParams) -> None:
        self._params = params
        self._estimator: Optional[_SkRF] = None

    @property
    def params(self) -> RandomForestParams:
        return self._params

    @property
    def is_fitted(self) -> bool:
        return self._estimator is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestRegressor":
        params = self._params
        estimator = _SkRF(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            max_features=params.max_features,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
        )
        estimator.fit(X.values, np.asarray(y.values, dtype=float))
        self._estimator = estimator
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._estimator is None:
            raise RuntimeError("RandomForestRegressor.predict called before .fit().")
        preds = self._estimator.predict(X.values)
        return pd.Series(np.asarray(preds, dtype=float), index=X.index)

    def feature_importances(self, feature_names: tuple[str, ...]) -> pd.Series:
        """Return per-feature importance, indexed by feature_names.

        Args:
            feature_names: Column names of the feature matrix the model
                was fit on, in fit order. Same as
                ``PreprocessedDataset.feature_columns``.

        Returns:
            A :class:`pandas.Series` of impurity-based importances
            (same as sklearn's ``feature_importances_``), summing to 1.0.

        Raises:
            RuntimeError: Called before :meth:`fit`.
            ValueError: If ``feature_names`` length doesn't match the
                fitted model's expected feature count.
        """
        if self._estimator is None:
            raise RuntimeError(
                "RandomForestRegressor.feature_importances called before .fit()."
            )
        importances = self._estimator.feature_importances_
        if len(feature_names) != len(importances):
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but the "
                f"fitted model has {len(importances)} features. The names "
                f"must match what was passed to .fit()."
            )
        return pd.Series(importances, index=list(feature_names))

    def _state(self) -> object:
        return self._estimator

    @classmethod
    def _from_state(
        cls, params: RandomForestParams, state: object
    ) -> "RandomForestRegressor":
        if not isinstance(state, _SkRF):
            raise ValueError(
                f"RandomForestRegressor._from_state expected an sklearn "
                f"RandomForestRegressor, got {type(state).__name__}."
            )
        instance = cls(params)
        instance._estimator = state
        return instance
