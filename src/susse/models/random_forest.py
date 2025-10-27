from __future__ import annotations

from .base import  BaseRegressor

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor


class RFRegressor(BaseRegressor):
    """RandomForestRegressor wrapper implementing BaseRegressor interface."""

    def __init__(self, **params: Any) -> None:
        if "random_state" not in params:
            params["random_state"] = 42
        self.estimator = RandomForestRegressor(**params)

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.asarray(self.estimator.predict(X), dtype=float)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path / "model.joblib")
        (path / "model_type.txt").write_text("random_forest", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> RFRegressor:
        est: RandomForestRegressor = joblib.load(path / "model.joblib")
        obj = RFRegressor()
        obj.estimator = est
        return obj

    def get_params(self) -> Dict[str, Any]:
        return self.estimator.get_params(deep=True)
