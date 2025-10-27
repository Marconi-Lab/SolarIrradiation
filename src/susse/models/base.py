from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import logging
import math

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# ---------------------------------------------------------------------------
# Model abstraction
# ---------------------------------------------------------------------------


class BaseRegressor:
    """Abstract base for regressors. Concrete subclasses wrap actual estimators."""

    def fit(self, X: NDArray[np.float_], y: NDArray[np.float_]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X: NDArray[np.float_]) -> NDArray[np.float_]:  # pragma: no cover - interface
        raise NotImplementedError

    def save(self, path: Path) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def load(path: Path) -> BaseRegressor:  # pragma: no cover - interface
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

