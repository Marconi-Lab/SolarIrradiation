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


def rmse(y_true: NDArray[np.float_], y_pred: NDArray[np.float_]) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mbe(y_true: NDArray[np.float_], y_pred: NDArray[np.float_]) -> float:
    return float(np.mean(y_pred - y_true))


def mape_safe(y_true: NDArray[np.float_], y_pred: NDArray[np.float_], eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
