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
# Configuration objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature/target preparation.

    Attributes
    ----------
    input_columns: list of source column names expected in training/inference data.
        These may include numerical, categorical, time, or spatial columns.
    target_column: optional name of the target column (e.g., "ghi").
        Required for training; omit for pure inference datasets.
    time_column: optional timestamp column name. If provided and `resample_rule`
        is set, the data will be resampled/aggregated by this timestamp.
    resample_rule: pandas offset alias (e.g., "H", "15T") for temporal resampling.
        If None, no resampling is performed.
    resample_aggs: optional per-column aggregation mapping for resampling.
        When omitted, numeric columns use mean; categorical columns use first.
    add_temporal_encoding: if True, derive cyclical features from time_column
        (month-of-year, day-of-year; sine/cosine).
    spatial_columns: optional (lat_col, lon_col). If provided and
        `add_spatial_encoding` is True, simple radial basis expansions can be added.
    add_spatial_encoding: if True, add basic spatial features from lat/lon.
    scale_numeric: apply StandardScaler to numeric features (not required for trees,
        but useful for linear/NN models). Retained for future extensibility.
    """

    input_columns: List[str]
    target_column: Optional[str]
    time_column: Optional[str] = None
    resample_rule: Optional[str] = None
    resample_aggs: Optional[Dict[str, str]] = None
    add_temporal_encoding: bool = True
    spatial_columns: Optional[Tuple[str, str]] = None
    add_spatial_encoding: bool = False
    scale_numeric: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model selection and hyperparameters."""

    model_type: str = "random_forest"  # future: "xgboost", "lightgbm", "linear", ...
    params: Dict[str, Any] | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """Training orchestration configuration."""

    test_size: float = 0.2
    random_state: int = 42
    n_splits_cv: int | None = None  # if provided, enables KFold CV
    shuffle: bool = True
