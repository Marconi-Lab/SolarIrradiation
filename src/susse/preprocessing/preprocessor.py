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

from ..configs import FeatureConfig, ModelConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Feature specification (immutable contract after Preprocessor.fit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable description of the feature pipeline after fitting.

    Attributes
    ----------
    ordered_feature_names: full list of expanded feature names after transforms
        (e.g., with one-hot categories, temporal encodings), in strict order.
    numeric_source_cols: numeric columns used prior to transforms.
    categorical_source_cols: categorical columns used prior to transforms.
    temporal_encoded: whether temporal encodings were added.
    spatial_encoded: whether spatial encodings were added.
    """

    ordered_feature_names: List[str]
    numeric_source_cols: List[str]
    categorical_source_cols: List[str]
    temporal_encoded: bool
    spatial_encoded: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> FeatureSpec:
        obj = json.loads(s)
        return FeatureSpec(**obj)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class Preprocessor:
    """Encapsulates pandas-based preparation into model-ready NumPy arrays.

    Public contract exposes only NumPy arrays and a FeatureSpec.
    """

    def __init__(self, feature_cfg: FeatureConfig) -> None:
        self._cfg = feature_cfg
        self._fitted: bool = False
        self._spec: Optional[FeatureSpec] = None
        self._pipeline: Optional[ColumnTransformer] = None

    # ---- Utility: temporal encoding -------------------------------------------------
    @staticmethod
    def _temporal_features(ts: pd.Series) -> pd.DataFrame:
        ts = pd.to_datetime(ts, utc=False)
        # Month-of-year encoding
        month = ts.dt.month.astype(int)
        # Day-of-year encoding
        dayofyear = ts.dt.dayofyear.astype(int)
        # Cyclical encodings
        month_sin = np.sin(2 * np.pi * (month / 12.0))
        month_cos = np.cos(2 * np.pi * (month / 12.0))
        doy_sin = np.sin(2 * np.pi * (dayofyear / 365.0))
        doy_cos = np.cos(2 * np.pi * (dayofyear / 365.0))
        return pd.DataFrame(
            {
                "month": month,
                "dayofyear": dayofyear,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "doy_sin": doy_sin,
                "doy_cos": doy_cos,
            }
        )

    # ---- Utility: spatial encoding --------------------------------------------------
    @staticmethod
    def _spatial_features(lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
        # Simple expansions; can be replaced with projections or RBFs later
        lat2 = lat.astype(float) ** 2
        lon2 = lon.astype(float) ** 2
        lat_lon = lat.astype(float) * lon.astype(float)
        return pd.DataFrame(
            {
                "lat": lat.astype(float),
                "lon": lon.astype(float),
                "lat2": lat2,
                "lon2": lon2,
                "lat_lon": lat_lon,
            }
        )

    # ---- Resampling ---------------------------------------------------------------
    def _maybe_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self._cfg
        if cfg.time_column and cfg.resample_rule:
            if cfg.time_column not in df.columns:
                raise KeyError(
                    f"time_column '{cfg.time_column}' not found in DataFrame for resampling"
                )
            df = df.copy()
            df[cfg.time_column] = pd.to_datetime(df[cfg.time_column], utc=False)
            df = df.set_index(cfg.time_column)
            if cfg.resample_aggs is None:
                # Default: mean for numeric, first for categorical
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                default_aggs: Dict[str, str] = {c: "mean" for c in numeric_cols}
                # Anything else gets 'first'
                for c in df.columns:
                    if c not in default_aggs:
                        default_aggs[c] = "first"
                aggs = default_aggs
            else:
                aggs = cfg.resample_aggs
            df = df.resample(cfg.resample_rule).agg(aggs)
            df = df.reset_index().rename(columns={cfg.time_column: cfg.time_column})
        return df

    # ---- Fit/Transform ------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> FeatureSpec:
        cfg = self._cfg
        df = self._maybe_resample(df)

        # Construct working dataframe containing requested inputs
        missing = [c for c in cfg.input_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing input columns: {missing}")

        X_df = df[cfg.input_columns].copy()

        # Optional engineered temporal features
        engineered: List[pd.DataFrame] = []
        if cfg.add_temporal_encoding and cfg.time_column is not None:
            if cfg.time_column not in df.columns:
                raise KeyError(
                    f"time_column '{cfg.time_column}' not found for temporal encoding"
                )
            engineered.append(self._temporal_features(df[cfg.time_column]))

        # Optional spatial features
        if cfg.add_spatial_encoding and cfg.spatial_columns is not None:
            lat_col, lon_col = cfg.spatial_columns
            if lat_col not in df.columns or lon_col not in df.columns:
                raise KeyError(
                    f"spatial columns {cfg.spatial_columns} not found for spatial encoding"
                )
            engineered.append(self._spatial_features(df[lat_col], df[lon_col]))

        if engineered:
            X_df = pd.concat([X_df.reset_index(drop=True)] + [e.reset_index(drop=True) for e in engineered], axis=1)

        # Split into numeric/categorical based on dtype
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

        transformers: List[Tuple[str, Pipeline, List[str]]] = []

        num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if cfg.scale_numeric:
            num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        num_pipe = Pipeline(num_steps)
        transformers.append(("num", num_pipe, numeric_cols))

        if categorical_cols:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", cat_pipe, categorical_cols))

        pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        pipeline.fit(X_df)

        # Derive expanded feature names
        feature_names: List[str] = []
        # Numeric
        feature_names.extend(numeric_cols)
        # Categorical (after OHE)
        if categorical_cols:
            ohe: OneHotEncoder = pipeline.named_transformers_["cat"].named_steps["ohe"]
            ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(ohe_names)

        # Create and store spec
        spec = FeatureSpec(
            ordered_feature_names=feature_names,
            numeric_source_cols=numeric_cols,
            categorical_source_cols=categorical_cols,
            temporal_encoded=bool(cfg.add_temporal_encoding and cfg.time_column),
            spatial_encoded=bool(cfg.add_spatial_encoding and cfg.spatial_columns),
        )

        self._pipeline = pipeline
        self._spec = spec
        self._fitted = True
        return spec

    def transform(self, df: pd.DataFrame) -> NDArray[np.float_]:
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        df = self._maybe_resample(df)

        cfg = self._cfg
        missing = [c for c in cfg.input_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing input columns: {missing}")

        X_df = df[cfg.input_columns].copy()
        engineered: List[pd.DataFrame] = []
        if cfg.add_temporal_encoding and cfg.time_column is not None:
            engineered.append(self._temporal_features(df[cfg.time_column]))
        if cfg.add_spatial_encoding and cfg.spatial_columns is not None:
            lat_col, lon_col = cfg.spatial_columns
            engineered.append(self._spatial_features(df[lat_col], df[lon_col]))
        if engineered:
            X_df = pd.concat([X_df.reset_index(drop=True)] + [e.reset_index(drop=True) for e in engineered], axis=1)

        X = self._pipeline.transform(X_df)
        X = np.asarray(X, dtype=float)
        return X

    def fit_transform(self, df: pd.DataFrame) -> Tuple[NDArray[np.float_], Optional[NDArray[np.float_]], FeatureSpec]:
        spec = self.fit(df)
        X = self.transform(df)
        y: Optional[NDArray[np.float_]] = None
        if self._cfg.target_column is not None:
            if self._cfg.target_column not in df.columns:
                raise KeyError(f"Target column '{self._cfg.target_column}' not found.")
            y = np.asarray(df[self._cfg.target_column].values, dtype=float)
        return X, y, spec

    # Persistence of the fitted preprocessor (pipeline + spec)
    def save(self, path: Path) -> None:
        if not self._fitted or self._pipeline is None or self._spec is None:
            raise RuntimeError("Preprocessor must be fitted before save().")
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path / "preprocessor_pipeline.joblib")
        (path / "feature_spec.json").write_text(self._spec.to_json(), encoding="utf-8")
        (path / "feature_cfg.json").write_text(json.dumps(asdict(self._cfg), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> Tuple[Preprocessor, FeatureSpec]:
        pipeline: ColumnTransformer = joblib.load(path / "preprocessor_pipeline.joblib")
        spec = FeatureSpec.from_json((path / "feature_spec.json").read_text(encoding="utf-8"))
        cfg_dict = json.loads((path / "feature_cfg.json").read_text(encoding="utf-8"))
        pre = Preprocessor(FeatureConfig(**cfg_dict))
        pre._pipeline = pipeline
        pre._spec = spec
        pre._fitted = True
        return pre, spec

