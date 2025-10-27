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
# Trainer / Pipeline
# ---------------------------------------------------------------------------


class Trainer:
    """High-level orchestration for training + inference.

    Public API keeps things simple; the internal steps are composable.
    """

    @staticmethod
    def train(
        df_raw: pd.DataFrame,
        feature_cfg: FeatureConfig,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        mlflow_logger: Optional[MLflowLogger] = None,
    ) -> TrainedBundle:
        # Fit preprocessor
        pre = Preprocessor(feature_cfg)
        X, y, spec = pre.fit_transform(df_raw)
        if y is None:
            raise ValueError("Training requires a target column in FeatureConfig.")

        # Split or CV
        metrics: Dict[str, float] = {}
        if training_cfg.n_splits_cv and training_cfg.n_splits_cv > 1:
            # KFold CV with averaged metrics
            kf = KFold(n_splits=training_cfg.n_splits_cv, shuffle=training_cfg.shuffle, random_state=training_cfg.random_state)
            maes: List[float] = []
            rmses: List[float] = []
            mbes: List[float] = []
            mapes: List[float] = []

            for train_idx, val_idx in kf.split(X):
                model = ModelFactory.create(model_cfg)
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[val_idx])
                maes.append(mean_absolute_error(y[val_idx], y_pred))
                rmses.append(rmse(y[val_idx], y_pred))
                mbes.append(mbe(y[val_idx], y_pred))
                mapes.append(mape_safe(y[val_idx], y_pred))

            metrics = {
                "cv_mae": float(np.mean(maes)),
                "cv_rmse": float(np.mean(rmses)),
                "cv_mbe": float(np.mean(mbes)),
                "cv_mape": float(np.mean(mapes)),
            }

            # Refit on full data
            model = ModelFactory.create(model_cfg)
            model.fit(X, y)
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=training_cfg.test_size, random_state=training_cfg.random_state, shuffle=training_cfg.shuffle
            )
            model = ModelFactory.create(model_cfg)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            metrics = {
                "val_mae": mean_absolute_error(y_val, y_pred),
                "val_rmse": rmse(y_val, y_pred),
                "val_mbe": mbe(y_val, y_pred),
                "val_mape": mape_safe(y_val, y_pred),
            }

        # MLflow logging
        if mlflow_logger is not None and mlflow_logger.enabled:
            mlflow_logger.start_run(run_name="solar-irradiation-train")
            mlflow_logger.log_params({
                "model_type": model_cfg.model_type,
                **(model.get_params()),
                "resample_rule": feature_cfg.resample_rule or "",
                "temporal_encoding": str(feature_cfg.add_temporal_encoding),
                "spatial_encoding": str(feature_cfg.add_spatial_encoding),
                "scale_numeric": str(feature_cfg.scale_numeric),
            })
            mlflow_logger.log_metrics(metrics)

        bundle = TrainedBundle(
            model=model,
            preprocessor=pre,
            feature_spec=spec,
            feature_cfg=feature_cfg,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            versions={
                "sklearn": _try_get_pkg_version("sklearn"),
                "pandas": _try_get_pkg_version("pandas"),
                "numpy": _try_get_pkg_version("numpy"),
            },
        )

        # Save artifacts as MLflow artifacts if enabled
        if mlflow_logger is not None and mlflow_logger.enabled:
            tmp_dir = Path("./_tmp_artifacts")
            bundle.save(tmp_dir, mlflow_logger=mlflow_logger)
            mlflow_logger.end_run()

        return bundle

    @staticmethod
    def predict(df_raw: pd.DataFrame, bundle: TrainedBundle) -> NDArray[np.float_]:
        X = bundle.preprocessor.transform(df_raw)
        y_hat = bundle.model.predict(X)
        return y_hat
