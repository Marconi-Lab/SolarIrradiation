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
# MLflow logging shim (no-op unless enabled)
# ---------------------------------------------------------------------------


class MLflowLogger:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._mlflow = None
        if enabled:
            try:
                import mlflow  # type: ignore

                self._mlflow = mlflow
            except Exception:  # pragma: no cover - optional
                logging.warning("MLflow not available; disabling MLflow logging.")
                self.enabled = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.log_artifact(str(path))

