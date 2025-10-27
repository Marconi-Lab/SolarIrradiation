from __future__ import annotations

from .base import BaseRegressor

from ..configs import ModelConfig
from .random_forest import RFRegressor

class ModelFactory:
    """Factory to build models from ModelConfig."""

    @staticmethod
    def create(cfg: ModelConfig) -> BaseRegressor:
        params = cfg.params or {}
        if cfg.model_type == "random_forest":
            return RFRegressor(**params)
        # Future extensions: xgboost, lightgbm, linear, etc.
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

