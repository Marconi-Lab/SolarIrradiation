"""Model layer — typed params, ABC regressor, factory, three concrete models.

Public surface:

* :class:`BaseRegressor`, :class:`BaseModelParams` — the contracts.
* :class:`ModelKind` — discriminator + dispatch enum.
* Concrete params: :class:`MeanBaselineParams`, :class:`RandomForestParams`,
  :class:`LinearParams`.
* Concrete regressors: :class:`MeanBaselineRegressor`,
  :class:`RandomForestRegressor`, :class:`LinearRegressor`.
* :class:`ModelFactory` — typed-params → regressor.
* :func:`load_regressor` — restore a fitted regressor from disk without
  knowing the concrete class up-front.
* :func:`params_from_dict` — JSON manifest → typed params, dispatched
  via the persisted ``kind`` tag.
"""

from .base import BaseRegressor, load_regressor
from .factory import ModelFactory
from .linear import LinearRegressor
from .mean_baseline import MeanBaselineRegressor
from .params import (
    BaseModelParams,
    LinearParams,
    MeanBaselineParams,
    ModelKind,
    RandomForestParams,
    params_from_dict,
)
from .random_forest import RandomForestRegressor

__all__ = [
    "BaseModelParams",
    "BaseRegressor",
    "LinearParams",
    "LinearRegressor",
    "MeanBaselineParams",
    "MeanBaselineRegressor",
    "ModelFactory",
    "ModelKind",
    "RandomForestParams",
    "RandomForestRegressor",
    "load_regressor",
    "params_from_dict",
]
