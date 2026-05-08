"""Typed model parameter dataclasses + dispatch enum.

Each model has its own frozen :class:`BaseModelParams` subclass holding
its hyperparameters. The :class:`ModelKind` enum routes between them —
construction (``params.kind.model_class()``), deserialisation
(``params.kind.params_type()``), and persistence (``kind`` is the
single string written into ``model_kind.txt`` next to ``state.joblib``).

No string-keyed dicts: a typo in a hyperparameter name surfaces as a
construction-time ``TypeError``, not a silent default value at training
time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid runtime import cycles
    from .base import BaseRegressor


class ModelKind(StrEnum):
    """Discriminator between concrete model implementations.

    Used as the persistence key (written to ``model_kind.txt``) and as
    the dispatch token in :class:`susse.models.ModelFactory`. The enum
    methods own the ``kind → params_type`` and ``kind → model_class``
    mappings — no string-keyed registry, no module-level dispatch dict.
    """

    MEAN_BASELINE = "mean_baseline"
    RANDOM_FOREST = "random_forest"
    LINEAR = "linear"

    def params_type(self) -> type["BaseModelParams"]:
        """Return the params dataclass for this kind."""
        # Local imports break a circular: subclasses define their own
        # `kind` property which references this enum.
        from .params import (  # noqa: PLC0415
            LinearParams, MeanBaselineParams, RandomForestParams,
        )
        return {
            ModelKind.MEAN_BASELINE: MeanBaselineParams,
            ModelKind.RANDOM_FOREST: RandomForestParams,
            ModelKind.LINEAR: LinearParams,
        }[self]

    def model_class(self) -> type["BaseRegressor"]:
        """Return the regressor class that consumes this kind's params."""
        from .linear import LinearRegressor  # noqa: PLC0415
        from .mean_baseline import MeanBaselineRegressor  # noqa: PLC0415
        from .random_forest import RandomForestRegressor  # noqa: PLC0415
        return {
            ModelKind.MEAN_BASELINE: MeanBaselineRegressor,
            ModelKind.RANDOM_FOREST: RandomForestRegressor,
            ModelKind.LINEAR: LinearRegressor,
        }[self]


@dataclass(frozen=True)
class BaseModelParams(ABC):
    """ABC for typed model hyperparameter sets.

    Subclasses are frozen dataclasses carrying only hyperparameters;
    runtime-fitted state lives on the regressor, not the params. The
    :attr:`kind` property is the dispatch token for the factory and the
    persistence layer.
    """

    @property
    @abstractmethod
    def kind(self) -> ModelKind:
        """The :class:`ModelKind` this params object belongs to."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict, with ``kind`` as a tag.

        The ``kind`` tag is what :func:`params_from_dict` uses to route
        deserialisation back to the correct subclass.
        """
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


def params_from_dict(d: dict[str, Any]) -> BaseModelParams:
    """Inverse of :meth:`BaseModelParams.to_dict`.

    Reads the ``kind`` tag, looks up the matching params dataclass, and
    constructs it from the remaining fields.
    """
    if "kind" not in d:
        raise ValueError(
            "params_from_dict expects a 'kind' tag identifying the model "
            "type. Pass a dict produced by BaseModelParams.to_dict()."
        )
    kind = ModelKind(d["kind"])
    params_cls = kind.params_type()
    return params_cls(**{k: v for k, v in d.items() if k != "kind"})


# ---------------------------------------------------------------------------
# Concrete params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanBaselineParams(BaseModelParams):
    """Params for the constant-mean baseline.

    Attributes:
        statistic: Which central tendency to use — ``"mean"`` or
            ``"median"``. Median is more robust to outliers but breaks
            differentiability if a downstream consumer expected
            mean-based predictions; default is ``"mean"``.
    """

    statistic: str = "mean"

    def __post_init__(self) -> None:
        if self.statistic not in ("mean", "median"):
            raise ValueError(
                f"MeanBaselineParams.statistic must be 'mean' or 'median', "
                f"got {self.statistic!r}."
            )

    @property
    def kind(self) -> ModelKind:
        return ModelKind.MEAN_BASELINE


@dataclass(frozen=True)
class RandomForestParams(BaseModelParams):
    """Hyperparameters for sklearn's :class:`RandomForestRegressor`."""

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(
                f"n_estimators={self.n_estimators} must be >= 1."
            )
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(
                f"max_depth={self.max_depth} must be >= 1 or None for unlimited."
            )

    @property
    def kind(self) -> ModelKind:
        return ModelKind.RANDOM_FOREST


@dataclass(frozen=True)
class LinearParams(BaseModelParams):
    """Hyperparameters for the linear regressor.

    Attributes:
        with_scaling: If True (default), fit a :class:`StandardScaler`
            on the training X and apply it before / inverse to inference
            X. Strongly recommended for linear models — otherwise
            features with different magnitudes (kt ~ 0..1 vs GHI in
            kWh/m²/day) bias the coefficients.
        fit_intercept: Sklearn's intercept toggle.
    """

    with_scaling: bool = True
    fit_intercept: bool = True

    @property
    def kind(self) -> ModelKind:
        return ModelKind.LINEAR
