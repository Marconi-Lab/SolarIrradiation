"""Abstract base for all regressors + the generic save/load contract.

Every concrete model is a :class:`BaseRegressor` subclass parameterised
by its typed params dataclass. The ABC pins the contract: ``fit`` →
``predict`` → ``save`` round-trippable to ``load``.

Predictions are returned as :class:`pandas.Series` aligned to the input
``X.index`` so downstream code can join predictions back onto rows by
identifier without keeping a parallel index. (Sklearn returns ndarray
and forces the caller to remember the row order.)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd

from .params import BaseModelParams, params_from_dict

P = TypeVar("P", bound=BaseModelParams)

_PARAMS_FILENAME = "params.json"
_KIND_FILENAME = "model_kind.txt"
_STATE_FILENAME = "state.joblib"


class BaseRegressor(ABC, Generic[P]):
    """ABC for typed regressors.

    Generic over ``P``, the params dataclass type. The bound on ``P``
    keeps the type relationship intact for static checkers without
    forcing every subclass to repeat the parameter type in two places.

    Concrete subclasses accept their typed ``params`` as the single
    constructor argument; the signature is declared here so the factory
    (``ModelKind.model_class()(params)``) type-checks against the ABC.
    """

    @abstractmethod
    def __init__(self, params: P) -> None:
        """Construct an unfitted regressor from its hyperparameter set."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseRegressor[P]":
        """Fit the model on ``(X, y)`` and return ``self`` for chaining.

        Subclasses fitting their own scaler / preprocessor must do it
        here, on this ``X`` only. The wrapper interface is intentionally
        single-fold — leakage protection is the caller's responsibility
        (i.e. only train-fold rows reach :meth:`fit`).
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict for each row in ``X``; index of the result == ``X.index``."""

    @property
    @abstractmethod
    def params(self) -> P:
        """The typed params used to construct this regressor."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """``True`` iff :meth:`fit` has run successfully."""

    # ---- persistence ------------------------------------------------------
    #
    # The base class owns the surface (``save`` / ``load``) and the file
    # layout; concrete subclasses only have to surrender / restore their
    # internal joblib-able state.

    def save(self, dir: Path) -> None:
        """Persist this regressor into ``dir`` (created if absent).

        Layout::

            <dir>/
              model_kind.txt    # ``params.kind.value``
              params.json       # ``params.to_dict()``
              state.joblib      # subclass _state()

        Raises:
            RuntimeError: If the regressor has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Cannot save unfitted {type(self).__name__}: call .fit(X, y) "
                f"before .save(...)."
            )
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        (dir / _KIND_FILENAME).write_text(self.params.kind.value)
        (dir / _PARAMS_FILENAME).write_text(
            json.dumps(self.params.to_dict(), indent=2, sort_keys=True)
        )
        import joblib  # local import — only models that get saved pay the cost

        joblib.dump(self._state(), dir / _STATE_FILENAME)

    @abstractmethod
    def _state(self) -> object:
        """Return the joblib-able fitted state for :meth:`save`.

        Whatever this returns is the only state restored by
        :meth:`_from_state`; everything else must be reconstructible
        from ``params``.
        """

    @classmethod
    @abstractmethod
    def _from_state(cls, params: P, state: object) -> "BaseRegressor[P]":
        """Reconstruct a fitted regressor from its params + saved state."""


def load_regressor(dir: Path) -> BaseRegressor:
    """Inverse of :meth:`BaseRegressor.save`: read the kind, dispatch, restore.

    Doesn't require knowing the concrete class up-front — the layout's
    ``model_kind.txt`` carries that. Useful for inference code that
    receives a model directory without compile-time knowledge of which
    flavour was trained.
    """
    dir = Path(dir)
    kind_str = (dir / _KIND_FILENAME).read_text().strip()
    params_dict = json.loads((dir / _PARAMS_FILENAME).read_text())
    params = params_from_dict(params_dict)
    if params.kind.value != kind_str:
        raise ValueError(
            f"Inconsistent saved model in {dir}: model_kind.txt says "
            f"{kind_str!r} but params.json declares "
            f"{params.kind.value!r}. Re-save the model from a consistent "
            f"in-memory state, or hand-fix the directory."
        )
    import joblib

    state = joblib.load(dir / _STATE_FILENAME)
    cls = params.kind.model_class()
    return cls._from_state(params, state)
