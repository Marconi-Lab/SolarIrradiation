"""Construction-time entry point for the model layer.

Single static method that instantiates the right :class:`BaseRegressor`
subclass from a typed :class:`BaseModelParams`. Dispatch goes through
the ``params.kind`` enum's :meth:`model_class` — no string-keyed
registry, no module-level lookup table.
"""

from __future__ import annotations

from .base import BaseRegressor
from .params import BaseModelParams


class ModelFactory:
    """Construct a fresh (unfitted) regressor from a typed params object."""

    @staticmethod
    def create(params: BaseModelParams) -> BaseRegressor:
        regressor_cls = params.kind.model_class()
        return regressor_cls(params)
