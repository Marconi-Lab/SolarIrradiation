"""Evaluation — post-training metrics, evaluator, and generic plots.

Public entry points:

* :class:`Metric` and the seven concrete metrics (:class:`RMSE`,
  :class:`MAE`, :class:`R2`, :class:`IndexOfAgreement`,
  :class:`MeanBiasError`, :class:`NormalisedRMSE`,
  :class:`NormalisedMAE`) — frozen, stateless scalar functions.
* :data:`DEFAULT_METRICS` — the seven-metric tuple in paper-table order.
* :class:`Evaluator` — score multiple named predictions against an
  observed series across multiple splits; returns a tidy DataFrame.

The ``plots`` submodule holds generic evaluation plot helpers
(predicted-vs-observed, per-station metric bars, covariate-shift KDE,
feature importances, …). Import from :mod:`susse.evaluation.plots`
directly when needed.
"""

from .evaluator import Evaluator
from .metrics import (
    DEFAULT_METRICS,
    MAE,
    R2,
    RMSE,
    IndexOfAgreement,
    MeanBiasError,
    Metric,
    NormalisedMAE,
    NormalisedRMSE,
)

__all__ = [
    "DEFAULT_METRICS",
    "Evaluator",
    "IndexOfAgreement",
    "MAE",
    "MeanBiasError",
    "Metric",
    "NormalisedMAE",
    "NormalisedRMSE",
    "R2",
    "RMSE",
]
