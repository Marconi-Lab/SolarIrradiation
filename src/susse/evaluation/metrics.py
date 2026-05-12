"""Scalar evaluation metrics — RMSE, MAE, R², IOA, MBE, normalised RMSE/MAE.

Each metric is a :class:`Metric` subclass, frozen and stateless. The
:attr:`name` property is the column header used in scoring tables;
:meth:`compute` returns the scalar value over aligned
``(y_true, y_pred)`` pairs. All metrics drop NaN-bearing rows before
computing, and return :data:`math.nan` (not zero) in pathological
cases — silent zero would be a gradient dead zone, not a sane default.

The seven concrete metrics:

* :class:`RMSE`, :class:`MAE`, :class:`R2` — the standard regression
  triple. These overlap with :class:`susse.training.ScoreSet`, which
  the :class:`Trainer` records on every val fold; the Metric versions
  here are for post-hoc scoring across multiple prediction series.
* :class:`IndexOfAgreement` — Willmott (1981), bounded ``[0, 1]``.
  Rewards matching the *magnitude* of the observed series, not just
  linear correlation; the paper's headline metric for monthly
  climatology comparisons.
* :class:`MeanBiasError` — mean(P − O). Surfaces systematic bias
  that MAE/RMSE absolute-value away.
* :class:`NormalisedRMSE`, :class:`NormalisedMAE` — RMSE/MAE as a
  percentage of mean(O). Standard convention in solar-resource
  assessment.

:data:`DEFAULT_METRICS` is the seven-metric tuple in paper-table order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

ArrayLike = pd.Series | np.ndarray | list[float]


def _aligned_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to aligned 1-D float arrays with NaN rows dropped.

    Args:
        y_true: Ground truth.
        y_pred: Prediction series. Must share shape with ``y_true``;
            pandas-index alignment is the caller's responsibility.

    Returns:
        ``(yt, yp)`` of equal length with NaN-bearing rows removed.

    Raises:
        ValueError: When the two inputs have different shapes.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true shape {yt.shape} does not match y_pred shape "
            f"{yp.shape}. Align by index (e.g. with "
            f"`y_pred = y_pred.loc[y_true.index]`) before scoring."
        )
    valid = ~(np.isnan(yt) | np.isnan(yp))
    return yt[valid], yp[valid]


class Metric(ABC):
    """A named scalar function of ``(y_true, y_pred)``.

    Concrete subclasses are frozen, value-equal dataclasses so two
    Evaluator instances configured with ``RMSE()`` produce identical
    metric tables.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Column-header string in scoring tables."""

    @abstractmethod
    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Return the scalar metric value over aligned inputs."""

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        return self.compute(y_true, y_pred)


@dataclass(frozen=True)
class RMSE(Metric):
    """Root-mean-square error: ``sqrt(mean((P - O)²))``."""

    @property
    def name(self) -> str:
        return "RMSE"

    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt, yp = _aligned_arrays(y_true, y_pred)
        if yt.size == 0:
            return float("nan")
        return float(np.sqrt(((yp - yt) ** 2).mean()))


@dataclass(frozen=True)
class MAE(Metric):
    """Mean absolute error: ``mean(|P - O|)``."""

    @property
    def name(self) -> str:
        return "MAE"

    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt, yp = _aligned_arrays(y_true, y_pred)
        if yt.size == 0:
            return float("nan")
        return float(np.abs(yp - yt).mean())


@dataclass(frozen=True)
class R2(Metric):
    """Coefficient of determination.

    Returns NaN when the observed series has zero variance (the
    formula's denominator is zero), rather than ``inf`` or an
    arbitrary sentinel.
    """

    @property
    def name(self) -> str:
        return "R²"

    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt, yp = _aligned_arrays(y_true, y_pred)
        if yt.size == 0:
            return float("nan")
        ss_res = float(((yp - yt) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        if ss_tot <= 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class IndexOfAgreement(Metric):
    """Willmott's (1981) Index of Agreement, bounded ``[0, 1]``.

    ``IOA = 1 − Σ(O − P)² / Σ(|P − Ō| + |O − Ō|)²``,  where ``Ō = mean(O)``.

    A value of 1 is perfect agreement; 0 is no agreement. Unlike R²,
    IOA rewards matching the *magnitude* of the observed series, not
    just the linear correlation with it — which is why monthly-climatology
    comparisons (where temporal aggregation collapses variance and
    depresses R²) keep IOA as the headline metric.
    """

    @property
    def name(self) -> str:
        return "IOA"

    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt, yp = _aligned_arrays(y_true, y_pred)
        if yt.size == 0:
            return float("nan")
        obs_mean = yt.mean()
        numerator = float(((yt - yp) ** 2).sum())
        denominator = float(
            ((np.abs(yp - obs_mean) + np.abs(yt - obs_mean)) ** 2).sum()
        )
        if denominator <= 0.0:
            return float("nan")
        return 1.0 - numerator / denominator


@dataclass(frozen=True)
class MeanBiasError(Metric):
    """Mean bias error: ``mean(P - O)``.

    Positive values mean the prediction over-estimates the
    observation; negative means under-estimate. Distinct from MAE in
    that errors of opposite sign cancel — which is exactly the
    property used to flag systematic bias in raw satellite products
    vs. a bias-corrected model.
    """

    @property
    def name(self) -> str:
        return "MBE"

    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt, yp = _aligned_arrays(y_true, y_pred)
        if yt.size == 0:
            return float("nan")
        return float((yp - yt).mean())


@dataclass(frozen=True)
class NormalisedRMSE(Metric):
    """RMSE as a percentage of ``mean(y_true)``.

    ``nRMSE = 100 × RMSE / mean(O)``. Returns NaN when ``mean(O)``
    is zero.
    """

    @property
    def name(self) -> str:
        return "nRMSE_%"

    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt, yp = _aligned_arrays(y_true, y_pred)
        if yt.size == 0:
            return float("nan")
        obs_mean = yt.mean()
        if obs_mean == 0.0:
            return float("nan")
        rmse = float(np.sqrt(((yp - yt) ** 2).mean()))
        return 100.0 * rmse / obs_mean


@dataclass(frozen=True)
class NormalisedMAE(Metric):
    """MAE as a percentage of ``mean(y_true)``.

    ``nMAE = 100 × MAE / mean(O)``. Returns NaN when ``mean(O)``
    is zero.
    """

    @property
    def name(self) -> str:
        return "nMAE_%"

    def compute(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt, yp = _aligned_arrays(y_true, y_pred)
        if yt.size == 0:
            return float("nan")
        obs_mean = yt.mean()
        if obs_mean == 0.0:
            return float("nan")
        mae = float(np.abs(yp - yt).mean())
        return 100.0 * mae / obs_mean


DEFAULT_METRICS: tuple[Metric, ...] = (
    RMSE(),
    NormalisedRMSE(),
    MAE(),
    NormalisedMAE(),
    MeanBiasError(),
    R2(),
    IndexOfAgreement(),
)
"""The seven paper-table metrics in display order."""
