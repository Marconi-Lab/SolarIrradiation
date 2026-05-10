"""Paper-style evaluation metrics for GHI bias-correction analysis.

Used by analysis notebooks (e.g.
``notebooks/papers/mukiibi_mikelson_2026/01_recomputation.ipynb``) when
the standard MAE / RMSE / R² triplet that
:func:`susse.training.score_predictions` already provides isn't enough —
in particular when a published baseline is reported in normalised /
agreement-style metrics.

All functions:

* take aligned ``y_true`` / ``y_pred`` :class:`pandas.Series` (or
  array-likes); rows where either side is NaN are dropped before scoring
* return a single ``float`` (NaN when there's no data left after the NaN
  filter, or when a denominator collapses to zero)
* are pure and stateless — safe to call from any notebook or script

The trainer-internal :func:`susse.training.score_predictions` is kept
separate (it returns a :class:`ScoreSet` value object suited to the
trainer's MAE/RMSE/R² triplet); these helpers are stand-alone for
analysis use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_arrays(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to aligned 1-D float arrays with NaN rows dropped.

    Treating NaN-bearing rows as scoreable would silently bias every
    metric, so we drop them up front. Length and (where applicable)
    pandas-index alignment is the caller's responsibility — this helper
    only guards against shape mismatches.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true shape {yt.shape} does not match y_pred shape {yp.shape}. "
            f"Align by index (e.g. with `y_pred = y_pred.loc[y_true.index]`) "
            f"before scoring."
        )
    valid = ~(np.isnan(yt) | np.isnan(yp))
    return yt[valid], yp[valid]


def index_of_agreement(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """Willmott's Index of Agreement (IOA), bounded in [0, 1].

    IOA = 1 − Σ(O − P)² / Σ(|P − Ō| + |O − Ō|)²,  where Ō = mean(O).

    A value of 1 is perfect agreement; 0 is no agreement. Unlike R², IOA
    rewards matching the *magnitude* of the observed series, not just
    the linear correlation with it — which is why Mukiibi & Mikelson
    (2026) report it as the headline metric for the Katongole
    monthly-climatology validation: temporal aggregation collapses
    variance and depresses R², but IOA stays meaningful.

    Returns NaN when ``y_true`` is constant (denominator is zero) or
    when no rows survive the NaN filter.
    """
    yt, yp = _aligned_arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    obs_mean = yt.mean()
    numerator = float(((yt - yp) ** 2).sum())
    denominator = float(((np.abs(yp - obs_mean) + np.abs(yt - obs_mean)) ** 2).sum())
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - numerator / denominator


def mean_bias_error(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """Mean bias error MBE = mean(P − O).

    Positive values mean the prediction over-estimates the observation;
    negative means under-estimate. Distinct from MAE in that errors of
    opposite sign cancel — which is exactly the property the paper uses
    to flag systematic bias in the raw satellite products vs. the
    bias-corrected RF.
    """
    yt, yp = _aligned_arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float((yp - yt).mean())


def normalised_rmse(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """RMSE expressed as a percentage of the mean of ``y_true``.

    nRMSE = 100 × RMSE / mean(O). Standard convention in solar resource
    assessment: lets you compare model accuracy across regions whose
    absolute GHI levels differ. Returns NaN when mean(O) is zero.
    """
    yt, yp = _aligned_arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    obs_mean = yt.mean()
    if obs_mean == 0.0:
        return float("nan")
    rmse = float(np.sqrt(((yp - yt) ** 2).mean()))
    return 100.0 * rmse / obs_mean


def normalised_mae(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> float:
    """MAE expressed as a percentage of the mean of ``y_true``.

    nMAE = 100 × MAE / mean(O). Same normalisation convention as
    :func:`normalised_rmse`. Returns NaN when mean(O) is zero.
    """
    yt, yp = _aligned_arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    obs_mean = yt.mean()
    if obs_mean == 0.0:
        return float("nan")
    mae = float(np.abs(yp - yt).mean())
    return 100.0 * mae / obs_mean
