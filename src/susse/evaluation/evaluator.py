"""Evaluator — score multiple prediction series against an observed series.

The :class:`Evaluator` takes a configured tuple of :class:`Metric`
instances and, given an observed series plus a mapping of
``{name: prediction_series}``, returns a tidy DataFrame with one row
per (split × prediction_name).

It is the post-training counterpart of
:func:`susse.training.score_predictions`: that helper produces a
single :class:`ScoreSet` (MAE/RMSE/R²) on one (y_true, y_pred) pair
during training; the Evaluator scores arbitrary metrics across
arbitrary predictions on arbitrary splits, for inspection and paper
tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from .metrics import DEFAULT_METRICS, Metric


class Evaluator:
    """Build a metrics table over named prediction series and named splits.

    Attributes:
        metrics: The tuple of :class:`Metric` instances configured on
            construction. Exposed as a :class:`property` for
            inspection but immutable.

    Example::

        evaluator = Evaluator()  # uses DEFAULT_METRICS
        table = evaluator.score(
            observed=df["y_obs"],
            predictions={
                "RF":   df["y_pred"],
                "NASA": df["sat_ghi_nasa_kwh_m2_day"],
                "CAMS": df["sat_ghi_cams_kwh_m2_day"],
            },
            splits={
                "all rows":          df.index,
                "excluding overlap": df.index[~df["seen_in_training"]],
            },
        )
    """

    def __init__(
        self, *, metrics: Sequence[Metric] = DEFAULT_METRICS
    ) -> None:
        if not metrics:
            raise ValueError(
                "Evaluator needs at least one Metric. Pass "
                "`metrics=DEFAULT_METRICS` or a non-empty tuple of "
                "Metric instances."
            )
        self._metrics: tuple[Metric, ...] = tuple(metrics)

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return self._metrics

    def score(
        self,
        *,
        observed: pd.Series,
        predictions: Mapping[str, pd.Series],
        splits: Mapping[str, pd.Index] | None = None,
    ) -> pd.DataFrame:
        """Compute every metric across every (split × prediction) combination.

        Args:
            observed: Observed values. Its index defines what rows
                the ``splits`` indices select from.
            predictions: Mapping from a human label (the column
                header in the result) to a prediction series aligned
                to ``observed``.
            splits: Mapping from a split label to a pandas Index
                selecting rows of ``observed``. When ``None``, all
                rows are scored together under the label ``"all"``.

        Returns:
            A :class:`pandas.DataFrame` with one row per
            ``(split × prediction)`` pair. Columns: ``"split"``,
            ``"prediction"``, ``"n"`` (row count after NaN filter),
            then one column per metric in :attr:`metrics` order.

        Raises:
            ValueError: When ``predictions`` is empty, when a
                prediction series has a shape mismatch with
                ``observed``, or when a split index references rows
                absent from ``observed``.
        """
        if not predictions:
            raise ValueError(
                "Evaluator.score: `predictions` is empty. Pass at "
                "least one named prediction series."
            )
        resolved_splits = self._resolve_splits(observed, splits)
        rows = [
            self._score_one(
                split_label=split_label,
                pred_label=pred_label,
                observed=observed.loc[split_idx],
                predicted=pred_series.loc[split_idx],
            )
            for split_label, split_idx in resolved_splits.items()
            for pred_label, pred_series in predictions.items()
        ]
        return pd.DataFrame(rows)

    def _resolve_splits(
        self,
        observed: pd.Series,
        splits: Mapping[str, pd.Index] | None,
    ) -> dict[str, pd.Index]:
        if splits is None:
            return {"all": observed.index}
        for label, idx in splits.items():
            if not idx.isin(observed.index).all():
                raise ValueError(
                    f"Evaluator.score: split {label!r} references rows "
                    f"absent from `observed`. Subset the split to "
                    f"`observed.index` first, or fix the split's "
                    f"construction."
                )
        return dict(splits)

    def _score_one(
        self,
        *,
        split_label: str,
        pred_label: str,
        observed: pd.Series,
        predicted: pd.Series,
    ) -> dict[str, object]:
        valid_mask = observed.notna() & predicted.notna()
        n_rows = int(valid_mask.sum())
        row: dict[str, object] = {
            "split": split_label,
            "prediction": pred_label,
            "n": n_rows,
        }
        for metric in self._metrics:
            row[metric.name] = metric.compute(observed, predicted)
        return row
