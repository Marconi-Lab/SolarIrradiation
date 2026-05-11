"""Splitter callables — produce ``(train_idx, val_idx)`` from a preprocessed dataset.

A :data:`Splitter` is any callable that takes a :class:`PreprocessedDataset`
and returns two disjoint :class:`pandas.Index` subsets covering the
train and val folds. The :class:`susse.training.Trainer` accepts
arbitrary splitters by design — NB 06 will define the full splitter
abstraction (random / temporal / station-LOSO / spatial-block); this
module provides the minimum surface NB 05 needs.

The two helpers here cover the demo path:

* :func:`make_station_loso_splitter` — leave-one-station-out, the
  validation strategy NB 04 inlined.
* :func:`auto_pick_largest_station` — selects the station with the
  most rows in the dataset, useful for the densest possible LOSO val
  set.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from ..preprocessing import PreprocessedDataset

Splitter = Callable[[PreprocessedDataset], tuple[pd.Index, pd.Index]]


def auto_pick_largest_station(
    processed: PreprocessedDataset,
    *,
    location_column: str = "location",
) -> str:
    """Return the station with the most rows in ``processed.df``.

    Useful as the held-out station for a station-LOSO split when you
    want the densest possible val series (e.g. for an annual-profile
    plot). Deterministic: ties are broken by the alphabetic order of
    station names.
    """
    if location_column not in processed.df.columns:
        raise ValueError(
            f"PreprocessedDataset has no column {location_column!r}. "
            f"Either include 'location' in FeatureSpec.id_columns, or "
            f"pass `location_column=` matching the column you store "
            f"station identifiers under."
        )
    counts = processed.df.groupby(location_column).size()
    if counts.empty:
        raise ValueError("PreprocessedDataset is empty; cannot pick a holdout station.")
    # ``idxmax`` is NOT deterministic across ties; sort the index first
    # so the returned name is reproducible across runs.
    counts_sorted = counts.sort_index().sort_values(
        ascending=False,
        kind="stable",
    )
    return str(counts_sorted.index[0])


def make_station_loso_splitter(
    held_out_station: str,
    *,
    location_column: str = "location",
) -> Splitter:
    """Build a splitter that holds out a single station's rows.

    Args:
        held_out_station: The station identifier whose rows go into the
            val fold. Every other station's rows go into train.
        location_column: Column on ``processed.df`` that carries the
            station identifier. Default ``"location"`` matches the
            convention from :class:`susse.datasets.FeatureService`.

    Returns:
        A :data:`Splitter` whose ``__name__`` is
        ``"station_loso[<held_out>]"`` so :class:`TrainingMetadata`
        can record the configuration verbatim.

    Raises:
        ValueError: If ``held_out_station`` has no rows in the
            dataset, or if ``location_column`` is missing. The error
            message lists a sample of available stations.
    """

    def splitter(
        processed: PreprocessedDataset,
    ) -> tuple[pd.Index, pd.Index]:
        if location_column not in processed.df.columns:
            raise ValueError(
                f"station-LOSO splitter needs column {location_column!r} on "
                f"processed.df, but it was not found. Add 'location' to "
                f"FeatureSpec.id_columns or rebuild the dataset."
            )
        mask = processed.df[location_column] == held_out_station
        if not mask.any():
            sample = sorted(processed.df[location_column].dropna().unique().tolist())[
                :8
            ]
            raise ValueError(
                f"Station {held_out_station!r} has no rows in this "
                f"dataset. Available stations (first 8): {sample}. "
                f"Use `auto_pick_largest_station(processed)` to choose "
                f"one programmatically."
            )
        train_idx = processed.df.index[~mask]
        val_idx = processed.df.index[mask]
        return train_idx, val_idx

    splitter.__name__ = f"station_loso[{held_out_station}]"
    return splitter
