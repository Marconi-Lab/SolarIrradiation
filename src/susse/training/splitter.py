"""Train / val splitters — produce ``(train_idx, val_idx)`` from a preprocessed dataset.

A :class:`Splitter` is an immutable parameterised object that, when
applied to a :class:`PreprocessedDataset`, returns two disjoint
:class:`pandas.Index` subsets covering its training and validation
folds. The base class is an ABC; every concrete splitter is a frozen
dataclass so its configuration is hashable, inspectable, and
serialisable into :class:`TrainingMetadata.splitter_name`.

Five concrete splitters are shipped:

* :class:`RandomSplitter` — uniformly shuffled fraction. The cheapest
  baseline; ignores all structure in the data.
* :class:`TemporalSplitter` — rows on or after ``split_date`` go to
  val. The right default for time-series generalisation.
* :class:`StationLOSOSplitter` — leave-one-station-out. The right
  default for spatial generalisation when one station's worth of
  rows is enough to score against.
* :class:`SpatialSpreadHoldoutSplitter` — hold out the ``n_holdout``
  stations with maximum pairwise haversine spread. A stronger spatial
  test than LOSO because the held-out set spans the region.
* :class:`SpatialBlockSplitter` — hold out every row whose
  ``block_column`` falls in a named set (typically country or admin
  region). The right default when blocks differ on a known
  categorical lever (climate zone, terrain class, …).

All five subclass :class:`Splitter` and implement the same
``(name, split)`` surface, so the :class:`susse.training.Trainer`
can swap between them without code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd

from ..preprocessing import PreprocessedDataset

# ---------------------------------------------------------------------------
# Abstract base.
# ---------------------------------------------------------------------------


class Splitter(ABC):
    """Strategy that splits a :class:`PreprocessedDataset` into train + val folds.

    Concrete subclasses are frozen dataclasses carrying their
    configuration. :attr:`name` is the stable string recorded in
    :class:`TrainingMetadata.splitter_name`; it must encode the
    splitter's identity so two runs with the same name are guaranteed
    to have used the same configuration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier of this splitter's configuration."""

    @abstractmethod
    def split(self, processed: PreprocessedDataset) -> tuple[pd.Index, pd.Index]:
        """Return ``(train_idx, val_idx)`` — disjoint subsets of ``processed.df.index``.

        Args:
            processed: The dataset to split.

        Returns:
            Two pandas indices: training rows first, validation rows
            second. The union does not have to equal
            ``processed.df.index`` (some splitters drop rows that
            belong to neither fold), but the intersection must be
            empty.

        Raises:
            ValueError: When the configuration cannot produce a valid
                split on this dataset (missing column, empty fold,
                unknown identifier). The message must include the
                remediation step.
        """

    def __call__(self, processed: PreprocessedDataset) -> tuple[pd.Index, pd.Index]:
        """Alias for :meth:`split`. Lets a :class:`Splitter` be passed to
        any callable-typed slot."""
        return self.split(processed)


# ---------------------------------------------------------------------------
# Concrete splitters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomSplitter(Splitter):
    """Uniformly shuffled validation fraction.

    Attributes:
        val_fraction: Fraction of rows that go to val. Must be in
            ``(0, 1)``.
        random_state: Seed for :func:`numpy.random.default_rng`.
            Re-running with the same seed yields the same split.
    """

    val_fraction: float
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                f"RandomSplitter.val_fraction={self.val_fraction} is not in "
                f"(0, 1). Pick a fraction strictly between 0 and 1 (e.g. 0.2 "
                f"for an 80/20 split)."
            )

    @property
    def name(self) -> str:
        return f"random[val={self.val_fraction:.2f},seed={self.random_state}]"

    def split(self, processed: PreprocessedDataset) -> tuple[pd.Index, pd.Index]:
        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(processed.df.index.to_numpy())
        n_val = max(1, int(round(self.val_fraction * len(shuffled))))
        val_idx = pd.Index(shuffled[:n_val])
        train_idx = pd.Index(shuffled[n_val:])
        return train_idx, val_idx


@dataclass(frozen=True)
class TemporalSplitter(Splitter):
    """Hold out every row on or after ``split_date``.

    Attributes:
        split_date: Boundary date. Rows whose ``date_column`` value is
            ``>= split_date`` go to val; everything earlier goes to
            train.
        date_column: Name of the column carrying the row's date.
            Default ``"date"`` matches the convention from
            :class:`FeatureService`.
    """

    split_date: date
    date_column: str = "date"

    @property
    def name(self) -> str:
        return f"temporal[val>={self.split_date.isoformat()}]"

    def split(self, processed: PreprocessedDataset) -> tuple[pd.Index, pd.Index]:
        if self.date_column not in processed.df.columns:
            raise ValueError(
                f"TemporalSplitter needs column {self.date_column!r} on "
                f"processed.df, but it was not found. Add the date column "
                f"to FeatureSpec.id_columns or pass "
                f"`date_column=` matching your column name."
            )
        dates = pd.to_datetime(processed.df[self.date_column]).dt.date
        is_val = dates >= self.split_date
        return (
            processed.df.index[~is_val],
            processed.df.index[is_val],
        )


@dataclass(frozen=True)
class StationLOSOSplitter(Splitter):
    """Leave-one-station-out: hold out every row from a named station.

    Attributes:
        held_out_station: Identifier whose rows go into the val fold.
        location_column: Column on ``processed.df`` carrying the
            station identifier. Default ``"location"``.
    """

    held_out_station: str
    location_column: str = "location"

    @property
    def name(self) -> str:
        return f"station_loso[{self.held_out_station}]"

    def split(self, processed: PreprocessedDataset) -> tuple[pd.Index, pd.Index]:
        if self.location_column not in processed.df.columns:
            raise ValueError(
                f"StationLOSOSplitter needs column {self.location_column!r} "
                f"on processed.df, but it was not found. Add 'location' "
                f"to FeatureSpec.id_columns or rebuild the dataset."
            )
        mask = processed.df[self.location_column] == self.held_out_station
        if not mask.any():
            sample = sorted(
                processed.df[self.location_column].dropna().unique().tolist()
            )[:8]
            raise ValueError(
                f"StationLOSOSplitter: station "
                f"{self.held_out_station!r} has no rows in this dataset. "
                f"Available stations (first 8): {sample}. Use "
                f"StationLOSOSplitter.auto_pick_largest(processed) to "
                f"choose one programmatically."
            )
        return (
            processed.df.index[~mask],
            processed.df.index[mask],
        )

    @classmethod
    def auto_pick_largest(
        cls,
        processed: PreprocessedDataset,
        *,
        location_column: str = "location",
    ) -> "StationLOSOSplitter":
        """Construct a splitter held out on the station with the most rows.

        Deterministic: ties are broken alphabetically by station name so
        a re-run on the same data yields the same splitter.
        """
        if location_column not in processed.df.columns:
            raise ValueError(
                f"auto_pick_largest needs column {location_column!r} on "
                f"processed.df, but it was not found. Either include "
                f"'location' in FeatureSpec.id_columns, or pass "
                f"`location_column=` matching your column."
            )
        counts = processed.df.groupby(location_column).size()
        if counts.empty:
            raise ValueError(
                "PreprocessedDataset is empty; cannot pick a holdout station."
            )
        sorted_counts = counts.sort_index().sort_values(ascending=False, kind="stable")
        return cls(
            held_out_station=str(sorted_counts.index[0]),
            location_column=location_column,
        )


@dataclass(frozen=True)
class SpatialSpreadHoldoutSplitter(Splitter):
    """Hold out the ``n_holdout`` stations with maximum pairwise haversine spread.

    Selects the validation stations greedily: starting from the
    farthest-apart pair, each subsequent station is the one that
    maximises the minimum haversine distance to those already chosen.
    For ``n_holdout=2`` this is just "the most distant pair".

    Attributes:
        n_holdout: Number of stations to hold out. Must be ``>= 2``.
        location_column: Station-identifier column. Default ``"location"``.
        lat_column: Latitude column. Default ``"lat"``.
        lon_column: Longitude column. Default ``"lon"``.
    """

    _EARTH_RADIUS_KM: ClassVar[float] = 6371.0

    n_holdout: int
    location_column: str = "location"
    lat_column: str = "lat"
    lon_column: str = "lon"

    def __post_init__(self) -> None:
        if self.n_holdout < 2:
            raise ValueError(
                f"SpatialSpreadHoldoutSplitter.n_holdout={self.n_holdout} "
                f"is below the minimum of 2. With one station the "
                f"'spread' criterion is degenerate; use "
                f"StationLOSOSplitter for n=1."
            )

    @property
    def name(self) -> str:
        return f"spatial_spread[n={self.n_holdout}]"

    def split(self, processed: PreprocessedDataset) -> tuple[pd.Index, pd.Index]:
        needed = (self.location_column, self.lat_column, self.lon_column)
        missing = [c for c in needed if c not in processed.df.columns]
        if missing:
            raise ValueError(
                f"SpatialSpreadHoldoutSplitter needs columns {needed} on "
                f"processed.df; missing: {missing}. Add the station "
                f"identifier and its coordinates to FeatureSpec.id_columns."
            )
        station_coords = (
            processed.df[list(needed)]
            .drop_duplicates(subset=self.location_column)
            .reset_index(drop=True)
        )
        if len(station_coords) < self.n_holdout:
            raise ValueError(
                f"SpatialSpreadHoldoutSplitter: only "
                f"{len(station_coords)} unique stations in dataset; "
                f"cannot hold out {self.n_holdout}. Lower n_holdout or "
                f"ingest more stations."
            )
        holdout_names = self._greedy_spread_selection(station_coords)
        mask = processed.df[self.location_column].isin(holdout_names)
        return (
            processed.df.index[~mask],
            processed.df.index[mask],
        )

    def _greedy_spread_selection(self, station_coords: pd.DataFrame) -> tuple[str, ...]:
        """Pick stations by greedy max-min haversine spread."""
        lats = station_coords[self.lat_column].to_numpy()
        lons = station_coords[self.lon_column].to_numpy()
        names = station_coords[self.location_column].to_numpy()
        distances = self._haversine_matrix(lats, lons)
        # Seed with the farthest-apart pair.
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        selected_indices = [int(i), int(j)]
        while len(selected_indices) < self.n_holdout:
            min_dist_to_selected = distances[:, selected_indices].min(axis=1)
            min_dist_to_selected[selected_indices] = -np.inf
            selected_indices.append(int(np.argmax(min_dist_to_selected)))
        return tuple(str(names[k]) for k in selected_indices)

    @classmethod
    def _haversine_matrix(cls, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Pairwise haversine distance matrix in kilometres."""
        phi = np.radians(lats)
        lam = np.radians(lons)
        d_phi = phi[:, None] - phi[None, :]
        d_lam = lam[:, None] - lam[None, :]
        a = (
            np.sin(d_phi / 2) ** 2
            + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(d_lam / 2) ** 2
        )
        return 2.0 * cls._EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@dataclass(frozen=True)
class SpatialBlockSplitter(Splitter):
    """Hold out every row whose ``block_column`` falls in ``val_blocks``.

    Used when the dataset carries a categorical block label
    (country, climate zone, terrain class) and the val fold should
    be a named subset of those blocks.

    Attributes:
        val_blocks: Block identifiers whose rows go to val.
        block_column: Column carrying the block identifier.
    """

    val_blocks: tuple[str, ...]
    block_column: str

    def __post_init__(self) -> None:
        if not self.val_blocks:
            raise ValueError(
                "SpatialBlockSplitter.val_blocks is empty. Pass at least "
                "one block identifier to send to the val fold."
            )

    @property
    def name(self) -> str:
        return f"spatial_block[val={','.join(self.val_blocks)}]"

    def split(self, processed: PreprocessedDataset) -> tuple[pd.Index, pd.Index]:
        if self.block_column not in processed.df.columns:
            raise ValueError(
                f"SpatialBlockSplitter needs column {self.block_column!r} "
                f"on processed.df, but it was not found. Add the block "
                f"identifier to FeatureSpec.id_columns or rebuild the "
                f"dataset with the block column included."
            )
        all_blocks = set(processed.df[self.block_column].dropna().unique())
        unknown = set(self.val_blocks) - all_blocks
        if unknown:
            sample = sorted(all_blocks)[:8]
            raise ValueError(
                f"SpatialBlockSplitter: blocks {sorted(unknown)} not "
                f"found in column {self.block_column!r}. Available "
                f"blocks (first 8): {sample}."
            )
        mask = processed.df[self.block_column].isin(self.val_blocks)
        return (
            processed.df.index[~mask],
            processed.df.index[mask],
        )
