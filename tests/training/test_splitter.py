"""Splitter ABC + concrete subclasses.

The contract every splitter must honour:

* :meth:`Splitter.split` returns disjoint ``(train_idx, val_idx)`` pairs
  that are subsets of ``processed.df.index``.
* :attr:`Splitter.name` is a stable, configuration-specific string.
* Configuration errors raise :class:`ValueError` with a remediation
  hint.
"""

from __future__ import annotations

from datetime import date

import pytest

from susse.preprocessing import PreprocessedDataset
from susse.training import (
    RandomSplitter,
    SpatialBlockSplitter,
    SpatialSpreadHoldoutSplitter,
    StationLOSOSplitter,
    TemporalSplitter,
)


# ---------------------------------------------------------------------------
# RandomSplitter
# ---------------------------------------------------------------------------


class TestRandomSplitter:
    def test_train_val_disjoint_and_cover_all_rows(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        splitter = RandomSplitter(val_fraction=0.25, random_state=0)
        train_idx, val_idx = splitter(toy_processed)
        assert len(train_idx.intersection(val_idx)) == 0
        assert len(train_idx) + len(val_idx) == len(toy_processed.df)

    def test_val_fraction_respected_within_one_row(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        # round(0.25 * 120) == 30 — split must be exactly that size.
        train_idx, val_idx = RandomSplitter(
            val_fraction=0.25, random_state=0
        ).split(toy_processed)
        assert len(val_idx) == 30

    def test_same_seed_gives_same_split(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        a = RandomSplitter(val_fraction=0.2, random_state=11).split(toy_processed)
        b = RandomSplitter(val_fraction=0.2, random_state=11).split(toy_processed)
        assert list(a[0]) == list(b[0])
        assert list(a[1]) == list(b[1])

    def test_name_encodes_configuration(self) -> None:
        assert (
            RandomSplitter(val_fraction=0.2, random_state=11).name
            == "random[val=0.20,seed=11]"
        )

    def test_rejects_invalid_fraction(self) -> None:
        with pytest.raises(ValueError, match="not in"):
            RandomSplitter(val_fraction=0.0)
        with pytest.raises(ValueError, match="not in"):
            RandomSplitter(val_fraction=1.5)


# ---------------------------------------------------------------------------
# TemporalSplitter
# ---------------------------------------------------------------------------


class TestTemporalSplitter:
    def test_rows_on_or_after_split_date_go_to_val(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        # Toy fixture spans 2024-01-01 to 2024-02-29 inclusive.
        # 2024-02-01 boundary splits the date range roughly in half per station.
        train_idx, val_idx = TemporalSplitter(
            split_date=date(2024, 2, 1)
        ).split(toy_processed)
        train_dates = toy_processed.df.loc[train_idx, "date"]
        val_dates = toy_processed.df.loc[val_idx, "date"]
        assert all(d < date(2024, 2, 1) for d in train_dates)
        assert all(d >= date(2024, 2, 1) for d in val_dates)

    def test_missing_date_column_raises_with_remediation(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        with pytest.raises(ValueError, match="FeatureSpec.id_columns"):
            TemporalSplitter(
                split_date=date(2024, 2, 1), date_column="when"
            ).split(toy_processed)

    def test_name_encodes_split_date(self) -> None:
        assert (
            TemporalSplitter(split_date=date(2024, 2, 1)).name
            == "temporal[val>=2024-02-01]"
        )


# ---------------------------------------------------------------------------
# StationLOSOSplitter
# ---------------------------------------------------------------------------


class TestStationLosoSplitter:
    def test_held_out_station_is_the_only_val_station(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        train_idx, val_idx = StationLOSOSplitter("sta_b").split(toy_processed)
        held = toy_processed.df.loc[val_idx, "location"].unique()
        assert list(held) == ["sta_b"]
        assert len(train_idx.intersection(val_idx)) == 0

    def test_unknown_station_raises_with_remediation(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        with pytest.raises(ValueError, match="auto_pick_largest"):
            StationLOSOSplitter("not_a_station").split(toy_processed)

    def test_name_records_held_out_station(self) -> None:
        assert (
            StationLOSOSplitter("sta_b").name == "station_loso[sta_b]"
        )

    def test_auto_pick_largest_breaks_ties_alphabetically(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        # Toy fixture has 60 rows for each station — tied. Alphabetic.
        splitter = StationLOSOSplitter.auto_pick_largest(toy_processed)
        assert splitter.held_out_station == "sta_a"

    def test_auto_pick_largest_with_custom_location_column_raises(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            StationLOSOSplitter.auto_pick_largest(
                toy_processed, location_column="station"
            )


# ---------------------------------------------------------------------------
# SpatialSpreadHoldoutSplitter
# ---------------------------------------------------------------------------


class TestSpatialSpreadHoldoutSplitter:
    def test_n_holdout_two_picks_arctic_and_cape_town(
        self, geo_processed: PreprocessedDataset
    ) -> None:
        # arctic and cape_town are deliberately by far the most separated
        # pair (~13,000 km vs <6,000 for any other pair).
        train_idx, val_idx = SpatialSpreadHoldoutSplitter(
            n_holdout=2
        ).split(geo_processed)
        held = set(geo_processed.df.loc[val_idx, "location"].unique())
        assert held == {"arctic", "cape_town"}
        assert len(train_idx.intersection(val_idx)) == 0

    def test_n_holdout_three_adds_the_farthest_third_station(
        self, geo_processed: PreprocessedDataset
    ) -> None:
        # With arctic + cape_town fixed, the third station maximising
        # min-distance to that pair is accra (West Africa, far from both).
        _, val_idx = SpatialSpreadHoldoutSplitter(
            n_holdout=3
        ).split(geo_processed)
        held = set(geo_processed.df.loc[val_idx, "location"].unique())
        assert held == {"arctic", "cape_town", "accra"}

    def test_rejects_n_holdout_below_two(self) -> None:
        with pytest.raises(ValueError, match="StationLOSOSplitter for n=1"):
            SpatialSpreadHoldoutSplitter(n_holdout=1)

    def test_more_holdout_than_stations_raises(
        self, geo_processed: PreprocessedDataset
    ) -> None:
        with pytest.raises(ValueError, match="ingest more stations"):
            SpatialSpreadHoldoutSplitter(n_holdout=10).split(geo_processed)

    def test_missing_coordinate_column_raises_with_remediation(
        self, toy_processed: PreprocessedDataset
    ) -> None:
        # toy_processed has no lat/lon columns.
        with pytest.raises(ValueError, match="FeatureSpec.id_columns"):
            SpatialSpreadHoldoutSplitter(n_holdout=2).split(toy_processed)

    def test_name_records_n_holdout(self) -> None:
        assert (
            SpatialSpreadHoldoutSplitter(n_holdout=3).name
            == "spatial_spread[n=3]"
        )


# ---------------------------------------------------------------------------
# SpatialBlockSplitter
# ---------------------------------------------------------------------------


class TestSpatialBlockSplitter:
    def test_named_blocks_go_to_val(
        self, geo_processed: PreprocessedDataset
    ) -> None:
        splitter = SpatialBlockSplitter(
            val_blocks=("uganda", "kenya"), block_column="country"
        )
        train_idx, val_idx = splitter(geo_processed)
        val_countries = set(geo_processed.df.loc[val_idx, "country"].unique())
        train_countries = set(geo_processed.df.loc[train_idx, "country"].unique())
        assert val_countries == {"uganda", "kenya"}
        assert val_countries.isdisjoint(train_countries)

    def test_unknown_block_raises_with_available_list(
        self, geo_processed: PreprocessedDataset
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            SpatialBlockSplitter(
                val_blocks=("atlantis",), block_column="country"
            ).split(geo_processed)

    def test_missing_block_column_raises_with_remediation(
        self, geo_processed: PreprocessedDataset
    ) -> None:
        with pytest.raises(ValueError, match="FeatureSpec.id_columns"):
            SpatialBlockSplitter(
                val_blocks=("uganda",), block_column="region"
            ).split(geo_processed)

    def test_rejects_empty_val_blocks(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            SpatialBlockSplitter(val_blocks=(), block_column="country")

    def test_name_lists_val_blocks(self) -> None:
        assert (
            SpatialBlockSplitter(
                val_blocks=("uganda", "kenya"), block_column="country"
            ).name
            == "spatial_block[val=uganda,kenya]"
        )
