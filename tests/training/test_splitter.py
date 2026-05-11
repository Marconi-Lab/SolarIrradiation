"""Splitter helpers: deterministic auto-pick + LOSO mask correctness."""

from __future__ import annotations

import pytest

from susse.preprocessing import PreprocessedDataset
from susse.training import auto_pick_largest_station, make_station_loso_splitter


class TestAutoPickLargestStation:
    def test_returns_station_with_most_rows(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        # Toy fixture has 60 rows for sta_a and 60 for sta_b — tied.
        # Deterministic tiebreak → alphabetic first.
        assert auto_pick_largest_station(toy_processed) == "sta_a"

    def test_missing_location_column_raises(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        with pytest.raises(ValueError, match="no column 'station'"):
            auto_pick_largest_station(toy_processed, location_column="station")


class TestStationLosoSplitter:
    def test_train_val_disjoint_and_cover_all_rows(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        splitter = make_station_loso_splitter("sta_b")
        train_idx, val_idx = splitter(toy_processed)
        assert len(train_idx.intersection(val_idx)) == 0
        assert len(train_idx) + len(val_idx) == len(toy_processed.df)
        # Val fold contains only the held-out station.
        held_out_locs = toy_processed.df.loc[val_idx, "location"].unique()
        assert list(held_out_locs) == ["sta_b"]

    def test_unknown_station_raises_with_suggestions(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        splitter = make_station_loso_splitter("kenya_location_not_real")
        with pytest.raises(ValueError, match="has no rows in this"):
            splitter(toy_processed)

    def test_splitter_name_records_held_out(self) -> None:
        splitter = make_station_loso_splitter("sta_b")
        assert splitter.__name__ == "station_loso[sta_b]"
