"""Tests for the curate_ground transformation against real data."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from susse.warehouse_ops.population.curation import CurationOptions, curate_ground


class TestCurationOptions:
    @pytest.mark.parametrize("precision", [0, -1, 13])
    def test_rejects_out_of_range_precision(self, precision: int) -> None:
        with pytest.raises(ValueError, match=r"outside \[1, 12\]"):
            CurationOptions(geohash_precision=precision)

    def test_rejects_empty_qc_level(self) -> None:
        with pytest.raises(ValueError, match="qc_level"):
            CurationOptions(qc_level="")

    def test_rejects_empty_version(self) -> None:
        with pytest.raises(ValueError, match="version"):
            CurationOptions(version="")


class TestCurateGround:
    def test_row_count_preserved(self, somalia_raw_df: pd.DataFrame) -> None:
        curated = curate_ground(somalia_raw_df)
        assert len(curated) == len(somalia_raw_df)

    def test_curated_schema_columns(self, somalia_raw_df: pd.DataFrame) -> None:
        curated = curate_ground(somalia_raw_df)
        expected = {
            "date",
            "month",
            "location",
            "lat",
            "lon",
            "geohash5",
            "ghi_wh_m2_day",
            "ghi_kwh_m2_day",
            "qc_level",
            "_version",
            "_curated_at",
        }
        assert set(curated.columns) == expected

    def test_geohash_matches_warehouse_for_somalia(
        self, somalia_raw_df: pd.DataFrame
    ) -> None:
        # The warehouse stores geohash5='sbx18' for somalia_location1
        # (verified directly via BQ query). Round-tripping through pygeohash
        # at precision=5 must produce the same value.
        curated = curate_ground(somalia_raw_df)
        assert (curated["geohash5"] == "sbx18").all()

    def test_unit_relationship(self, somalia_raw_df: pd.DataFrame) -> None:
        # By construction kWh/m²/day = Wh/m²/day / 1000. The validator
        # enforces this; this test asserts the relationship is exact (within
        # floating-point tolerance) on real values.
        curated = curate_ground(somalia_raw_df)
        diff = (curated["ghi_kwh_m2_day"] * 1000.0) - curated["ghi_wh_m2_day"]
        assert diff.abs().max() < 1e-6

    def test_month_is_first_of_month(self, somalia_raw_df: pd.DataFrame) -> None:
        curated = curate_ground(somalia_raw_df)
        for d, m in zip(curated["date"], curated["month"]):
            assert m == date(d.year, d.month, 1)

    def test_qc_level_and_version_constants_applied(
        self, somalia_raw_df: pd.DataFrame
    ) -> None:
        opts = CurationOptions(qc_level="manual", version="v2")
        curated = curate_ground(somalia_raw_df, opts)
        assert (curated["qc_level"] == "manual").all()
        assert (curated["_version"] == "v2").all()

    def test_curation_is_deterministic_modulo_timestamp(
        self, somalia_raw_df: pd.DataFrame
    ) -> None:
        a = curate_ground(somalia_raw_df).drop(columns=["_curated_at"])
        b = curate_ground(somalia_raw_df).drop(columns=["_curated_at"])
        pd.testing.assert_frame_equal(a, b)
