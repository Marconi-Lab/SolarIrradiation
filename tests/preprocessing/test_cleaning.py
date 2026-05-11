"""Tests for :class:`DataCleaner` subclasses + the kind-dispatch JSON loader.

Each cleaner is pinned against an analytical reference (constructed
input → expected surviving / imputed rows). The tests exercise the
public contract (``apply``, ``required_input_columns``, ``to_dict``
roundtrip via :func:`data_cleaner_from_dict`) rather than the private
helpers.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from susse.preprocessing import (
    CleanerKind,
    DataCleaner,
    GhiUpperBoundCleaner,
    HighMissingYearExcluder,
    IqrLowerBoundCleaner,
    KnnYearGapImputer,
    PerStationMeanImputer,
    data_cleaner_from_dict,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _daily_frame(
    location: str,
    start: date,
    n_days: int,
    *,
    value: float = 5.0,
) -> pd.DataFrame:
    """Construct ``n_days`` consecutive daily rows for one station."""
    return pd.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(n_days)],
            "location": [location] * n_days,
            "ghi_kwh_m2_day": [value] * n_days,
        }
    )


# ---------------------------------------------------------------------------
# GhiUpperBoundCleaner
# ---------------------------------------------------------------------------


class TestGhiUpperBoundCleaner:
    def test_drops_rows_strictly_above_threshold(self) -> None:
        # Boundary check: rows exactly at the threshold are kept; rows
        # strictly above are dropped. Pinning this prevents off-by-one
        # creep that could either over- or under-clip the dataset.
        df = _frame(
            [
                {"ghi_kwh_m2_day": 11.9},
                {"ghi_kwh_m2_day": 12.0},  # boundary — keep
                {"ghi_kwh_m2_day": 12.5},  # drop
                {"ghi_kwh_m2_day": 9.5},
            ]
        )
        cleaned = GhiUpperBoundCleaner().apply(df)
        assert cleaned["ghi_kwh_m2_day"].tolist() == [11.9, 12.0, 9.5]

    def test_custom_column_and_threshold(self) -> None:
        # Same cleaner reused with a different column name + bound — the
        # paper's defaults aren't the only valid configuration.
        df = _frame(
            [
                {"x": 0.5},
                {"x": 1.0},
                {"x": 2.0},
            ]
        )
        cleaned = GhiUpperBoundCleaner(column="x", threshold=1.0).apply(df)
        assert cleaned["x"].tolist() == [0.5, 1.0]

    def test_required_inputs(self) -> None:
        cleaner = GhiUpperBoundCleaner()
        assert cleaner.required_input_columns == ("ghi_kwh_m2_day",)

    def test_rejects_non_finite_threshold(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            GhiUpperBoundCleaner(threshold=float("nan"))

    def test_to_dict_roundtrip(self) -> None:
        c = GhiUpperBoundCleaner(column="x", threshold=8.5)
        rebuilt = data_cleaner_from_dict(c.to_dict())
        assert isinstance(rebuilt, GhiUpperBoundCleaner)
        assert rebuilt.column == "x"
        assert rebuilt.threshold == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# IqrLowerBoundCleaner
# ---------------------------------------------------------------------------


class TestIqrLowerBoundCleaner:
    def test_drops_low_outlier(self) -> None:
        # Q1=2, Q3=4, IQR=2 → lower bound = 2 − 1.5×2 = −1. With a clear
        # negative outlier far below that, expect it to be dropped.
        df = _frame([{"ghi_kwh_m2_day": v} for v in [-50.0, 2.0, 3.0, 4.0, 5.0]])
        cleaned = IqrLowerBoundCleaner().apply(df)
        assert -50.0 not in cleaned["ghi_kwh_m2_day"].tolist()
        assert len(cleaned) == 4

    def test_keeps_all_rows_when_no_low_outlier(self) -> None:
        df = _frame([{"ghi_kwh_m2_day": v} for v in [3.0, 4.0, 5.0, 6.0]])
        cleaned = IqrLowerBoundCleaner().apply(df)
        assert len(cleaned) == 4

    def test_does_not_clip_upper_outliers(self) -> None:
        # The upper-side fence is intentionally not applied — clear-sky
        # days at the top of the distribution are legitimate values.
        # Pin that an extreme high value survives this cleaner.
        df = _frame([{"ghi_kwh_m2_day": v} for v in [3.0, 4.0, 5.0, 6.0, 50.0]])
        cleaned = IqrLowerBoundCleaner().apply(df)
        assert 50.0 in cleaned["ghi_kwh_m2_day"].tolist()

    def test_empty_frame_returns_empty(self) -> None:
        empty = pd.DataFrame({"ghi_kwh_m2_day": []}, dtype=float)
        cleaned = IqrLowerBoundCleaner().apply(empty)
        assert cleaned.empty

    def test_rejects_non_positive_multiplier(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            IqrLowerBoundCleaner(multiplier=0.0)

    def test_to_dict_roundtrip(self) -> None:
        c = IqrLowerBoundCleaner(column="x", multiplier=2.0)
        rebuilt = data_cleaner_from_dict(c.to_dict())
        assert isinstance(rebuilt, IqrLowerBoundCleaner)
        assert rebuilt.column == "x"
        assert rebuilt.multiplier == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# HighMissingYearExcluder
# ---------------------------------------------------------------------------


class TestHighMissingYearExcluder:
    def test_drops_sparse_year_keeps_full_year(self) -> None:
        # Station A: full year of data → keep.
        # Station B: only 30 days within a 365-day span → drop entire group.
        full = _daily_frame("A", date(2024, 1, 1), 365)
        sparse_dates = [
            date(2024, 1, 1) + timedelta(days=i * 12) for i in range(30)
        ]  # 30 days spread over ~360
        sparse = pd.DataFrame(
            {
                "date": sparse_dates,
                "location": ["B"] * 30,
                "ghi_kwh_m2_day": [4.0] * 30,
            }
        )
        df = pd.concat([full, sparse], ignore_index=True)
        cleaner = HighMissingYearExcluder()  # 5% default threshold
        cleaned = cleaner.apply(df)
        assert set(cleaned["location"]) == {
            "A"
        }, "Station B's missing fraction is far above 5% — its rows must be dropped."
        assert len(cleaned) == 365

    def test_keeps_year_within_threshold(self) -> None:
        # 360 of 365 expected days = ~1.4% missing, below 5% → keep.
        days = [
            date(2024, 1, 1) + timedelta(days=i)
            for i in range(365)
            if i not in {10, 20, 30, 40, 50}
        ]
        df = pd.DataFrame(
            {
                "date": days,
                "location": ["A"] * len(days),
                "ghi_kwh_m2_day": [5.0] * len(days),
            }
        )
        cleaner = HighMissingYearExcluder()
        cleaned = cleaner.apply(df)
        assert len(cleaned) == len(days)

    def test_partial_year_with_full_density_kept(self) -> None:
        # Station active Jan-Mar only, every day present. Span = 90 days,
        # observed = 90 → missing fraction = 0 → keep. The missing-fraction
        # is computed against the station's own active span, not the full
        # calendar year (a station that operated Jan-Mar shouldn't be
        # penalised for "missing" Apr-Dec).
        df = _daily_frame("A", date(2024, 1, 1), 90)
        cleaner = HighMissingYearExcluder()
        cleaned = cleaner.apply(df)
        assert len(cleaned) == 90

    def test_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            HighMissingYearExcluder(missing_fraction_threshold=1.5)

    def test_to_dict_roundtrip(self) -> None:
        c = HighMissingYearExcluder(
            station_column="loc",
            date_column="d",
            missing_fraction_threshold=0.1,
        )
        rebuilt = data_cleaner_from_dict(c.to_dict())
        assert isinstance(rebuilt, HighMissingYearExcluder)
        assert rebuilt.station_column == "loc"
        assert rebuilt.date_column == "d"
        assert rebuilt.missing_fraction_threshold == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# KnnYearGapImputer
# ---------------------------------------------------------------------------


class TestKnnYearGapImputer:
    def test_fills_single_day_gap_with_neighbour_mean(self) -> None:
        # 4 consecutive days, plus a 5th day after a 1-day gap. The
        # missing day's k=2 nearest existing values are days 4 and 6
        # (1 step away each), so the imputed value is their mean.
        df = pd.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 6),
                    date(2024, 1, 7),
                ],
                "location": ["A"] * 6,
                "ghi_kwh_m2_day": [5.0, 5.5, 5.2, 6.0, 4.8, 4.5],
            }
        )
        cleaner = KnnYearGapImputer(k=2)
        imputed = cleaner.apply(df)
        # The Jan 5 row should have appeared.
        jan5_rows = imputed[imputed["date"] == date(2024, 1, 5)]
        assert len(jan5_rows) == 1
        # Mean of the two neighbours (Jan 4 = 6.0, Jan 6 = 4.8) = 5.4.
        assert jan5_rows.iloc[0]["ghi_kwh_m2_day"] == pytest.approx(5.4)
        assert jan5_rows.iloc[0]["location"] == "A"

    def test_propagates_context_columns_on_imputed_rows(self) -> None:
        # Carry-through columns (geohash5, lat, lon) on imputed rows
        # must come from the same (station, year) group, not appear as
        # NaN. A model that joins back on geohash5 would mis-route
        # imputed days otherwise.
        df = pd.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                ],
                "location": ["A"] * 4,
                "geohash5": ["xyz12"] * 4,
                "lat": [0.5] * 4,
                "lon": [32.0] * 4,
                "ghi_kwh_m2_day": [5.0, 5.5, 5.2, 5.4],
            }
        )
        cleaner = KnnYearGapImputer(k=2)
        imputed = cleaner.apply(df)
        jan3_rows = imputed[imputed["date"] == date(2024, 1, 3)]
        assert len(jan3_rows) == 1
        assert jan3_rows.iloc[0]["geohash5"] == "xyz12"
        assert jan3_rows.iloc[0]["lat"] == 0.5
        assert jan3_rows.iloc[0]["lon"] == 32.0

    def test_no_gaps_returns_input_unchanged(self) -> None:
        df = _daily_frame("A", date(2024, 1, 1), 30)
        cleaner = KnnYearGapImputer(k=3)
        imputed = cleaner.apply(df)
        assert len(imputed) == len(df)
        # Still the same target values (no spurious imputation).
        assert imputed["ghi_kwh_m2_day"].tolist() == df["ghi_kwh_m2_day"].tolist()

    def test_under_k_observations_skipped(self) -> None:
        # Group too small to compute a k-NN mean → cleaner skips it
        # rather than crashing. Pairs with HighMissingYearExcluder which
        # would normally have removed these groups upstream.
        df = pd.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 5)],
                "location": ["A"] * 2,
                "ghi_kwh_m2_day": [5.0, 5.5],
            }
        )
        cleaner = KnnYearGapImputer(k=5)
        imputed = cleaner.apply(df)
        assert len(imputed) == 2  # untouched

    def test_separate_groups_imputed_independently(self) -> None:
        # Station A has a gap on Jan 3; Station B has different values
        # and a gap on Jan 4. Each group's imputation must use only that
        # group's observed values (kNN never crosses station boundaries).
        df = pd.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 4),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 5),
                ],
                "location": ["A", "A", "A", "B", "B", "B", "B"],
                "ghi_kwh_m2_day": [3.0, 3.5, 4.0, 7.0, 7.2, 7.4, 7.6],
            }
        )
        cleaner = KnnYearGapImputer(k=2)
        imputed = cleaner.apply(df)
        a_jan3 = imputed[
            (imputed["location"] == "A") & (imputed["date"] == date(2024, 1, 3))
        ]
        b_jan4 = imputed[
            (imputed["location"] == "B") & (imputed["date"] == date(2024, 1, 4))
        ]
        assert len(a_jan3) == 1
        assert len(b_jan4) == 1
        # Imputed values come from each group's own neighbours, not the
        # other group's range.
        assert 3.0 < a_jan3.iloc[0]["ghi_kwh_m2_day"] < 4.5
        assert 7.0 < b_jan4.iloc[0]["ghi_kwh_m2_day"] < 8.0

    def test_rejects_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            KnnYearGapImputer(k=0)

    def test_to_dict_roundtrip(self) -> None:
        c = KnnYearGapImputer(target_column="x", k=7)
        rebuilt = data_cleaner_from_dict(c.to_dict())
        assert isinstance(rebuilt, KnnYearGapImputer)
        assert rebuilt.target_column == "x"
        assert rebuilt.k == 7


# ---------------------------------------------------------------------------
# Cross-cutting dispatch + protocol compliance
# ---------------------------------------------------------------------------


class TestDispatchByKind:
    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            data_cleaner_from_dict({"kind": "not_a_real_kind"})

    def test_missing_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            data_cleaner_from_dict({"threshold": 12.0})

    def test_each_kind_dispatches_to_correct_class(self) -> None:
        # Pin the round-trip: every CleanerKind member maps to the class
        # whose instances declare that kind, and back. Catches the
        # "added an enum value but forgot the cleaner_class branch" bug.
        instances: list[DataCleaner] = [
            GhiUpperBoundCleaner(),
            IqrLowerBoundCleaner(),
            HighMissingYearExcluder(),
            KnnYearGapImputer(),
            PerStationMeanImputer(columns=("temperature",)),
        ]
        for original in instances:
            assert original.kind.cleaner_class() is type(original)
            rebuilt = data_cleaner_from_dict(original.to_dict())
            assert type(rebuilt) is type(original)


class TestProtocolCompliance:
    """Every concrete DataCleaner must implement the abstract contract."""

    def test_all_subclasses_expose_kind_and_required_inputs(self) -> None:
        instances: list[DataCleaner] = [
            GhiUpperBoundCleaner(),
            IqrLowerBoundCleaner(),
            HighMissingYearExcluder(),
            KnnYearGapImputer(),
            PerStationMeanImputer(columns=("temperature",)),
        ]
        for c in instances:
            assert isinstance(c.kind, CleanerKind)
            assert len(c.required_input_columns) >= 1
            assert "kind" in c.to_dict()


class TestPerStationMeanImputer:
    """Imputer fills NaN with per-station column means."""

    def test_fills_nan_with_station_mean(self) -> None:
        df = pd.DataFrame(
            {
                "location": ["a", "a", "a", "b", "b", "b"],
                "temperature": [10.0, 20.0, np.nan, 5.0, np.nan, 15.0],
            }
        )
        result = PerStationMeanImputer(columns=("temperature",)).apply(df)
        # Station 'a' has values [10, 20] → mean 15 fills the NaN.
        # Station 'b' has values [5, 15] → mean 10 fills the NaN.
        assert result.loc[2, "temperature"] == pytest.approx(15.0)
        assert result.loc[4, "temperature"] == pytest.approx(10.0)
        # Non-NaN values unchanged.
        assert result.loc[0, "temperature"] == 10.0
        assert result.loc[3, "temperature"] == 5.0

    def test_leaves_all_nan_station_alone(self) -> None:
        # Station 'c' has every value NaN — no mean to compute, so the
        # imputer must leave those NaNs in place. Caller's
        # ``dropna_features`` will then drop the rows downstream.
        df = pd.DataFrame(
            {
                "location": ["a", "a", "c", "c"],
                "temperature": [10.0, 20.0, np.nan, np.nan],
            }
        )
        result = PerStationMeanImputer(columns=("temperature",)).apply(df)
        assert result.loc[0, "temperature"] == 10.0
        assert pd.isna(result.loc[2, "temperature"])
        assert pd.isna(result.loc[3, "temperature"])

    def test_multiple_columns_imputed_independently(self) -> None:
        df = pd.DataFrame(
            {
                "location": ["a", "a", "a"],
                "temperature": [10.0, 20.0, np.nan],
                "humidity": [np.nan, 40.0, 60.0],
            }
        )
        result = PerStationMeanImputer(
            columns=("temperature", "humidity"),
        ).apply(df)
        assert result.loc[2, "temperature"] == pytest.approx(15.0)
        assert result.loc[0, "humidity"] == pytest.approx(50.0)

    def test_empty_columns_rejected(self) -> None:
        with pytest.raises(ValueError, match="columns is empty"):
            PerStationMeanImputer(columns=())

    def test_missing_column_in_input_raises(self) -> None:
        df = pd.DataFrame({"location": ["a"], "temperature": [10.0]})
        with pytest.raises(KeyError, match="humidity"):
            PerStationMeanImputer(columns=("humidity",)).apply(df)

    def test_to_dict_roundtrip(self) -> None:
        original = PerStationMeanImputer(
            columns=("temperature", "humidity"),
            station_column="station_id",
        )
        rebuilt = data_cleaner_from_dict(original.to_dict())
        assert isinstance(rebuilt, PerStationMeanImputer)
        assert rebuilt.columns == ("temperature", "humidity")
        assert rebuilt.station_column == "station_id"

    def test_empty_frame_returns_empty(self) -> None:
        df = pd.DataFrame({"location": [], "temperature": []})
        result = PerStationMeanImputer(columns=("temperature",)).apply(df)
        assert result.empty
