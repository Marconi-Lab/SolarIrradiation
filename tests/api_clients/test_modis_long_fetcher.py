"""Tests for ModisLongFetcher — chunking + cadence helpers (offline).

The wrapping behaviour around date-range splitting and per-product
cadence is what the warehouse-ingest layer depends on. The actual
ORNL DAAC HTTP round-trip is exercised by the live notebook smoke-test.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from susse.api_clients.modis import ModisLongFetcher, product_cadence_days
from susse.api_clients.modis.modis_long_fetcher import (
    _MAX_TILES_PER_REQUEST,
    _date_chunks,
    _max_days_per_request,
)


class TestProductCadenceDays:
    """Cadence is part of the public API: warehouse-ingest jobs use it
    to estimate expected row counts when checking coverage."""

    @pytest.mark.parametrize(
        ("product_id", "expected_cadence"),
        [
            ("MCD43A4", 1),
            ("MOD11A2", 8),
            ("MOD13Q1", 16),
        ],
    )
    def test_known_products(self, product_id: str, expected_cadence: int) -> None:
        assert product_cadence_days(product_id) == expected_cadence

    def test_unknown_product_falls_back_to_one(self, caplog) -> None:
        # Unknown products should not crash (callers may carry historical
        # ids), but they must log a warning so the typo is visible.
        import logging
        with caplog.at_level(logging.WARNING):
            assert product_cadence_days("NOT_A_PRODUCT") == 1
        assert any("PRODUCT_CADENCE_DAYS" in r.message for r in caplog.records)


class TestMaxDaysPerRequest:
    """The 10-tile-per-request cap translates to product-specific budgets."""

    def test_daily_product_caps_at_ten_days(self) -> None:
        # Daily cadence × 10 tiles = 10 calendar days max per request.
        assert _max_days_per_request("MCD43A4") == 10

    def test_eight_day_product(self) -> None:
        assert _max_days_per_request("MOD11A2") == 8 * _MAX_TILES_PER_REQUEST

    def test_sixteen_day_product(self) -> None:
        assert _max_days_per_request("MOD13Q1") == 16 * _MAX_TILES_PER_REQUEST


class TestDateChunks:
    """``_date_chunks`` slices a range into spans of <= max_days each."""

    def test_single_chunk_when_range_fits(self) -> None:
        chunks = _date_chunks(date(2024, 1, 1), date(2024, 1, 5), max_days=10)
        assert chunks == [(date(2024, 1, 1), date(2024, 1, 5))]

    def test_multiple_chunks_at_exact_boundary(self) -> None:
        # 20 days / 10 max → exactly 2 chunks of 10 days.
        chunks = _date_chunks(date(2024, 1, 1), date(2024, 1, 20), max_days=10)
        assert chunks == [
            (date(2024, 1, 1), date(2024, 1, 10)),
            (date(2024, 1, 11), date(2024, 1, 20)),
        ]

    def test_chunks_cover_every_day_exactly_once(self) -> None:
        # Property: union of chunk dates == [start, end], no overlap.
        start, end = date(2024, 1, 1), date(2024, 4, 15)
        chunks = _date_chunks(start, end, max_days=16)
        days_seen: set[date] = set()
        for chunk_start, chunk_end in chunks:
            assert chunk_start <= chunk_end
            for d in pd.date_range(chunk_start, chunk_end, freq="D"):
                assert d.date() not in days_seen, (
                    f"chunk overlap at {d.date()}: chunks={chunks}"
                )
                days_seen.add(d.date())
        all_days = {d.date() for d in pd.date_range(start, end, freq="D")}
        assert days_seen == all_days

    def test_empty_when_start_after_end(self) -> None:
        # Defensive: caller may pass an inverted range; should not crash.
        chunks = _date_chunks(date(2024, 1, 5), date(2024, 1, 1), max_days=10)
        assert chunks == []


class TestFetchLongForLocation:
    """The fetcher fans out one request per (product, chunk).

    We stub the per-request inner call and assert the outer behaviour:
    the right number of tasks are dispatched, results are concatenated,
    and the output frame has the expected columns.
    """

    def test_dispatches_one_task_per_product_chunk(self, monkeypatch) -> None:
        # 90-day window. MCD43A4 (cadence 1) → 10 days/chunk → 9 chunks.
        # MOD13Q1 (cadence 16) → 160 days/chunk → 1 chunk.
        # Total: 10 inner calls.
        fetcher = ModisLongFetcher(max_workers=1)

        captured: list[tuple] = []

        def _fake_fetch_one(self, product_id, band_id, chunk_start, chunk_end, location):
            captured.append((product_id, band_id, chunk_start, chunk_end))
            return [{
                "date": chunk_start, "product_id": product_id,
                "band_id": band_id, "value": 0.5,
            }]

        # Skip the network warm-up; it would hit ORNL DAAC.
        monkeypatch.setattr(ModisLongFetcher, "_warm_up", lambda self: None)
        monkeypatch.setattr(ModisLongFetcher, "_fetch_one", _fake_fetch_one)

        df = fetcher.fetch_long_for_location(
            latitude=0.333, longitude=32.568,
            date_start=date(2024, 6, 1), date_end=date(2024, 8, 29),  # 90 days
            products_and_bands=(
                ("MCD43A4", "Nadir_Reflectance_Band1"),
                ("MOD13Q1", "250m_16_days_NDVI"),
            ),
        )

        mcd_calls = [c for c in captured if c[0] == "MCD43A4"]
        ndvi_calls = [c for c in captured if c[0] == "MOD13Q1"]
        assert len(mcd_calls) == 9, f"expected 9 daily-cadence chunks, got {mcd_calls!r}"
        assert len(ndvi_calls) == 1, f"expected 1 16-day-cadence chunk, got {ndvi_calls!r}"
        assert list(df.columns) == ["date", "product_id", "band_id", "value"]
        assert len(df) == 10  # one row per chunk in the stub

    def test_empty_input_returns_empty_frame_with_schema(self) -> None:
        # No (product, band) pairs requested → empty frame, but with the
        # correct columns so downstream code (loaders, BigQuery upserts)
        # doesn't crash on a missing column.
        fetcher = ModisLongFetcher(max_workers=1)
        df = fetcher.fetch_long_for_location(
            latitude=0.0, longitude=0.0,
            date_start=date(2024, 1, 1), date_end=date(2024, 1, 31),
            products_and_bands=(),
        )
        assert df.empty
        assert list(df.columns) == ["date", "product_id", "band_id", "value"]

    def test_inverted_date_range_raises(self) -> None:
        fetcher = ModisLongFetcher(max_workers=1)
        with pytest.raises(ValueError, match="date_start"):
            fetcher.fetch_long_for_location(
                latitude=0.0, longitude=0.0,
                date_start=date(2024, 2, 1), date_end=date(2024, 1, 1),
                products_and_bands=(("MOD13Q1", "250m_16_days_NDVI"),),
            )

    def test_max_workers_zero_rejected(self) -> None:
        # 0-or-negative max_workers is a usage bug — must surface clearly
        # rather than producing a silently-broken thread pool.
        with pytest.raises(ValueError, match="max_workers"):
            ModisLongFetcher(max_workers=0)
