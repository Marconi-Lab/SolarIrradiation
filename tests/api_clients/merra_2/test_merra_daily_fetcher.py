"""Tests for the daily-aggregated MERRA-2 fetcher.

Network and Earthdata authentication are stubbed out; these tests cover
the parts of the fetcher that don't depend on either:

* The cosine-zenith aggregation function — math, NaN handling, polar-night
  defensive branch, and the daytime-vs-nighttime weighting that's the
  entire reason this aggregator exists.
* The URL builder — ensures the time-slice in the OPeNDAP constraint
  matches the collection's native cadence (24 for tavg1_*, 8 for inst3_*).
* The auth gate — ``MerraAuthError`` raised when credentials are missing.

The end-to-end OPeNDAP path is exercised by running B8 against the live
Earthdata service.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from susse.api_clients.merra_2 import MerraAuthError
from susse.api_clients.merra_2.merra_daily_fetcher import (
    MerraDailyFetcher,
    _Earthdata,
    _cadence_for,
    _timestamps_for,
    cos_zenith_aggregate,
)
from susse.api_clients.merra_2.merra_product import MerraProducts


def _kampala_lat_lon() -> tuple[float, float]:
    return (0.333542, 32.568630)


def _hourly_index_utc(d: date) -> pd.DatetimeIndex:
    base = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    return pd.DatetimeIndex(
        [base + timedelta(hours=h, minutes=30) for h in range(24)]
    )


class TestCosZenithAggregate:
    """The whole reason the aggregator exists is to weight daytime hours
    more than nighttime. These tests verify that math directly."""

    def test_constant_value_yields_same_value(self) -> None:
        # A constant input must produce that same value as the daily
        # aggregate, regardless of the weights — sanity check that the
        # weighted-mean denominator equals the numerator.
        lat, lon = _kampala_lat_lon()
        times = _hourly_index_utc(date(2024, 6, 21))
        result = cos_zenith_aggregate(np.full(24, 5.0), times, lat, lon)
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_noon_spike_dominates_midnight_spike(self) -> None:
        # Identical magnitude spikes — one at noon, one at midnight —
        # must produce very different daily aggregates because of the
        # cos(zenith) weighting.
        lat, lon = _kampala_lat_lon()
        times = _hourly_index_utc(date(2024, 6, 21))
        noon_only = np.zeros(24); noon_only[12] = 100.0
        midnight_only = np.zeros(24); midnight_only[0] = 100.0
        noon_daily = cos_zenith_aggregate(noon_only, times, lat, lon)
        midnight_daily = cos_zenith_aggregate(midnight_only, times, lat, lon)
        assert noon_daily > 5.0, (
            "A 100-unit spike at noon should produce a meaningful daily "
            f"aggregate; got {noon_daily}."
        )
        assert midnight_daily == pytest.approx(0.0, abs=1e-9), (
            "A spike at solar midnight should be zero-weighted; "
            f"got {midnight_daily}."
        )

    def test_handles_all_nan_input(self) -> None:
        lat, lon = _kampala_lat_lon()
        times = _hourly_index_utc(date(2024, 6, 21))
        result = cos_zenith_aggregate(
            np.full(24, np.nan), times, lat, lon
        )
        assert np.isnan(result)

    def test_partial_nan_excluded_from_average(self) -> None:
        # NaN values must be dropped from the weighted mean entirely
        # (numerator and weight sum). Otherwise a NaN would propagate.
        lat, lon = _kampala_lat_lon()
        times = _hourly_index_utc(date(2024, 6, 21))
        values = np.full(24, 5.0)
        values[0] = np.nan  # midnight, weight is zero anyway
        result = cos_zenith_aggregate(values, times, lat, lon)
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_rejects_length_mismatch(self) -> None:
        lat, lon = _kampala_lat_lon()
        times = _hourly_index_utc(date(2024, 6, 21))
        with pytest.raises(ValueError, match="Mismatched lengths"):
            cos_zenith_aggregate(np.zeros(8), times, lat, lon)


class TestCadenceLookup:
    @pytest.mark.parametrize("database_id, expected", [
        ("tavg1_2d_aer_Nx", 24),
        ("tavg1_2d_slv_Nx", 24),
        ("inst1_2d_asm_Nx", 24),
        ("tavg3_3d_cld_Np", 8),
        ("inst3_2d_gas_Nx", 8),
        ("statD_2d_slv_Nx", 1),
        ("const_2d_asm_Nx", 1),
    ])
    def test_known_collections(self, database_id: str, expected: int) -> None:
        assert _cadence_for(database_id) == expected

    def test_unknown_collection_raises_with_remediation(self) -> None:
        # If a future MERRA-2 collection is introduced, the error must
        # tell the developer where to add it. Otherwise they'd see an
        # opaque KeyError or wrong number of timesteps.
        with pytest.raises(ValueError, match="Add it to _COLLECTION_CADENCE"):
            _cadence_for("MADE_UP_PREFIX_2d_xyz_Nx")


class TestTimestampsFor:
    def test_24_step_uses_half_hour_offsets(self) -> None:
        # tavg1 timestamps are centered on the hour midpoint.
        idx = _timestamps_for(date(2024, 6, 21), 24)
        assert len(idx) == 24
        assert idx[0] == datetime(2024, 6, 21, 0, 30, tzinfo=timezone.utc)
        assert idx[12] == datetime(2024, 6, 21, 12, 30, tzinfo=timezone.utc)
        assert idx[-1] == datetime(2024, 6, 21, 23, 30, tzinfo=timezone.utc)

    def test_8_step_three_hourly_starting_at_midnight(self) -> None:
        idx = _timestamps_for(date(2024, 6, 21), 8)
        assert len(idx) == 8
        assert idx[0] == datetime(2024, 6, 21, 0, 0, tzinfo=timezone.utc)
        assert idx[-1] == datetime(2024, 6, 21, 21, 0, tzinfo=timezone.utc)


class TestUrlBuilder:
    """The URL constraint must respect the collection's native cadence,
    or the OPeNDAP server returns an out-of-range error."""

    def test_24_step_collection_uses_full_day_slice(self) -> None:
        url = MerraDailyFetcher._build_url(
            product_data=MerraProducts.AEROSOL_EXTINCTION_550nm.value,
            date=date(2024, 6, 21),
            merra_lat_idx=180,
            merra_lon_idx=240,
            cadence=24,
        )
        assert "[0:1:23]" in url
        assert "TOTEXTTAU" in url
        assert "/2024/06/" in url

    def test_8_step_collection_uses_8_index_slice(self) -> None:
        url = MerraDailyFetcher._build_url(
            product_data=MerraProducts.AEROSOL_OPTICAL_DEPTH_ANALYSIS.value,
            date=date(2024, 6, 21),
            merra_lat_idx=180,
            merra_lon_idx=240,
            cadence=8,
        )
        # Critical: 3-hourly collection has only 8 timesteps; using
        # [0:1:23] would error. This test guards against that regression.
        assert "[0:1:7]" in url
        assert "[0:1:23]" not in url
        assert "AODANA" in url

    def test_url_does_not_double_append_nc4(self) -> None:
        # create_file_name already includes the .nc4 suffix; the URL
        # builder must not add it again (the legacy non-stream builder
        # has this bug and we don't inherit it).
        url = MerraDailyFetcher._build_url(
            product_data=MerraProducts.AEROSOL_EXTINCTION_550nm.value,
            date=date(2024, 6, 21),
            merra_lat_idx=180,
            merra_lon_idx=240,
            cadence=24,
        )
        assert ".nc4.nc4" not in url
        assert ".nc4?" in url


class TestEarthdataCredentials:
    def test_missing_credentials_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
        monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
        # _Earthdata.from_env loads .env at import time, but we want a
        # truly empty environment here. Patch load_dotenv to no-op so it
        # can't repopulate the variables in CI environments that have
        # them.
        monkeypatch.setattr(
            "susse.api_clients.merra_2.merra_daily_fetcher.load_dotenv",
            lambda *a, **kw: False,
        )
        with pytest.raises(MerraAuthError, match="EARTHDATA_USERNAME"):
            _Earthdata.from_env()

    def test_present_credentials_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EARTHDATA_USERNAME", "alice")
        monkeypatch.setenv("EARTHDATA_PASSWORD", "secret")
        # No-op the .env loader so test env wins deterministically.
        monkeypatch.setattr(
            "susse.api_clients.merra_2.merra_daily_fetcher.load_dotenv",
            lambda *a, **kw: False,
        )
        creds = _Earthdata.from_env()
        assert creds.username == "alice"
        assert creds.password == "secret"


class TestParallelEquivalence:
    """Parallel and serial fetch paths must produce identical results.

    Without parallelism a real bulk ingest takes ~9 days of wall time
    (300K OPeNDAP requests at ~2.5 s each). The fetcher uses a thread
    pool to batch them, but threading must not change the *content* of
    the returned DataFrame — only the row order, which is documented as
    non-deterministic when ``max_workers > 1``.

    These tests stub ``_fetch_sub_daily`` so neither path hits the
    network or requires Earthdata credentials.
    """

    def _stub_fetch(self) -> callable:
        """Returns a deterministic synthetic ``_fetch_sub_daily`` whose
        output uniquely encodes (api_code, date) so we can verify each
        task got the right inputs after re-shuffling.
        """
        # Map api_code → small integer offset so each variable is
        # distinguishable in the output.
        api_offset = {"TOTEXTTAU": 0, "TOTSCATAU": 1000, "AODANA": 2000, "TQV": 3000}

        def fake_fetch(self, *, product_data, date, merra_lat_idx,
                       merra_lon_idx, cadence):
            # All-zeros except a spike at noon whose value encodes
            # (api_code, date). cos_zenith_aggregate weights noon highest
            # so the daily value will be dominated by that spike.
            arr = np.zeros(cadence)
            offset = api_offset.get(product_data.product_name, 9999)
            # `date` here is a `date` instance (not datetime) — month*100+day
            # gives a unique integer for each day in a single-month test.
            arr[cadence // 2] = float(offset + date.month * 100 + date.day)
            return arr

        return fake_fetch

    def test_parallel_matches_serial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from susse.api_clients.merra_2.merra_daily_fetcher import MerraDailyFetcher

        monkeypatch.setattr(MerraDailyFetcher, "_fetch_sub_daily", self._stub_fetch())
        # Skip auth — never reached because _fetch_sub_daily is stubbed,
        # but the pre-auth call in fetch_long_for_location still tries
        # to set up a session.
        monkeypatch.setattr(
            MerraDailyFetcher, "_authenticated_session",
            lambda self, url: object(),
        )

        kwargs = dict(
            latitude=0.5179, longitude=32.4715,
            date_start=date(2024, 6, 1), date_end=date(2024, 6, 14),
            api_codes=("TOTEXTTAU", "TOTSCATAU"),
        )

        serial_df = MerraDailyFetcher(max_workers=1).fetch_long_for_location(**kwargs)
        parallel_df = MerraDailyFetcher(max_workers=4).fetch_long_for_location(**kwargs)

        # Two variables × 14 days = 28 rows in each result.
        assert len(serial_df) == 28
        assert len(parallel_df) == 28

        # Sort both before comparing — parallel results come back in
        # completion order, not submission order.
        sort_cols = ["variable_id", "date"]
        s = serial_df.sort_values(sort_cols).reset_index(drop=True)
        p = parallel_df.sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(s, p)

    def test_rejects_zero_workers(self) -> None:
        from susse.api_clients.merra_2.merra_daily_fetcher import MerraDailyFetcher

        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            MerraDailyFetcher(max_workers=0)


class TestSessionRetryConfig:
    """Sessions returned by ``_authenticated_session`` must mount an
    HTTPAdapter with a Retry policy that handles 503/502/504 patiently
    enough to ride out NASA's OPeNDAP load spikes.

    Without this config, ~25-30% of bulk-ingest fetches fail because
    urllib3's default ``Retry`` only attempts 3 times with no backoff,
    which loses to busy windows that last >1 second.
    """

    def test_retry_policy_mounted_on_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests
        from susse.api_clients.merra_2.merra_daily_fetcher import (
            MerraDailyFetcher, _Earthdata,
        )

        # Stub the URS handshake — we just want a Session object back.
        def fake_setup_session(username, password, check_url):
            return requests.Session()
        monkeypatch.setattr(
            "susse.api_clients.merra_2.merra_daily_fetcher.setup_session",
            fake_setup_session,
        )

        fetcher = MerraDailyFetcher(
            credentials=_Earthdata(username="alice", password="x"),
        )
        session = fetcher._authenticated_session("https://example.invalid/")

        # Both schemes must have the patient retry adapter mounted.
        for scheme in ("http://", "https://"):
            adapter = session.get_adapter(scheme)
            retry = adapter.max_retries
            assert retry.total >= 5, (
                f"Retry.total={retry.total} is too few for OPeNDAP load spikes; "
                "we want 8."
            )
            assert 503 in retry.status_forcelist, (
                "503 must trigger a retry — that's the most common transient "
                "OPeNDAP failure."
            )
            assert retry.backoff_factor > 0, (
                "Retries with no backoff slam the busy server immediately and "
                "exhaust before the load window clears."
            )


class TestProductLookup:
    def test_known_api_code_resolves(self) -> None:
        product_data = MerraDailyFetcher._product_data_for("TOTEXTTAU")
        assert product_data.product_name == "TOTEXTTAU"
        assert product_data.database_id == "tavg1_2d_aer_Nx"

    def test_unknown_api_code_raises_with_remediation(self) -> None:
        with pytest.raises(KeyError, match="Add it to merra_product.py"):
            MerraDailyFetcher._product_data_for("NOT_A_REAL_CODE")
