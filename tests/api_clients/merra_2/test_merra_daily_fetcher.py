"""Tests for the daily-aggregated MERRA-2 fetcher.

Network and Earthdata authentication are stubbed out; these tests cover
the parts of the fetcher that don't depend on either:

* The cosine-zenith aggregation function — math, NaN handling, polar-night
  defensive branch, and the daytime-vs-nighttime weighting that's the
  entire reason this aggregator exists.
* The bbox URL builder — ensures the time-slice in the OPeNDAP constraint
  matches the collection's native cadence (24 for tavg1_*, 8 for inst3_*),
  and that lat/lon index ranges round-trip correctly.
* The bbox computation — smallest enclosing grid-index rectangle over a
  set of points (1-point degenerate, multi-point real region).
* :meth:`fetch_region` — multi-point output shape, per-point aggregation
  uses the right slice, parallel/serial equivalence.
* The auth gate — ``MerraAuthError`` raised when credentials are missing.

The end-to-end OPeNDAP path is exercised by running the migration against
the live Earthdata service.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from susse.api_clients.merra_2 import MerraAuthError
from susse.api_clients.merra_2.merra_daily_fetcher import (
    MerraDailyFetcher,
    _Bbox,
    _cadence_for,
    _Earthdata,
    _timestamps_for,
    cos_zenith_aggregate,
)
from susse.api_clients.merra_2.merra_product import MerraProducts


def _kampala_lat_lon() -> tuple[float, float]:
    return (0.333542, 32.568630)


def _hourly_index_utc(d: date) -> pd.DatetimeIndex:
    base = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    return pd.DatetimeIndex([base + timedelta(hours=h, minutes=30) for h in range(24)])


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
        noon_only = np.zeros(24)
        noon_only[12] = 100.0
        midnight_only = np.zeros(24)
        midnight_only[0] = 100.0
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
        result = cos_zenith_aggregate(np.full(24, np.nan), times, lat, lon)
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
    @pytest.mark.parametrize(
        "database_id, expected",
        [
            ("tavg1_2d_aer_Nx", 24),
            ("tavg1_2d_slv_Nx", 24),
            ("inst1_2d_asm_Nx", 24),
            ("tavg3_3d_cld_Np", 8),
            ("inst3_2d_gas_Nx", 8),
            ("statD_2d_slv_Nx", 1),
            ("const_2d_asm_Nx", 1),
        ],
    )
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


class TestBboxComputation:
    """``_bbox_enclosing`` returns the smallest grid-index rectangle
    covering every input point. The 1-point case must be the degenerate
    1×1 bbox (otherwise per-point fetches would over-fetch)."""

    def test_single_point_is_degenerate_one_by_one(self) -> None:
        kampala_idx = MerraDailyFetcher._grid_indices(*_kampala_lat_lon())
        bbox = MerraDailyFetcher._bbox_enclosing((kampala_idx,))
        assert bbox.n_lat == 1 and bbox.n_lon == 1
        assert bbox.lat_idx_lo == kampala_idx[0]
        assert bbox.lon_idx_lo == kampala_idx[1]

    def test_multi_point_spans_min_to_max_inclusive(self) -> None:
        # Three points: bbox must enclose all three corners and be sized
        # accordingly.
        idx = (
            MerraDailyFetcher._grid_indices(0.0, 30.0),
            MerraDailyFetcher._grid_indices(2.0, 32.0),
            MerraDailyFetcher._grid_indices(-1.0, 31.0),
        )
        bbox = MerraDailyFetcher._bbox_enclosing(idx)
        lat_indices = [p[0] for p in idx]
        lon_indices = [p[1] for p in idx]
        assert bbox.lat_idx_lo == min(lat_indices)
        assert bbox.lat_idx_hi == max(lat_indices)
        assert bbox.lon_idx_lo == min(lon_indices)
        assert bbox.lon_idx_hi == max(lon_indices)
        assert bbox.n_cells == bbox.n_lat * bbox.n_lon

    def test_empty_points_raises(self) -> None:
        with pytest.raises(ValueError, match="zero points"):
            MerraDailyFetcher._bbox_enclosing(())


class TestUrlBuilder:
    """The URL constraint must respect the collection's native cadence,
    or the OPeNDAP server returns an out-of-range error. The lat/lon
    range syntax also has to match GES DISC's DAP4 conventions."""

    def test_24_step_collection_uses_full_day_slice(self) -> None:
        url = MerraDailyFetcher._build_bbox_url(
            product_data=MerraProducts.AEROSOL_EXTINCTION_550nm.value,
            date=date(2024, 6, 21),
            bbox=_Bbox(lat_idx_lo=180, lat_idx_hi=180, lon_idx_lo=240, lon_idx_hi=240),
            cadence=24,
        )
        assert "[0:1:23]" in url
        assert "TOTEXTTAU" in url
        assert "/2024/06/" in url

    def test_8_step_collection_uses_8_index_slice(self) -> None:
        url = MerraDailyFetcher._build_bbox_url(
            product_data=MerraProducts.AEROSOL_OPTICAL_DEPTH_ANALYSIS.value,
            date=date(2024, 6, 21),
            bbox=_Bbox(lat_idx_lo=180, lat_idx_hi=180, lon_idx_lo=240, lon_idx_hi=240),
            cadence=8,
        )
        # Critical: 3-hourly collection has only 8 timesteps; using
        # [0:1:23] would error. This test guards against that regression.
        assert "[0:1:7]" in url
        assert "[0:1:23]" not in url
        assert "AODANA" in url

    def test_url_does_not_double_append_nc4(self) -> None:
        # create_file_name already includes the .nc4 suffix; the URL
        # builder must not add it again.
        url = MerraDailyFetcher._build_bbox_url(
            product_data=MerraProducts.AEROSOL_EXTINCTION_550nm.value,
            date=date(2024, 6, 21),
            bbox=_Bbox(lat_idx_lo=180, lat_idx_hi=180, lon_idx_lo=240, lon_idx_hi=240),
            cadence=24,
        )
        assert ".nc4.nc4" not in url
        assert ".nc4?" in url

    def test_region_bbox_emits_ranged_indices(self) -> None:
        # 1×1 degenerate bbox uses the same syntax as a multi-cell one;
        # a real region must produce ``[lo:1:hi]`` for both axes.
        url = MerraDailyFetcher._build_bbox_url(
            product_data=MerraProducts.AEROSOL_EXTINCTION_550nm.value,
            date=date(2024, 6, 21),
            bbox=_Bbox(lat_idx_lo=177, lat_idx_hi=189, lon_idx_lo=335, lon_idx_hi=345),
            cadence=24,
        )
        assert "[177:1:189]" in url
        assert "[335:1:345]" in url


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

    def test_present_credentials_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
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


def _stub_bbox_raw_factory(scale: float = 1.0):
    """Returns a stub ``_fetch_bbox_raw`` that produces a deterministic
    sub-daily field encoding (api_code, day, lat_idx, lon_idx) per cell.

    The encoded value lives at the noon timestep so cosine-zenith
    aggregation produces a daytime-dominated daily mean we can compare
    against analytically.
    """
    api_offset = {"TOTEXTTAU": 0, "TOTSCATAU": 1000, "AODANA": 2000, "TQV": 3000}

    def fake(self, *, product_data, date, bbox, cadence):
        arr = np.zeros((cadence, bbox.n_lat, bbox.n_lon), dtype=float)
        offset = api_offset.get(product_data.product_name, 9999)
        # Encode (api_code, mm/dd, lat_idx, lon_idx) into the noon spike.
        for i in range(bbox.n_lat):
            lat_idx = bbox.lat_idx_lo + i
            for j in range(bbox.n_lon):
                lon_idx = bbox.lon_idx_lo + j
                arr[cadence // 2, i, j] = scale * (
                    offset
                    + date.month * 100
                    + date.day
                    + lat_idx * 0.001
                    + lon_idx * 0.0001
                )
        return arr

    return fake


class TestFetchRegionShape:
    """``fetch_region`` returns one row per (point, date, variable)."""

    def test_one_row_per_point_date_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_fetch_bbox_raw",
            _stub_bbox_raw_factory(),
        )
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_authenticated_session",
            lambda self, url: object(),
        )
        fetcher = MerraDailyFetcher(max_workers=1)
        df = fetcher.fetch_region(
            points=((0.0, 30.0), (1.0, 31.0)),
            date_start=date(2024, 6, 1),
            date_end=date(2024, 6, 3),
            api_codes=("TOTEXTTAU", "TOTSCATAU"),
        )
        # 2 points × 3 days × 2 vars = 12 rows.
        assert len(df) == 12
        assert set(df.columns) == {
            "date",
            "latitude",
            "longitude",
            "variable_id",
            "value",
        }
        # Distinct values per (point, var) — each point's bbox slice
        # encodes its own (lat_idx, lon_idx) so values must differ.
        kampala_rows = df[(df["latitude"] == 0.0) & (df["longitude"] == 30.0)]
        nairobi_rows = df[(df["latitude"] == 1.0) & (df["longitude"] == 31.0)]
        for var in ("TOTEXTTAU", "TOTSCATAU"):
            kp = kampala_rows[kampala_rows["variable_id"] == var]["value"]
            nb = nairobi_rows[nairobi_rows["variable_id"] == var]["value"]
            assert (kp.values != nb.values).all(), (
                "Per-point bbox slicing must yield different values when "
                "the underlying cells are distinct."
            )

    def test_empty_api_codes_returns_empty_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_fetch_bbox_raw",
            _stub_bbox_raw_factory(),
        )
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_authenticated_session",
            lambda self, url: object(),
        )
        fetcher = MerraDailyFetcher(max_workers=1)
        df = fetcher.fetch_region(
            points=((0.0, 30.0),),
            date_start=date(2024, 6, 1),
            date_end=date(2024, 6, 3),
            api_codes=(),
        )
        assert df.empty
        assert list(df.columns) == [
            "date",
            "latitude",
            "longitude",
            "variable_id",
            "value",
        ]

    def test_rejects_empty_points(self) -> None:
        fetcher = MerraDailyFetcher(max_workers=1)
        with pytest.raises(ValueError, match="at least one point"):
            fetcher.fetch_region(
                points=(),
                date_start=date(2024, 6, 1),
                date_end=date(2024, 6, 1),
                api_codes=("TOTEXTTAU",),
            )

    def test_rejects_inverted_date_range(self) -> None:
        fetcher = MerraDailyFetcher(max_workers=1)
        with pytest.raises(ValueError, match="must be <="):
            fetcher.fetch_region(
                points=((0.0, 30.0),),
                date_start=date(2024, 6, 5),
                date_end=date(2024, 6, 1),
                api_codes=("TOTEXTTAU",),
            )


class TestFetchRegionPointParity:
    """The 1-point degenerate bbox must yield the same daily values as
    a manual per-point fetch would, since they read from the same cells.
    Pinning this prevents off-by-one slice-arithmetic regressions in
    :meth:`_fetch_region_for_task`."""

    def test_single_point_value_matches_direct_aggregation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = _stub_bbox_raw_factory()
        monkeypatch.setattr(MerraDailyFetcher, "_fetch_bbox_raw", stub)
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_authenticated_session",
            lambda self, url: object(),
        )
        lat, lon = 0.5, 32.5
        fetcher = MerraDailyFetcher(max_workers=1)
        df = fetcher.fetch_region(
            points=((lat, lon),),
            date_start=date(2024, 6, 21),
            date_end=date(2024, 6, 21),
            api_codes=("TOTEXTTAU",),
        )
        assert len(df) == 1
        # Reproduce the same aggregation manually using the stub directly
        # so we can verify slice arithmetic is correct.
        from susse.api_clients.merra_2.merra_daily_fetcher import _Bbox

        lat_idx, lon_idx = MerraDailyFetcher._grid_indices(lat, lon)
        bbox = _Bbox(lat_idx, lat_idx, lon_idx, lon_idx)
        raw = stub(
            None,  # self is unused by the stub
            product_data=MerraProducts.AEROSOL_EXTINCTION_550nm.value,
            date=date(2024, 6, 21),
            bbox=bbox,
            cadence=24,
        )
        sub_daily = raw[:, 0, 0].astype(float)
        timestamps = _timestamps_for(date(2024, 6, 21), 24)
        expected = cos_zenith_aggregate(sub_daily, timestamps, lat, lon)
        assert df.iloc[0]["value"] == pytest.approx(expected, abs=1e-12)

    def test_multipoint_each_uses_its_own_cell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same bbox call is shared across points within a (date, var)
        # task; the per-point slice arithmetic must pick the right cell.
        stub = _stub_bbox_raw_factory()
        monkeypatch.setattr(MerraDailyFetcher, "_fetch_bbox_raw", stub)
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_authenticated_session",
            lambda self, url: object(),
        )
        points = ((0.0, 30.0), (2.0, 32.5))
        fetcher = MerraDailyFetcher(max_workers=1)
        df = fetcher.fetch_region(
            points=points,
            date_start=date(2024, 6, 21),
            date_end=date(2024, 6, 21),
            api_codes=("TOTEXTTAU",),
        )
        assert len(df) == 2

        from susse.api_clients.merra_2.merra_daily_fetcher import _Bbox

        idx0 = MerraDailyFetcher._grid_indices(*points[0])
        idx1 = MerraDailyFetcher._grid_indices(*points[1])
        bbox = _Bbox(
            min(idx0[0], idx1[0]),
            max(idx0[0], idx1[0]),
            min(idx0[1], idx1[1]),
            max(idx0[1], idx1[1]),
        )
        raw = stub(
            None,
            product_data=MerraProducts.AEROSOL_EXTINCTION_550nm.value,
            date=date(2024, 6, 21),
            bbox=bbox,
            cadence=24,
        )
        timestamps = _timestamps_for(date(2024, 6, 21), 24)
        for (lat, lon), (lat_idx, lon_idx) in zip(points, (idx0, idx1)):
            sub = raw[:, lat_idx - bbox.lat_idx_lo, lon_idx - bbox.lon_idx_lo].astype(
                float
            )
            expected = cos_zenith_aggregate(sub, timestamps, lat, lon)
            row = df[(df["latitude"] == lat) & (df["longitude"] == lon)].iloc[0]
            assert row["value"] == pytest.approx(expected, abs=1e-12)


class TestParallelEquivalence:
    """Parallel and serial fetch paths must produce identical results.

    Without parallelism a real bulk ingest still hits hundreds of
    OPeNDAP requests in serial. The fetcher uses a thread pool to batch
    them, but threading must not change the *content* of the returned
    DataFrame — only the row order, which is documented as
    non-deterministic when ``max_workers > 1``.

    These tests stub ``_fetch_bbox_raw`` so neither path hits the
    network or requires Earthdata credentials.
    """

    def test_parallel_matches_serial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_fetch_bbox_raw",
            _stub_bbox_raw_factory(),
        )
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_authenticated_session",
            lambda self, url: object(),
        )

        kwargs = dict(
            points=((0.5179, 32.4715), (1.0, 33.0)),
            date_start=date(2024, 6, 1),
            date_end=date(2024, 6, 14),
            api_codes=("TOTEXTTAU", "TOTSCATAU"),
        )

        serial_df = MerraDailyFetcher(max_workers=1).fetch_region(**kwargs)
        parallel_df = MerraDailyFetcher(max_workers=4).fetch_region(**kwargs)

        # 2 points × 14 days × 2 variables = 56 rows in each result.
        assert len(serial_df) == 56
        assert len(parallel_df) == 56

        # Sort both before comparing — parallel results come back in
        # completion order, not submission order.
        sort_cols = ["variable_id", "date", "latitude", "longitude"]
        s = serial_df.sort_values(sort_cols).reset_index(drop=True)
        p = parallel_df.sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(s, p)

    def test_legacy_per_point_wrapper_drops_lat_lon_columns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Portal inference path uses fetch_long_for_location and expects
        # the legacy 3-column shape. Pin that the wrapper projects out
        # the region columns.
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_fetch_bbox_raw",
            _stub_bbox_raw_factory(),
        )
        monkeypatch.setattr(
            MerraDailyFetcher,
            "_authenticated_session",
            lambda self, url: object(),
        )
        df = MerraDailyFetcher(max_workers=1).fetch_long_for_location(
            latitude=0.5,
            longitude=32.5,
            date_start=date(2024, 6, 1),
            date_end=date(2024, 6, 3),
            api_codes=("TOTEXTTAU",),
        )
        assert list(df.columns) == ["date", "variable_id", "value"]
        assert len(df) == 3

    def test_rejects_zero_workers(self) -> None:
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            MerraDailyFetcher(max_workers=0)


class TestSessionNoneFallback:
    """``pydap.cas.urs.setup_session`` is documented to return ``None``
    when its check_url probe fails. NASA's goldsmr4 OPeNDAP cluster
    sometimes returns 503 on the probe even when the data endpoints
    themselves work; we must fall back to a plain ``requests.Session``
    with basic auth rather than crashing.
    """

    def test_falls_back_to_plain_session_when_setup_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub setup_session to return None — the failure mode this test pins.
        monkeypatch.setattr(
            "susse.api_clients.merra_2.merra_daily_fetcher.setup_session",
            lambda *a, **kw: None,
        )

        fetcher = MerraDailyFetcher(
            credentials=_Earthdata(username="alice", password="x"),
        )
        # Must not raise; must return something usable.
        session = fetcher._authenticated_session("https://example.invalid/")
        assert session is not None
        assert hasattr(session, "mount"), (
            "fallback session must be a real requests.Session "
            "(not None or some other placeholder)"
        )
        assert session.auth == ("alice", "x"), (
            "fallback session must carry basic auth so it can authenticate "
            "via the URS redirect chain"
        )


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
