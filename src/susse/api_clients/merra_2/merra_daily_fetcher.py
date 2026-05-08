"""Daily-aggregated MERRA-2 fetcher tuned for the warehouse-ops pipeline.

MERRA-2's native temporal resolution is hourly (or 3-hourly for some
collections), but the warehouse stores daily values. This fetcher pulls
the native cadence over OPeNDAP for a single (location, date, variable),
applies a cosine-zenith-weighted mean to collapse to a single daily
value, and returns a long-format DataFrame matching the
:class:`susse.warehouse_ops.population.jobs.satellite_job.BaseSatelliteJob`
contract.

Why a fresh fetcher instead of extending :class:`MerraDataStreamFetcher`:

* The legacy fetcher hard-codes ``[0:1:23]`` for the time slice, which
  works for ``tavg1_*`` collections (24 hourly steps) but fails for
  ``inst3_*`` (8 three-hourly steps). We need to support both.
* The legacy fetcher returns a custom result object that needs further
  reshaping; the new one returns a plain long-format DataFrame.
* The legacy fetcher pulls credentials through ``MerraDownloadManager``
  via the OS keyring; we want a simpler ``.env`` flow consistent with
  CAMS_EMAIL.
"""

from __future__ import annotations

import logging
import os
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pvlib
from dotenv import load_dotenv
from pydap.cas.urs import setup_session
from pydap.client import open_url
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .merra_config import Merra2Config
from .merra_product import MerraProductData, MerraProducts

_logger = logging.getLogger(__name__)

# OPeNDAP requests are I/O-bound and serially issued at ~2.5 s each, so
# bulk ingest takes a week without parallelism. 8 workers gives ~8×
# wall-clock speedup with low (<5%) 503-loss observed empirically.
# Higher counts (16+) trigger noticeable server-side throttling; lower
# counts leave wall-time on the table.
#
# Transient failures are not catastrophic: each (date, geohash5) MERGE
# is idempotent, so re-running ingest jobs picks up days that 503'd
# in a previous run. Plan on multiple ingest runs over a few days,
# rather than expecting one run to hit every day.
_DEFAULT_MAX_WORKERS = 8

# How often to log a "X / N done" line during a long fan-out. Per-request
# logs are still emitted for every fetch, but the progress line aggregates
# them so a long ingest produces a readable summary even with workers
# interleaving.
_PROGRESS_LOG_EVERY = 100

# urllib3 retry policy for transient OPeNDAP failures. NASA's GES DISC
# cluster periodically returns 503 under load and the default
# 3-attempts-no-backoff policy fails on most bursts; an 8-attempt
# exponential-backoff policy (~64 s ceiling) recovers from the typical
# load spike. ``status_forcelist`` covers the gateway errors we see in
# practice; ``respect_retry_after_header=True`` honours any Retry-After
# the server sends back.
_REQUEST_RETRY = Retry(
    total=8,
    backoff_factor=0.5,
    status_forcelist=(502, 503, 504),
    allowed_methods=frozenset(("GET", "HEAD")),
    respect_retry_after_header=True,
)


class MerraAuthError(RuntimeError):
    """Raised when Earthdata credentials are missing or rejected."""


# Hard upper bound per OPeNDAP query. Single-day fetches normally complete
# in 1–3 seconds; anything past 5 minutes is almost certainly a hung
# connection rather than a slow response.
_REQUEST_TIMEOUT_SECONDS = 300

# MERRA-2's missing-data fill value is in the 1e15 range. Use a generous
# threshold so we don't mistake legitimate values (e.g. ozone Dobson units)
# as fills.
_FILL_VALUE_THRESHOLD = 1e14


# Hourly cadence per MERRA-2 collection prefix. The first matching prefix
# wins. Add a row here when introducing a new collection.
_COLLECTION_CADENCE: tuple[tuple[str, int], ...] = (
    ("tavg1_", 24),  # 1-hour time-averaged → 24 timesteps per day
    ("inst1_", 24),  # 1-hour instantaneous
    ("tavg3_", 8),  # 3-hour time-averaged → 8 timesteps per day
    ("inst3_", 8),  # 3-hour instantaneous
    ("statD_", 1),  # daily statistic
    ("const_", 1),  # constant
)


def _cadence_for(database_id: str) -> int:
    """Number of timesteps per day for the given MERRA-2 collection."""
    for prefix, n in _COLLECTION_CADENCE:
        if database_id.startswith(prefix):
            return n
    raise ValueError(
        f"Unknown MERRA-2 collection cadence for database_id={database_id!r}. "
        f"Add it to _COLLECTION_CADENCE in {__name__}."
    )


def _timestamps_for(date: _date, n_timesteps: int) -> pd.DatetimeIndex:
    """UTC timestamps for one day's MERRA-2 sub-daily data.

    ``tavg1_*`` collections are time-averaged over a centered hour, so the
    canonical timestamp is ``HH:30``. ``inst3_*`` are instantaneous
    snapshots every 3 hours starting at ``00:00``. Constants (1 timestep)
    use ``12:00``.
    """
    base = datetime.combine(date, datetime.min.time(), tzinfo=timezone.utc)
    if n_timesteps == 24:
        return pd.DatetimeIndex(
            [base + timedelta(hours=h, minutes=30) for h in range(24)]
        )
    if n_timesteps == 8:
        return pd.DatetimeIndex(
            [base + timedelta(hours=h * 3) for h in range(8)]
        )
    if n_timesteps == 1:
        return pd.DatetimeIndex([base + timedelta(hours=12)])
    raise ValueError(
        f"Unsupported timestep count: {n_timesteps}. "
        f"Add a branch to _timestamps_for() if a new cadence is needed."
    )


def cos_zenith_aggregate(
    sub_daily_values: np.ndarray,
    sub_daily_times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
) -> float:
    """Aggregate sub-daily values to one daily number, weighted by cos(zenith).

    Daytime hours dominate the result; nighttime hours contribute zero
    (cos(zenith) clipped at 0). NaN values in ``sub_daily_values`` are
    excluded from both the numerator and the weight sum.

    Returns ``nan`` if every weight is zero (polar night — won't happen
    in sub-Saharan Africa, but defensive) or every value is NaN.
    """
    if sub_daily_values.size == 0:
        return float("nan")
    if sub_daily_values.size != len(sub_daily_times):
        raise ValueError(
            f"Mismatched lengths: {sub_daily_values.size} values vs "
            f"{len(sub_daily_times)} timestamps."
        )
    sp = pvlib.solarposition.get_solarposition(
        time=sub_daily_times, latitude=latitude, longitude=longitude
    )
    weights = np.cos(np.radians(sp.zenith.values)).clip(min=0.0)
    valid = ~np.isnan(sub_daily_values)
    if not valid.any():
        return float("nan")
    weights = weights * valid
    if weights.sum() <= 0:
        return float("nan")
    values = np.where(valid, sub_daily_values, 0.0)
    return float((values * weights).sum() / weights.sum())


@dataclass(frozen=True)
class _Earthdata:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "_Earthdata":
        """Read credentials from environment, loading ``.env`` if present."""
        load_dotenv()
        username = os.getenv("EARTHDATA_USERNAME")
        password = os.getenv("EARTHDATA_PASSWORD")
        if not username or not password:
            raise MerraAuthError(
                "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set "
                "(in .env or shell environment) to fetch MERRA-2 data. "
                "Register at https://urs.earthdata.nasa.gov/ if needed, then "
                "approve the 'NASA GESDISC DATA ARCHIVE' application via "
                "your Earthdata profile."
            )
        return cls(username=username, password=password)


class MerraDailyFetcher:
    """Daily-aggregated MERRA-2 fetcher.

    Per-(variable, day) OPeNDAP fetches are issued concurrently via a
    thread pool. NASA's OPeNDAP at goldsmr4.gesdisc.eosdis.nasa.gov is
    I/O-bound (each request waits ~2.5 s on network round-trip), so
    threading delivers near-linear speedup until we hit the worker count.

    Authenticated session is created lazily on the first request and
    reused across all worker threads. ``requests.Session`` is documented
    as thread-safe for concurrent reads; we additionally guard the
    create-session-on-first-use path with a lock for safety. Set
    ``max_workers=1`` (or use the constructor default of 1 in unit tests)
    to disable parallelism entirely.
    """

    def __init__(
        self,
        credentials: _Earthdata | None = None,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}.")
        self._credentials = credentials  # None → load from env on first use
        self._session = None
        self._session_lock = threading.Lock()
        self._max_workers = max_workers

    def fetch_long_for_location(
        self,
        *,
        latitude: float,
        longitude: float,
        date_start: _date,
        date_end: _date,
        api_codes: tuple[str, ...],
    ) -> pd.DataFrame:
        """Fetch daily-aggregated values for one point and date range.

        Args:
            latitude: degrees, [-90, 90].
            longitude: degrees, [-180, 180].
            date_start: inclusive UTC date.
            date_end: inclusive UTC date.
            api_codes: tuple of MERRA-2 ``product_name`` values
                (e.g. ``("TOTEXTTAU", "TOTSCATAU")``). Looked up in
                :class:`MerraProducts`.

        Returns:
            Long-format DataFrame with columns ``(date, variable_id, value)``,
            where ``variable_id`` echoes the input ``api_code`` (the caller
            is responsible for any mapping to warehouse variable_ids).
            Row order is non-deterministic when ``max_workers > 1``; sort
            by ``(variable_id, date)`` if ordering matters downstream.
        """
        if date_start > date_end:
            raise ValueError(
                f"date_start ({date_start}) must be <= date_end ({date_end})."
            )
        merra_lat_idx, merra_lon_idx = self._grid_indices(latitude, longitude)

        # Build the unit-of-work list: one entry per (variable, day).
        tasks: list[tuple[str, MerraProductData, int, _date]] = []
        for api_code in api_codes:
            product_data = self._product_data_for(api_code)
            cadence = _cadence_for(product_data.database_id)
            day = date_start
            while day <= date_end:
                tasks.append((api_code, product_data, cadence, day))
                day += timedelta(days=1)

        if not tasks:
            return pd.DataFrame(columns=("date", "variable_id", "value"))

        # Pre-authenticate so the first request doesn't race the others
        # for session creation. After this returns, _session is set and
        # all subsequent _fetch_sub_daily calls reuse it.
        first_api_code, first_product, first_cadence, first_day = tasks[0]
        sample_url = self._build_url(
            product_data=first_product,
            date=first_day,
            merra_lat_idx=merra_lat_idx,
            merra_lon_idx=merra_lon_idx,
            cadence=first_cadence,
        )
        self._authenticated_session(sample_url)

        rows: list[dict] = []
        n_total = len(tasks)
        n_done = 0

        def _run(task: tuple[str, MerraProductData, int, _date]) -> dict | None:
            api_code, product_data, cadence, day = task
            return self._fetch_one_task(
                api_code=api_code,
                product_data=product_data,
                cadence=cadence,
                day=day,
                merra_lat_idx=merra_lat_idx,
                merra_lon_idx=merra_lon_idx,
                latitude=latitude,
                longitude=longitude,
            )

        if self._max_workers == 1:
            iterator = (_run(task) for task in tasks)
        else:
            pool = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="merra-fetch"
            )
            futures = [pool.submit(_run, task) for task in tasks]
            iterator = (fut.result() for fut in as_completed(futures))

        try:
            for row in iterator:
                n_done += 1
                if row is not None:
                    rows.append(row)
                if n_done % _PROGRESS_LOG_EVERY == 0:
                    _logger.info(
                        "MERRA-2 fetcher progress: %d / %d (%.1f%%)",
                        n_done, n_total, 100.0 * n_done / n_total,
                    )
        finally:
            if self._max_workers > 1:
                pool.shutdown(wait=True)

        return pd.DataFrame(rows, columns=("date", "variable_id", "value"))

    def _fetch_one_task(
        self,
        *,
        api_code: str,
        product_data: MerraProductData,
        cadence: int,
        day: _date,
        merra_lat_idx: int,
        merra_lon_idx: int,
        latitude: float,
        longitude: float,
    ) -> dict | None:
        """Single (variable, day) work unit: fetch, aggregate, return one row.

        Returns ``None`` when the fetch fails or the day's data is all
        missing. The caller filters nones out of the row list.
        """
        hourly = self._fetch_sub_daily(
            product_data=product_data,
            date=day,
            merra_lat_idx=merra_lat_idx,
            merra_lon_idx=merra_lon_idx,
            cadence=cadence,
        )
        if hourly is None:
            return None
        timestamps = _timestamps_for(day, cadence)
        daily = cos_zenith_aggregate(hourly, timestamps, latitude, longitude)
        if not pd.notna(daily):
            return None
        return {"date": day, "variable_id": api_code, "value": float(daily)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_sub_daily(
        self,
        *,
        product_data: MerraProductData,
        date: _date,
        merra_lat_idx: int,
        merra_lon_idx: int,
        cadence: int,
    ) -> np.ndarray | None:
        url = self._build_url(
            product_data=product_data,
            date=date,
            merra_lat_idx=merra_lat_idx,
            merra_lon_idx=merra_lon_idx,
            cadence=cadence,
        )
        session = self._authenticated_session(url)
        _logger.info(
            "MERRA-2 request: %s %s lat_idx=%d lon_idx=%d (cadence=%d/day)",
            product_data.product_name, date.isoformat(),
            merra_lat_idx, merra_lon_idx, cadence,
        )
        t0 = _time.monotonic()
        try:
            dataset = open_url(url, session=session, protocol="dap4")
            raw = np.array(dataset[product_data.product_name][:])
        except Exception as exc:
            _logger.warning(
                "MERRA-2 fetch failed for %s %s: %s",
                product_data.product_name, date.isoformat(), exc,
            )
            return None
        elapsed = _time.monotonic() - t0
        flat = raw.reshape(-1).astype(float)
        if flat.size != cadence:
            _logger.warning(
                "Expected %d sub-daily values for %s on %s, got %d. Skipping.",
                cadence, product_data.product_name, date.isoformat(), flat.size,
            )
            return None
        flat = np.where(flat > _FILL_VALUE_THRESHOLD, np.nan, flat)
        if np.isnan(flat).all():
            _logger.warning(
                "All sub-daily values are missing for %s on %s.",
                product_data.product_name, date.isoformat(),
            )
            return None
        _logger.info(
            "MERRA-2 response: %d values in %.1fs.", flat.size, elapsed,
        )
        return flat

    def _authenticated_session(self, url: str):
        # Double-checked locking around lazy session creation. Once set,
        # ``self._session`` is read without the lock from worker threads;
        # ``requests.Session`` handles concurrent reads safely.
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                creds = self._credentials or _Earthdata.from_env()
                try:
                    session = setup_session(
                        creds.username, creds.password, check_url=url
                    )
                except Exception as exc:
                    raise MerraAuthError(
                        "Earthdata authentication failed. Check that "
                        "EARTHDATA_USERNAME / EARTHDATA_PASSWORD are correct "
                        "and that the 'NASA GESDISC DATA ARCHIVE' application "
                        "is approved on your Earthdata profile."
                    ) from exc
                # Replace the default HTTPAdapter on this session with one
                # that retries 503/502/504 with exponential backoff. NASA's
                # OPeNDAP returns 503 under load and the default 3-attempt
                # policy is too eager to give up.
                adapter = HTTPAdapter(max_retries=_REQUEST_RETRY)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                self._session = session
        return self._session

    @staticmethod
    def _grid_indices(latitude: float, longitude: float) -> tuple[int, int]:
        lat_geos5 = Merra2Config._translate_lat_to_geos5_native(latitude)
        lon_geos5 = Merra2Config._translate_lon_to_geos5_native(longitude)
        merra_lat_idx = int(Merra2Config._find_closest_merra_coordinate(
            lat_geos5, Merra2Config.MERRA_LAT_COORDS
        ))
        merra_lon_idx = int(Merra2Config._find_closest_merra_coordinate(
            lon_geos5, Merra2Config.MERRA_LON_COORDS
        ))
        return merra_lat_idx, merra_lon_idx

    @staticmethod
    def _build_url(
        *,
        product_data: MerraProductData,
        date: _date,
        merra_lat_idx: int,
        merra_lon_idx: int,
        cadence: int,
    ) -> str:
        file_name = Merra2Config.create_file_name(date, product_data)
        m_str = str(date.month).zfill(2)
        y_str = str(date.year)
        time_slice = f"[0:1:{cadence - 1}]"
        suffix = (
            f"dap4.ce=/{product_data.product_name}{time_slice}"
            f"[{merra_lat_idx}:1:{merra_lat_idx}]"
            f"[{merra_lon_idx}:1:{merra_lon_idx}]"
        )
        return (
            f"{Merra2Config.generate_database_url(product_data)}"
            f"/{y_str}/{m_str}/{file_name}?{suffix}"
        )

    @staticmethod
    def _product_data_for(api_code: str) -> MerraProductData:
        for product in MerraProducts:
            if product.value.product_name == api_code:
                return product.value
        raise KeyError(
            f"No MerraProducts entry has product_name={api_code!r}. "
            f"Add it to merra_product.py before requesting it via the fetcher."
        )
