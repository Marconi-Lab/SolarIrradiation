"""Daily-aggregated MERRA-2 fetcher tuned for the warehouse-ops pipeline.

MERRA-2's native temporal resolution is hourly (or 3-hourly for some
collections), but the warehouse stores daily values. This fetcher pulls
the native cadence over OPeNDAP for a region of points + date range +
variable list, applies a cosine-zenith-weighted mean to collapse to a
single daily value per (point, date, variable), and returns a long-format
DataFrame matching the MERRA-region ingest job contract.

Region-shaped fetching
----------------------
GES DISC OPeNDAP serves region selects (lat-and-lon index ranges) at
roughly the same wall-clock cost as a single-cell select — server
overhead dominates the per-cell read cost for any reasonable region.
This fetcher therefore issues **one OPeNDAP call per (date, variable)**
covering all points in the request, and slices results client-side.
Compared to the pre-refactor "one call per (date, variable, point)"
shape this is ~100× faster for grid-sized requests.

The 1-point case is a degenerate region (1×1 bbox) and is served by the
same code path; :meth:`fetch_long_for_location` is a thin wrapper that
projects the region output back to ``(date, variable_id, value)``.

Why this fetcher rather than the legacy ``MerraDataStreamFetcher``:

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


@dataclass(frozen=True)
class _Bbox:
    """MERRA-2 grid-index bounding box (inclusive on all four sides)."""

    lat_idx_lo: int
    lat_idx_hi: int
    lon_idx_lo: int
    lon_idx_hi: int

    @property
    def n_lat(self) -> int:
        return self.lat_idx_hi - self.lat_idx_lo + 1

    @property
    def n_lon(self) -> int:
        return self.lon_idx_hi - self.lon_idx_lo + 1

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon


class MerraDailyFetcher:
    """Daily-aggregated MERRA-2 fetcher with region-shaped requests.

    The public entry point is :meth:`fetch_region`, which fans out one
    OPeNDAP call per (date, variable) covering all requested points. Each
    call's response is sliced client-side and aggregated per point with
    cosine-zenith weighting. Per-(date, variable) tasks run concurrently
    via a thread pool — NASA's OPeNDAP at goldsmr4 is I/O-bound (~2.5 s
    per round-trip), so threading delivers near-linear speedup until we
    hit the worker count.

    :meth:`fetch_long_for_location` is a 1-point convenience wrapper that
    drops the lat/lon columns from the region output. The portal inference
    path (single point, single date) uses this; bulk ingest uses
    :meth:`fetch_region`.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_region(
        self,
        *,
        points: tuple[tuple[float, float], ...],
        date_start: _date,
        date_end: _date,
        api_codes: tuple[str, ...],
    ) -> pd.DataFrame:
        """Fetch daily values for many points, dates, and variables.

        Issues one OPeNDAP bbox call per (date, variable), covering the
        smallest grid-index rectangle that encloses all ``points``. Each
        call's sub-daily data is sliced per point and reduced to a single
        daily number via cosine-zenith weighting.

        Args:
            points: tuple of ``(latitude, longitude)`` pairs in degrees.
                Latitudes in [-90, 90], longitudes in [-180, 180]. A
                single-point tuple is the degenerate 1×1 case and is
                served by the same code path.
            date_start: inclusive UTC date.
            date_end: inclusive UTC date.
            api_codes: tuple of MERRA-2 ``product_name`` values (e.g.
                ``("TOTEXTTAU", "TOTSCATAU")``). Looked up in
                :class:`MerraProducts`.

        Returns:
            Long-format DataFrame with columns
            ``(date, latitude, longitude, variable_id, value)``. One row
            per (point, date, variable) tuple where the underlying
            sub-daily data was non-missing. Row order is non-deterministic
            when ``max_workers > 1``; sort by
            ``(variable_id, date, latitude, longitude)`` if ordering matters.
        """
        if not points:
            raise ValueError("fetch_region requires at least one point.")
        if date_start > date_end:
            raise ValueError(
                f"date_start ({date_start}) must be <= date_end ({date_end})."
            )
        if not api_codes:
            return self._empty_region_frame()

        point_indices = self._grid_indices_for_points(points)
        bbox = self._bbox_enclosing(point_indices)

        tasks = self._build_tasks(date_start, date_end, api_codes)
        if not tasks:
            return self._empty_region_frame()

        # Pre-authenticate so the first request doesn't race the others
        # for session creation.
        first_api_code, first_product, first_cadence, first_day = tasks[0]
        sample_url = self._build_bbox_url(
            product_data=first_product,
            date=first_day,
            bbox=bbox,
            cadence=first_cadence,
        )
        self._authenticated_session(sample_url)

        rows: list[dict] = []
        n_total = len(tasks)
        n_done = 0

        def _run(task: tuple[str, MerraProductData, int, _date]) -> list[dict]:
            api_code, product_data, cadence, day = task
            return self._fetch_region_for_task(
                api_code=api_code,
                product_data=product_data,
                cadence=cadence,
                day=day,
                points=points,
                point_indices=point_indices,
                bbox=bbox,
            )

        if self._max_workers == 1:
            iterator = (_run(task) for task in tasks)
            pool = None
        else:
            pool = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="merra-fetch"
            )
            futures = [pool.submit(_run, task) for task in tasks]
            iterator = (fut.result() for fut in as_completed(futures))

        try:
            for task_rows in iterator:
                n_done += 1
                if task_rows:
                    rows.extend(task_rows)
                if n_done % _PROGRESS_LOG_EVERY == 0:
                    _logger.info(
                        "MERRA-2 fetcher progress: %d / %d (%.1f%%)",
                        n_done, n_total, 100.0 * n_done / n_total,
                    )
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        return pd.DataFrame(
            rows,
            columns=("date", "latitude", "longitude", "variable_id", "value"),
        )

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

        Thin 1-point wrapper around :meth:`fetch_region` that drops the
        lat/lon columns from the output, preserving the legacy contract
        used by the portal inference path.

        Returns:
            Long-format DataFrame with columns ``(date, variable_id, value)``.
            ``variable_id`` echoes the input ``api_code``; the caller is
            responsible for any mapping to warehouse variable_ids.
        """
        df = self.fetch_region(
            points=((latitude, longitude),),
            date_start=date_start,
            date_end=date_end,
            api_codes=api_codes,
        )
        if df.empty:
            return pd.DataFrame(columns=("date", "variable_id", "value"))
        return df[["date", "variable_id", "value"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Region task internals
    # ------------------------------------------------------------------

    def _fetch_region_for_task(
        self,
        *,
        api_code: str,
        product_data: MerraProductData,
        cadence: int,
        day: _date,
        points: tuple[tuple[float, float], ...],
        point_indices: tuple[tuple[int, int], ...],
        bbox: _Bbox,
    ) -> list[dict]:
        """One bbox fetch + per-point cosine-zenith aggregation.

        Returns a list of row dicts (possibly empty if the fetch failed
        or every requested point's sub-daily slice is all-NaN).
        """
        raw = self._fetch_bbox_raw(
            product_data=product_data,
            date=day,
            bbox=bbox,
            cadence=cadence,
        )
        if raw is None:
            return []
        timestamps = _timestamps_for(day, cadence)
        rows: list[dict] = []
        for (lat, lon), (lat_idx, lon_idx) in zip(points, point_indices):
            sub_daily = raw[
                :,
                lat_idx - bbox.lat_idx_lo,
                lon_idx - bbox.lon_idx_lo,
            ].astype(float)
            sub_daily = np.where(
                sub_daily > _FILL_VALUE_THRESHOLD, np.nan, sub_daily
            )
            if np.isnan(sub_daily).all():
                continue
            daily = cos_zenith_aggregate(sub_daily, timestamps, lat, lon)
            if not pd.notna(daily):
                continue
            rows.append({
                "date": day,
                "latitude": lat,
                "longitude": lon,
                "variable_id": api_code,
                "value": float(daily),
            })
        return rows

    def _fetch_bbox_raw(
        self,
        *,
        product_data: MerraProductData,
        date: _date,
        bbox: _Bbox,
        cadence: int,
    ) -> np.ndarray | None:
        """Single OPeNDAP bbox fetch.

        Returns a ``(cadence, n_lat, n_lon)`` float array, or ``None`` on
        transient failure or shape mismatch. Fill-value masking is applied
        by the caller (per-point, after slicing).
        """
        url = self._build_bbox_url(
            product_data=product_data,
            date=date,
            bbox=bbox,
            cadence=cadence,
        )
        session = self._authenticated_session(url)
        _logger.info(
            "MERRA-2 region request: %s %s lat_idx=[%d:%d] lon_idx=[%d:%d] "
            "(cadence=%d/day, %d cells)",
            product_data.product_name, date.isoformat(),
            bbox.lat_idx_lo, bbox.lat_idx_hi,
            bbox.lon_idx_lo, bbox.lon_idx_hi,
            cadence, bbox.n_cells,
        )
        t0 = _time.monotonic()
        try:
            dataset = open_url(url, session=session, protocol="dap4")
            raw = np.array(dataset[product_data.product_name][:])
        except Exception as exc:
            _logger.warning(
                "MERRA-2 region fetch failed for %s %s: %s",
                product_data.product_name, date.isoformat(), exc,
            )
            return None
        elapsed = _time.monotonic() - t0
        expected_shape = (cadence, bbox.n_lat, bbox.n_lon)
        if raw.shape != expected_shape:
            _logger.warning(
                "Expected shape %s for %s on %s, got %s. Skipping.",
                expected_shape, product_data.product_name,
                date.isoformat(), raw.shape,
            )
            return None
        _logger.info(
            "MERRA-2 region response: %s shape=%s in %.1fs.",
            product_data.product_name, raw.shape, elapsed,
        )
        return raw

    @staticmethod
    def _build_tasks(
        date_start: _date,
        date_end: _date,
        api_codes: tuple[str, ...],
    ) -> list[tuple[str, MerraProductData, int, _date]]:
        """Enumerate (api_code, product_data, cadence, day) work units."""
        tasks: list[tuple[str, MerraProductData, int, _date]] = []
        for api_code in api_codes:
            product_data = MerraDailyFetcher._product_data_for(api_code)
            cadence = _cadence_for(product_data.database_id)
            day = date_start
            while day <= date_end:
                tasks.append((api_code, product_data, cadence, day))
                day += timedelta(days=1)
        return tasks

    @staticmethod
    def _grid_indices_for_points(
        points: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[int, int], ...]:
        """Return per-point ``(lat_idx, lon_idx)`` MERRA-2 grid indices."""
        return tuple(
            MerraDailyFetcher._grid_indices(lat, lon) for lat, lon in points
        )

    @staticmethod
    def _bbox_enclosing(
        point_indices: tuple[tuple[int, int], ...],
    ) -> _Bbox:
        """Smallest grid-index bbox enclosing all input points."""
        if not point_indices:
            raise ValueError("Cannot compute bbox over zero points.")
        lat_indices = [p[0] for p in point_indices]
        lon_indices = [p[1] for p in point_indices]
        return _Bbox(
            lat_idx_lo=min(lat_indices),
            lat_idx_hi=max(lat_indices),
            lon_idx_lo=min(lon_indices),
            lon_idx_hi=max(lon_indices),
        )

    @staticmethod
    def _empty_region_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=("date", "latitude", "longitude", "variable_id", "value")
        )

    # ------------------------------------------------------------------
    # Auth + URL builder
    # ------------------------------------------------------------------

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
                # ``setup_session`` returns ``None`` (rather than raising)
                # when its check_url probe fails — typically because
                # goldsmr4 is briefly busy at startup. Fall back to a
                # plain ``requests.Session`` with basic auth; URS handles
                # the redirect chain via ``.netrc`` if the user has set
                # one up, and the actual per-day fetches will validate
                # auth on first use.
                if session is None:
                    import requests
                    _logger.warning(
                        "pydap.setup_session returned None (check_url probe "
                        "failed). Falling back to a plain requests.Session "
                        "with basic auth + .netrc redirect handling."
                    )
                    session = requests.Session()
                    session.auth = (creds.username, creds.password)
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
    def _build_bbox_url(
        *,
        product_data: MerraProductData,
        date: _date,
        bbox: _Bbox,
        cadence: int,
    ) -> str:
        """OPeNDAP DAP4 URL for a region select (1×1 degenerate is fine)."""
        file_name = Merra2Config.create_file_name(date, product_data)
        m_str = str(date.month).zfill(2)
        y_str = str(date.year)
        time_slice = f"[0:1:{cadence - 1}]"
        suffix = (
            f"dap4.ce=/{product_data.product_name}{time_slice}"
            f"[{bbox.lat_idx_lo}:1:{bbox.lat_idx_hi}]"
            f"[{bbox.lon_idx_lo}:1:{bbox.lon_idx_hi}]"
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
