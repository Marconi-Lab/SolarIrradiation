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
import time as _time
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pvlib
from dotenv import load_dotenv
from pydap.cas.urs import setup_session
from pydap.client import open_url

from .merra_config import Merra2Config
from .merra_product import MerraProductData, MerraProducts

_logger = logging.getLogger(__name__)


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

    Authenticated session is created lazily on the first request (so an
    instance can be constructed in tests without hitting the network) and
    reused across all subsequent OPeNDAP queries.
    """

    def __init__(self, credentials: _Earthdata | None = None) -> None:
        self._credentials = credentials  # None → load from env on first use
        self._session = None

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
        """
        if date_start > date_end:
            raise ValueError(
                f"date_start ({date_start}) must be <= date_end ({date_end})."
            )
        merra_lat_idx, merra_lon_idx = self._grid_indices(latitude, longitude)
        rows: list[dict] = []

        for api_code in api_codes:
            product_data = self._product_data_for(api_code)
            cadence = _cadence_for(product_data.database_id)
            day = date_start
            while day <= date_end:
                hourly = self._fetch_sub_daily(
                    product_data=product_data,
                    date=day,
                    merra_lat_idx=merra_lat_idx,
                    merra_lon_idx=merra_lon_idx,
                    cadence=cadence,
                )
                if hourly is not None:
                    timestamps = _timestamps_for(day, cadence)
                    daily = cos_zenith_aggregate(
                        hourly, timestamps, latitude, longitude
                    )
                    if pd.notna(daily):
                        rows.append({
                            "date": day,
                            "variable_id": api_code,
                            "value": daily,
                        })
                day += timedelta(days=1)

        return pd.DataFrame(rows, columns=("date", "variable_id", "value"))

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
        if self._session is None:
            creds = self._credentials or _Earthdata.from_env()
            try:
                self._session = setup_session(
                    creds.username, creds.password, check_url=url
                )
            except Exception as exc:
                raise MerraAuthError(
                    "Earthdata authentication failed. Check that "
                    "EARTHDATA_USERNAME / EARTHDATA_PASSWORD are correct and "
                    "that the 'NASA GESDISC DATA ARCHIVE' application is "
                    "approved on your Earthdata profile."
                ) from exc
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
