"""Region-shaped NASA POWER fetcher.

Companion to :class:`NASAPowerFetchData`: instead of one HTTP call per
point (with up to 20 parameters per call), this fetcher issues one HTTP
call per (tile, parameter) covering the smallest set of bbox tiles that
encloses all requested points.

NASA POWER constraints on the daily ``/regional`` endpoint
----------------------------------------------------------

* Maximum bbox: **4.5° × 4.5°** per request.
* Maximum points returned: **100** per request.
* Exactly **one** parameter per request (in contrast to the point
  endpoint's 20-per-request batch).
* The returned grid is **NASA's native pixel grid for the parameter's
  source**: ``ALLSKY_SFC_SW_DWN`` etc. come from CERES SYN1DEG (1°
  cells), while T2M / WS2M etc. come from MERRA-2 (0.5° × 0.625°). For
  a given bbox + parameter, the response is the rectangle of native
  pixels whose centres fall inside the bbox.

For a 6° × 5.55° Uganda bbox × 29 catalog variables × 1 year, the
regional path issues ~4 tiles × 29 vars × 1 year-chunk ≈ 116 HTTP calls
in place of the ~40,000 calls the per-point path makes — a ~350× cost
reduction. The caller is responsible for mapping NASA's native pixels
back onto their own grid (e.g. via nearest-pixel snap onto a denser
geohash5 grid).
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from datetime import date as _date
from typing import ClassVar

import pandas as pd
import requests

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RegionalTile:
    """A bbox slice that fits inside the regional endpoint's size cap."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def span_lat(self) -> float:
        return self.max_lat - self.min_lat

    @property
    def span_lon(self) -> float:
        return self.max_lon - self.min_lon


class NASAPowerRegionalFetcher:
    """Region-shaped NASA POWER fetcher.

    The public entry point is :meth:`fetch_region_long`, which fans the
    request out into one HTTP call per (tile, api_code) and reduces the
    responses to a single long-format DataFrame.

    The fetcher does not snap points to NASA's native pixels — it returns
    the native pixels themselves. Consumers that want per-geohash5 rows
    must apply a nearest-pixel mapping over the result.

    Args:
        session: Optional pre-configured :class:`requests.Session`. The
            default is a fresh session per fetcher instance, which is fine
            for migrations but should be reused across calls if the
            fetcher is held as a long-lived service.
    """

    _MAX_TILE_SPAN_DEG: ClassVar[float] = 4.0
    """Tile-edge cap. NASA's hard limit is 4.5° per side and 100 points
    per request; using 4.0° gives a safety margin (≤ 9×9 = 81 native
    pixels at 0.5° resolution, ≤ 5×5 = 25 at 1°)."""

    _REQUEST_TIMEOUT_SECONDS: ClassVar[int] = 300
    """Hard upper bound per HTTP request. Regional requests for one year
    over a 4° tile normally return in 5–30 s; anything past 5 minutes is
    a hung connection rather than a slow response."""

    _MISSING_VALUE: ClassVar[float] = -999.0
    """NASA POWER's fill value for missing data. Rows with this value
    are dropped during parsing."""

    _COMMUNITY: ClassVar[str] = "RE"
    """The 'Renewable Energy' community; matches the point endpoint
    used by :class:`NASAPowerFetchData`."""

    _BASE_URL: ClassVar[str] = "https://power.larc.nasa.gov/api/temporal/daily/regional"

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def fetch_region_long(
        self,
        *,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        date_start: _date,
        date_end: _date,
        api_codes: tuple[str, ...],
    ) -> pd.DataFrame:
        """Fetch every (tile, api_code) covering the bbox and date range.

        Args:
            min_lat / max_lat / min_lon / max_lon: Inclusive bbox edges
                in degrees. The fetcher tiles internally to stay inside
                the 4.5°-per-side endpoint cap. Primitive args (rather
                than a ``BoundingBox``) keep this module independent of
                ``warehouse_ops`` and avoid circular imports.
            date_start / date_end: Inclusive date range. Passed through
                unchanged; no chunking is applied. NASA POWER tolerates
                multi-year regional requests for daily resolution.
            api_codes: NASA POWER parameter names (e.g. ``("T2M",
                "ALLSKY_SFC_SW_DWN")``). The endpoint accepts exactly one
                parameter per HTTP call, so the total request count is
                ``n_tiles * len(api_codes)``.

        Returns:
            Long-format DataFrame with columns
            ``(date, latitude, longitude, variable_id, value)``. The
            ``variable_id`` column carries the input ``api_code`` (the
            caller maps to warehouse variable_id). Latitudes / longitudes
            are NASA's native pixel centres; the same (lat, lon) appears
            across multiple variables when those variables happen to
            share a source grid. Empty if every cell of every tile is
            missing.
        """
        if min_lat > max_lat or min_lon > max_lon:
            raise ValueError(
                f"Invalid bbox: lat=[{min_lat}, {max_lat}] "
                f"lon=[{min_lon}, {max_lon}]. Need min <= max."
            )
        if date_start > date_end:
            raise ValueError(
                f"date_start ({date_start}) must be <= date_end ({date_end})."
            )
        if not api_codes:
            return self._empty_long_frame()
        tiles = self._tile_bbox(min_lat, max_lat, min_lon, max_lon)
        n_tiles = len(tiles)
        n_calls_total = n_tiles * len(api_codes)
        _logger.info(
            "NASA POWER regional fetch plan: bbox lat[%.3f..%.3f] "
            "lon[%.3f..%.3f] → %d tile(s) × %d variable(s) = %d call(s), "
            "dates %s..%s.",
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            n_tiles,
            len(api_codes),
            n_calls_total,
            date_start,
            date_end,
        )

        rows: list[dict] = []
        call_idx = 0
        for tile in tiles:
            for api_code in api_codes:
                call_idx += 1
                tile_rows = self._fetch_tile_variable(
                    tile=tile,
                    date_start=date_start,
                    date_end=date_end,
                    api_code=api_code,
                )
                _logger.info(
                    "[%d/%d] tile lat[%.2f..%.2f] lon[%.2f..%.2f] %s: %d row(s)",
                    call_idx,
                    n_calls_total,
                    tile.min_lat,
                    tile.max_lat,
                    tile.min_lon,
                    tile.max_lon,
                    api_code,
                    len(tile_rows),
                )
                rows.extend(tile_rows)

        if not rows:
            return self._empty_long_frame()
        df = pd.DataFrame(
            rows,
            columns=("date", "latitude", "longitude", "variable_id", "value"),
        )
        # De-duplicate native pixels that appear in adjacent overlapping
        # tiles. Tiling places a single edge between tiles, so duplicates
        # are typically zero, but a defensive drop keeps the contract
        # ('one row per (date, lat, lon, variable_id)') tight.
        df = df.drop_duplicates(
            subset=("date", "latitude", "longitude", "variable_id"),
            keep="first",
        ).reset_index(drop=True)
        return df

    @classmethod
    def _tile_bbox(
        cls,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> list[_RegionalTile]:
        """Split a bbox into tiles each ≤ ``_MAX_TILE_SPAN_DEG`` per side.

        Returns a list of non-overlapping tiles whose union covers the
        input bbox. The final tile in each dimension may be smaller than
        the others if the bbox span isn't a clean multiple of the cap.
        """
        lat_edges = cls._edges_for_span(min_lat, max_lat)
        lon_edges = cls._edges_for_span(min_lon, max_lon)
        tiles: list[_RegionalTile] = []
        for i in range(len(lat_edges) - 1):
            for j in range(len(lon_edges) - 1):
                tiles.append(
                    _RegionalTile(
                        min_lat=lat_edges[i],
                        max_lat=lat_edges[i + 1],
                        min_lon=lon_edges[j],
                        max_lon=lon_edges[j + 1],
                    )
                )
        return tiles

    @classmethod
    def _edges_for_span(cls, lo: float, hi: float) -> list[float]:
        """Edges that split ``[lo, hi]`` into pieces ≤ ``_MAX_TILE_SPAN_DEG``."""
        span = hi - lo
        if span <= cls._MAX_TILE_SPAN_DEG:
            return [lo, hi]
        # Use enough equal-width pieces to stay under the cap. Equal width
        # is preferable to "max-cap + remainder" because it spreads any
        # per-tile latency more evenly across requests.
        n_pieces = int(span // cls._MAX_TILE_SPAN_DEG) + 1
        step = span / n_pieces
        return [lo + k * step for k in range(n_pieces)] + [hi]

    def _fetch_tile_variable(
        self,
        *,
        tile: _RegionalTile,
        date_start: _date,
        date_end: _date,
        api_code: str,
    ) -> list[dict]:
        """One HTTP call → list of long-format row dicts."""
        url = self._build_regional_url(
            tile=tile,
            date_start=date_start,
            date_end=date_end,
            api_code=api_code,
        )
        t0 = _time.monotonic()
        response = self._session.get(url, timeout=self._REQUEST_TIMEOUT_SECONDS)
        elapsed = _time.monotonic() - t0
        response.raise_for_status()
        payload = response.json()
        _logger.info(
            "NASA POWER regional response: %d bytes in %.1fs (%s).",
            len(response.content),
            elapsed,
            api_code,
        )
        return self._parse_feature_collection(payload, api_code=api_code)

    @classmethod
    def _build_regional_url(
        cls,
        *,
        tile: _RegionalTile,
        date_start: _date,
        date_end: _date,
        api_code: str,
    ) -> str:
        start_str = date_start.strftime("%Y%m%d")
        end_str = date_end.strftime("%Y%m%d")
        return (
            f"{cls._BASE_URL}"
            f"?parameters={api_code}"
            f"&community={cls._COMMUNITY}"
            f"&latitude-min={tile.min_lat}"
            f"&latitude-max={tile.max_lat}"
            f"&longitude-min={tile.min_lon}"
            f"&longitude-max={tile.max_lon}"
            f"&start={start_str}"
            f"&end={end_str}"
            f"&format=JSON"
        )

    @classmethod
    def _parse_feature_collection(
        cls,
        payload: dict,
        *,
        api_code: str,
    ) -> list[dict]:
        """Flatten a NASA POWER FeatureCollection into long-format rows.

        Schema (as of 2026-05):

        ``payload["features"]`` is a list, one feature per native pixel.
        Each feature has ``geometry.coordinates = [lon, lat, elev]`` and
        ``properties.parameter[api_code][YYYYMMDD] = value``. Fill value
        (-999) is dropped.
        """
        features = payload.get("features") or ()
        rows: list[dict] = []
        for feature in features:
            coords = feature.get("geometry", {}).get("coordinates", ())
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            param_block = (
                feature.get("properties", {}).get("parameter", {}).get(api_code, {})
            )
            for date_str, raw_value in param_block.items():
                if raw_value is None:
                    continue
                value = float(raw_value)
                if value == cls._MISSING_VALUE:
                    continue
                rows.append(
                    {
                        "date": _date.fromisoformat(
                            f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        ),
                        "latitude": lat,
                        "longitude": lon,
                        "variable_id": api_code,
                        "value": value,
                    }
                )
        return rows

    @staticmethod
    def _empty_long_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=("date", "latitude", "longitude", "variable_id", "value"),
        )
