"""Snap arbitrary query coordinates onto a satellite source's warehouse keys.

The warehouse stores each satellite source at a discrete set of cells, one
``geohash5`` per cell. To read a value for an arbitrary coordinate — a portal
click, a ground station — the query point must first be snapped onto a cell
the warehouse actually holds for that source.

:class:`NearestPixelSnapper` is the single place that snap happens: it is
built from the set of cells actually present in the warehouse for one source
and maps any coordinate to the ``geohash5`` of the nearest one. Because it is
built from the warehouse's own rows it carries no hard-coded grid constants —
it stays correct automatically as a source's ingested footprint changes.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Mean Earth radius (km), WGS-84 derived. Used for the great-circle snap.
_EARTH_RADIUS_KM: float = 6371.0088


class OutOfCoverageError(LookupError):
    """Raised when a query point has no warehouse cell within the snap guard.

    Signals that the requested coordinate falls outside the region ingested
    for the source — a genuine cache miss, not a snap that should silently
    reach across the gap to a distant, unrelated cell.
    """


class NearestPixelSnapper:
    """Snaps coordinates to the ``geohash5`` of the nearest stored cell.

    Constructed from the cells of a single satellite source as they are
    stored in the warehouse. The snap is a nearest-neighbour lookup by
    great-circle distance; a query point farther than ``max_snap_km`` from
    every known cell is reported as out of coverage rather than snapped to a
    distant, unrelated cell. A snapper built from an empty cell set (a source
    with no warehouse rows) reports every point as out of coverage.

    Args:
        geohash5s: ``geohash5`` of each stored cell centre.
        latitudes: Latitude of each stored cell centre, degrees.
        longitudes: Longitude of each stored cell centre, degrees.
        max_snap_km: Snap-distance guard. A query point whose nearest cell
            is farther than this is treated as out of coverage. Defaults to
            :attr:`_DEFAULT_MAX_SNAP_KM`.

    Raises:
        ValueError: If the three input sequences differ in length, or if a
            coordinate is outside valid lat/lon ranges.
    """

    _DEFAULT_MAX_SNAP_KM: ClassVar[float] = 120.0
    """Default snap-distance guard. NASA POWER's coarsest native grid is
    CERES' 1° (~111 km cell pitch); a point genuinely inside an ingested
    region sits at most ~80 km from a cell centre, so 120 km admits every
    in-region query while still rejecting points in un-ingested regions."""

    def __init__(
        self,
        *,
        geohash5s: Sequence[str],
        latitudes: Sequence[float],
        longitudes: Sequence[float],
        max_snap_km: float | None = None,
    ) -> None:
        if not (len(geohash5s) == len(latitudes) == len(longitudes)):
            raise ValueError(
                f"NearestPixelSnapper inputs differ in length: "
                f"{len(geohash5s)} geohash5s, {len(latitudes)} latitudes, "
                f"{len(longitudes)} longitudes. All three must align."
            )
        self._geohash5s: np.ndarray = np.asarray(geohash5s, dtype=object)
        self._latitudes: np.ndarray = np.asarray(latitudes, dtype=float)
        self._longitudes: np.ndarray = np.asarray(longitudes, dtype=float)
        if not np.all((self._latitudes >= -90.0) & (self._latitudes <= 90.0)):
            raise ValueError("Cell latitudes must all lie within [-90, 90].")
        if not np.all((self._longitudes >= -180.0) & (self._longitudes <= 180.0)):
            raise ValueError("Cell longitudes must all lie within [-180, 180].")
        self._max_snap_km: float = (
            self._DEFAULT_MAX_SNAP_KM if max_snap_km is None else float(max_snap_km)
        )
        if self._max_snap_km <= 0.0:
            raise ValueError(f"max_snap_km={self._max_snap_km} must be positive.")

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        max_snap_km: float | None = None,
        geohash5_col: str = "geohash5",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> "NearestPixelSnapper":
        """Build a snapper from a distinct-cells DataFrame.

        Args:
            df: One row per stored cell, carrying the geohash5 and centre
                coordinate columns named by ``geohash5_col`` / ``lat_col`` /
                ``lon_col``. The satellite-table column convention
                (``geohash5``, ``latitude``, ``longitude``) is the default.
                An empty frame yields a snapper with no coverage.
            max_snap_km: Forwarded to the constructor.

        Raises:
            ValueError: If a required column is absent from ``df``.
        """
        missing = [c for c in (geohash5_col, lat_col, lon_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"NearestPixelSnapper.from_dataframe: DataFrame is missing "
                f"column(s) {missing}. Present columns: {list(df.columns)}."
            )
        return cls(
            geohash5s=df[geohash5_col].tolist(),
            latitudes=df[lat_col].tolist(),
            longitudes=df[lon_col].tolist(),
            max_snap_km=max_snap_km,
        )

    @property
    def n_pixels(self) -> int:
        """Number of stored cells this snapper was built from."""
        return int(self._geohash5s.shape[0])

    @property
    def max_snap_km(self) -> float:
        """Snap-distance guard, in kilometres."""
        return self._max_snap_km

    def snap(self, lat: float, lon: float) -> str:
        """Snap one coordinate, raising if it is out of coverage.

        Raises:
            OutOfCoverageError: If ``(lat, lon)`` is outside coverage.
        """
        result = self.snap_or_none([lat], [lon])[0]
        if result is None:
            raise OutOfCoverageError(
                f"Coordinate ({lat:.4f}, {lon:.4f}) is outside the source's "
                f"ingested footprint — no warehouse cell is within range. "
                f"Ingest that region, or drop the point from the request."
            )
        return result

    def snap_or_none(
        self, latitudes: Sequence[float], longitudes: Sequence[float]
    ) -> list[str | None]:
        """Snap a batch of coordinates; yield ``None`` for any out of coverage.

        Args:
            latitudes: Query-point latitudes, degrees.
            longitudes: Query-point longitudes, degrees, aligned to
                ``latitudes``.

        Returns:
            One ``geohash5`` (or ``None`` when the point is outside the
            source's coverage) per query point, in input order.

        Raises:
            ValueError: If the two input sequences differ in length.
        """
        if len(latitudes) != len(longitudes):
            raise ValueError(
                f"snap_or_none got {len(latitudes)} latitudes but "
                f"{len(longitudes)} longitudes; they must align."
            )
        if len(latitudes) == 0:
            return []
        if self.n_pixels == 0:
            # No cells ingested for this source: every point is a miss.
            return [None] * len(latitudes)
        query_lat = np.asarray(latitudes, dtype=float)
        query_lon = np.asarray(longitudes, dtype=float)
        nearest_idx, nearest_km = self._nearest(query_lat, query_lon)
        return [
            str(self._geohash5s[idx]) if km <= self._max_snap_km else None
            for idx, km in zip(nearest_idx, nearest_km)
        ]

    def _nearest(
        self, query_lat: np.ndarray, query_lon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-cell index and great-circle distance (km) for each query point.

        Builds the full ``(n_query, n_pixel)`` distance matrix in one pass —
        per CLAUDE.md, no Python loop over the batch dimension.
        """
        # (n_query, 1) against (1, n_pixel) → (n_query, n_pixel).
        distances = _haversine_km(
            query_lat[:, None],
            query_lon[:, None],
            self._latitudes[None, :],
            self._longitudes[None, :],
        )
        nearest_idx = np.argmin(distances, axis=1)
        nearest_km = distances[np.arange(distances.shape[0]), nearest_idx]
        return nearest_idx, nearest_km

    def __repr__(self) -> str:
        return (
            f"NearestPixelSnapper(n_pixels={self.n_pixels}, "
            f"max_snap_km={self._max_snap_km:.0f})"
        )


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km between two sets of points (broadcast).

    All four inputs are broadcast against one another, so passing shapes
    ``(M, 1)`` and ``(1, N)`` yields an ``(M, N)`` distance matrix.
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    d_lat = lat2_rad - lat1_rad
    d_lon = np.radians(lon2 - lon1)
    a = (
        np.sin(d_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
