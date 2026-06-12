"""Elevation lookup — on-demand altitude for any (lat, lon).

Altitude is the only geographical feature the model consumes (deliberately;
raw lat/lon would let the model memorise station identities given the
small ~28-station training corpus). It is *not* stored in the BigQuery
warehouse: lookups are fast and static, so caching them in BQ adds no
value. Instead, an :class:`ElevationProvider` is captured by an
:class:`~susse.preprocessing.AltitudeFeature` instance and invoked at
preprocessing time.

The Protocol stays minimal so swapping backends (Open-Elevation API vs
local SRTM GeoTIFF vs an in-memory test stub) is a one-class change. The
v1 production backend is :class:`PvlibElevationProvider`, which delegates
to ``pvlib.location.lookup_altitude`` (Open-Elevation under the hood).
"""

from __future__ import annotations

from typing import Protocol


class ElevationProvider(Protocol):
    """Callable returning elevation in metres for a single (lat, lon).

    Implementations decide their own caching / batching strategy. The
    :class:`~susse.preprocessing.Preprocessor` dedupes ``(lat, lon)``
    pairs before invoking the provider, so a naive implementation that
    fetches one point per call is acceptable; a caching implementation
    just makes repeated calls cheap.
    """

    def __call__(self, lat: float, lon: float) -> float:
        """Return elevation in metres above mean sea level."""
        ...


# Cache keys round to this many decimal places (~10 m at the equator).
# Two coordinates within rounding distance return the same cached value;
# anything finer is below the resolution of any free elevation source.
_CACHE_DECIMALS: int = 4


class PvlibElevationProvider:
    """Production :class:`ElevationProvider` delegating to pvlib.

    Uses ``pvlib.location.lookup_altitude``, which calls Open-Elevation
    over HTTPS. Two consequences worth noting at the call site:

    * **Network dependency at inference time.** Portal deployments will
      hit ``api.open-elevation.com`` once per novel ``(lat, lon)``. A
      future iteration may swap the backend for a local SRTM GeoTIFF
      while keeping the :class:`ElevationProvider` Protocol unchanged.
    * **Failures raise.** No silent fallback to zero / NaN — a
      mis-configured network propagates as a clear exception rather
      than a feature column quietly filled with bad values.

    Cache is in-memory and per-instance: keep one provider alive for
    the duration of a training or inference run.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[float, float], float] = {}

    def __call__(self, lat: float, lon: float) -> float:
        key = (round(lat, _CACHE_DECIMALS), round(lon, _CACHE_DECIMALS))
        if key in self._cache:
            return self._cache[key]
        # Local import — only paid by users that actually invoke a lookup.
        from pvlib.location import lookup_altitude  # noqa: PLC0415

        elevation = float(lookup_altitude(latitude=lat, longitude=lon))
        self._cache[key] = elevation
        return elevation

    @property
    def cache_size(self) -> int:
        """Number of unique (lat, lon) pairs currently cached."""
        return len(self._cache)
