"""Typed value objects describing what to ingest.

The population subsystem supports three ingest patterns:

1. **Named-location** — fetch a satellite source for a discrete list of
   ``LocationSpec`` points (e.g. one per ground station). Drives training
   data quality.
2. **Grid** — fetch a satellite source for every point in a ``GridSpec``.
   Drives portal inference-cache coverage across sub-Saharan Africa.
3. **Ground-file** — parse a heterogeneous ground-measurement file via a
   :class:`susse.warehouse_ops.population.adapters.GroundSourceAdapter`,
   then curate into the standard schema.

A :class:`FetchPlan` subclass describes one execution. Jobs accept a plan
and dispatch on its concrete type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Iterator


class Source(StrEnum):
    """Origin of a satellite or reanalysis dataset.

    Enum values match the ``source`` column convention already established
    in the warehouse (``'NASA'``, ``'CAMS'``, ``'MERRA2'``, ``'MODIS'``).
    The enum *member name* is more specific (``NASA_POWER`` distinguishes
    from MERRA-2, which is also a NASA dataset) while the *value* is the
    string actually written to BigQuery.
    """

    NASA_POWER = "NASA"
    CAMS = "CAMS"
    MERRA_2 = "MERRA2"
    MODIS = "MODIS"


@dataclass(frozen=True)
class DateRange:
    """Inclusive [start, end] date range.

    Both endpoints are inclusive — they represent calendar dates, not
    half-open intervals. Use :meth:`days` to enumerate.
    """

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"DateRange.start ({self.start}) is after end ({self.end}). "
                f"Swap the arguments."
            )

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class LocationSpec:
    """A named geographic point (ground station, test site, grid cell)."""

    name: str
    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LocationSpec.name must be non-empty.")
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(
                f"LocationSpec.lat={self.lat} is outside [-90, 90]. "
                f"Did you swap lat and lon?"
            )
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(
                f"LocationSpec.lon={self.lon} is outside [-180, 180]."
            )


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned lat/lon bounding box, inclusive on all four sides."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.min_lat <= self.max_lat <= 90.0):
            raise ValueError(
                f"BoundingBox latitude range ({self.min_lat}, {self.max_lat}) "
                f"is invalid. Need -90 <= min_lat <= max_lat <= 90."
            )
        if not (-180.0 <= self.min_lon <= self.max_lon <= 180.0):
            raise ValueError(
                f"BoundingBox longitude range ({self.min_lon}, {self.max_lon}) "
                f"is invalid. Need -180 <= min_lon <= max_lon <= 180."
            )


@dataclass(frozen=True)
class GridSpec:
    """A bounding-box grid sampled at fixed lat/lon resolution.

    Iterating yields :class:`LocationSpec` points named ``grid_{lat:.4f}_{lon:.4f}``.
    The resolution is chosen to match the satellite source's native grid
    (e.g. 0.05° for CAMS, 0.5° for NASA POWER) so the cached estimates
    correspond to the source's own sample points.
    """

    bbox: BoundingBox
    lat_step: float
    lon_step: float

    def __post_init__(self) -> None:
        if self.lat_step <= 0:
            raise ValueError(
                f"GridSpec.lat_step={self.lat_step} must be positive."
            )
        if self.lon_step <= 0:
            raise ValueError(
                f"GridSpec.lon_step={self.lon_step} must be positive."
            )

    @property
    def n_points(self) -> int:
        n_lat = int(round((self.bbox.max_lat - self.bbox.min_lat) / self.lat_step)) + 1
        n_lon = int(round((self.bbox.max_lon - self.bbox.min_lon) / self.lon_step)) + 1
        return n_lat * n_lon

    def iter_locations(self) -> Iterator[LocationSpec]:
        """Yield one :class:`LocationSpec` per grid point.

        Ordering is row-major: latitude outer loop, longitude inner loop.
        Points are produced at ``min_lat + k * lat_step`` for integer k
        such that the value remains ``<= max_lat`` (and analogously for lon).
        """
        n_lat = int(round((self.bbox.max_lat - self.bbox.min_lat) / self.lat_step))
        n_lon = int(round((self.bbox.max_lon - self.bbox.min_lon) / self.lon_step))
        for i in range(n_lat + 1):
            lat = round(self.bbox.min_lat + i * self.lat_step, 6)
            for j in range(n_lon + 1):
                lon = round(self.bbox.min_lon + j * self.lon_step, 6)
                yield LocationSpec(
                    name=f"grid_{lat:.4f}_{lon:.4f}",
                    lat=lat,
                    lon=lon,
                )


@dataclass(frozen=True)
class VariableSpec:
    """Catalogue entry for a satellite/ground variable.

    Mirrors the columns of the ``dim_variable`` table so an instance can be
    written directly. Use :class:`susse.warehouse_ops.population.dim_variable.VariableCatalog`
    for the registered set of variables.
    """

    variable_id: str
    source: Source
    api_code: str
    display_name: str
    unit: str
    native_unit: str
    description: str
    temporal_granularity: str = "daily"
    spatial_resolution_km: float | None = None
    valid_min: float | None = None
    valid_max: float | None = None


# ---------------------------------------------------------------------------
# Ingest plans (tagged hierarchy)
# ---------------------------------------------------------------------------


class FetchPlan(ABC):
    """Tagged base class for the three ingest patterns."""

    @abstractmethod
    def describe(self) -> str:
        """Return a one-line human-readable summary for logging."""


@dataclass(frozen=True)
class NamedLocationsPlan(FetchPlan):
    """Fetch a satellite source for a discrete set of locations."""

    source: Source
    date_range: DateRange
    locations: tuple[LocationSpec, ...]
    variables: tuple[VariableSpec, ...]

    def __post_init__(self) -> None:
        if not self.locations:
            raise ValueError("NamedLocationsPlan requires at least one location.")
        if not self.variables:
            raise ValueError("NamedLocationsPlan requires at least one variable.")
        for v in self.variables:
            if v.source is not self.source:
                raise ValueError(
                    f"Variable '{v.variable_id}' is from source {v.source} but "
                    f"plan source is {self.source}. Variables must match the "
                    f"plan's source."
                )

    def describe(self) -> str:
        return (
            f"NamedLocations[source={self.source.value}, "
            f"locations={len(self.locations)}, "
            f"vars={len(self.variables)}, "
            f"dates={self.date_range.start}..{self.date_range.end}]"
        )


@dataclass(frozen=True)
class GridPlan(FetchPlan):
    """Fetch a satellite source for every point in a grid."""

    source: Source
    date_range: DateRange
    grid: GridSpec
    variables: tuple[VariableSpec, ...]

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("GridPlan requires at least one variable.")
        for v in self.variables:
            if v.source is not self.source:
                raise ValueError(
                    f"Variable '{v.variable_id}' is from source {v.source} but "
                    f"plan source is {self.source}."
                )

    def describe(self) -> str:
        return (
            f"Grid[source={self.source.value}, "
            f"points={self.grid.n_points}, "
            f"vars={len(self.variables)}, "
            f"dates={self.date_range.start}..{self.date_range.end}]"
        )


@dataclass(frozen=True)
class GroundFilePlan(FetchPlan):
    """Parse and ingest a heterogeneous ground-measurement file."""

    source_id: str
    file_path: Path
    adapter_id: str

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("GroundFilePlan.source_id must be non-empty.")
        if not self.adapter_id:
            raise ValueError("GroundFilePlan.adapter_id must be non-empty.")
        if not self.file_path.exists():
            raise ValueError(
                f"GroundFilePlan.file_path does not exist: {self.file_path}. "
                f"Pass an absolute path to a parseable CSV/Excel/etc."
            )

    def describe(self) -> str:
        return (
            f"GroundFile[source={self.source_id}, "
            f"adapter={self.adapter_id}, "
            f"path={self.file_path.name}]"
        )
