"""Warehouse population subsystem.

Three ingest patterns:

* :class:`NamedLocationsPlan` + :class:`CamsSatelliteJob` /
  :class:`NasaPowerSatelliteJob` — fetch a satellite source for a list of
  named points (typically ground stations).
* :class:`GridPlan` + the same satellite jobs — fetch a satellite source
  for every point of a lat/lon grid (portal coverage cache).
* :class:`GroundFilePlan` + :class:`GroundIngestJob` — parse a heterogeneous
  ground-measurement file via a :class:`GroundSourceAdapter` and curate.

All jobs are idempotent: they pre-check what's already in the warehouse
and skip API calls / disk reads for rows that are cached, then rely on
MERGE-on-write to dedupe at the table level.
"""

from .adapters import GroundSourceAdapter, StandardCsvAdapter
from .base_job import BaseJob, JobResult
from .coverage import CoverageRepository
from .curation import CurationOptions, curate_ground
from .dim_variable import VariableCatalog, populate_dim_variable, variables_to_dataframe
from .jobs import (
    BaseSatelliteJob,
    CamsSatelliteJob,
    GroundIngestJob,
    NasaPowerSatelliteJob,
)
from .loaders import DerivedColumn, MergeLoader, MergeSpec
from .types import (
    BoundingBox,
    DateRange,
    FetchPlan,
    GridPlan,
    GridSpec,
    GroundFilePlan,
    LocationSpec,
    NamedLocationsPlan,
    Source,
    VariableSpec,
)
from .validators import (
    validate_ground_curated,
    validate_ground_raw,
    validate_long_format,
)

__all__ = [
    "BaseJob",
    "BaseSatelliteJob",
    "BoundingBox",
    "CamsSatelliteJob",
    "CoverageRepository",
    "CurationOptions",
    "DateRange",
    "DerivedColumn",
    "FetchPlan",
    "GridPlan",
    "GridSpec",
    "GroundFilePlan",
    "GroundIngestJob",
    "GroundSourceAdapter",
    "JobResult",
    "LocationSpec",
    "MergeLoader",
    "MergeSpec",
    "NamedLocationsPlan",
    "NasaPowerSatelliteJob",
    "Source",
    "StandardCsvAdapter",
    "VariableCatalog",
    "VariableSpec",
    "curate_ground",
    "populate_dim_variable",
    "validate_ground_curated",
    "validate_ground_raw",
    "validate_long_format",
    "variables_to_dataframe",
]
