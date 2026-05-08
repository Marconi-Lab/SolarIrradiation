"""Warehouse identity, table catalogue, and feature-assembly options.

This module is the single source of truth for the BigQuery warehouse layout.
It declares:

* :class:`WarehouseConfig` — the GCP project + dataset hosting the warehouse.
* :class:`TableSchema` and :class:`TableSchemas` — the static identity of each
  table (table_id, MERGE-key contract, description). Add a new table here
  whenever you introduce one.
* :class:`TableRefs` — fully-qualified BQ table names derived from a
  :class:`WarehouseConfig`.
* :class:`MatchStrategy` and :class:`WarehouseOptions` — query-time options
  used by the feature-assembly layer.

Per CLAUDE.md, structural constants (table_ids, MERGE keys) live as
``ClassVar`` entries on :class:`TableSchemas`, not as module-level globals.
Callers that need both the FQTN and the merge keys for a table should
reference ``TableSchemas.<NAME>`` and combine with
``WarehouseConfig.fqn(...)`` rather than carrying the strings around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar


@dataclass(frozen=True)
class WarehouseConfig:
    """Identifies the GCP project and BigQuery dataset hosting the warehouse.

    Default values point to the production warehouse used by the SuSSE project.
    Tests and alternate environments construct an instance with explicit
    project/dataset values.

    Attributes:
        project_id: GCP project ID (e.g. ``solar-irradiation-estimation``).
        dataset: BigQuery dataset name within the project.
        location: GCP region for query/job placement (e.g. ``EU``). ``None``
            lets the BQ client infer from the dataset.
    """

    project_id: str = "solar-irradiation-estimation"
    dataset: str = "solar_warehouse"
    location: str | None = None

    def fqn(self, table_id: str) -> str:
        """Return the fully-qualified BQ name for a table in this dataset."""
        return f"{self.project_id}.{self.dataset}.{table_id}"


@dataclass(frozen=True)
class TableSchema:
    """Static identity of a warehouse table.

    Bundles the unqualified ``table_id`` with its MERGE-deduplication contract
    so loaders and coverage queries always reference the same key columns.
    """

    table_id: str
    merge_keys: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.table_id:
            raise ValueError("TableSchema.table_id must be non-empty.")
        if not self.merge_keys:
            raise ValueError(
                f"TableSchema for '{self.table_id}' must declare at least one "
                f"merge key. If the table is append-only with no natural key, "
                f"use a synthetic one (e.g. an ingest_id column)."
            )


class TableSchemas:
    """Registry of all known warehouse tables.

    Single source of truth for ``table_id`` strings and MERGE-key contracts
    across the codebase. Adding a new table is a one-line ``ClassVar`` edit
    here; loaders, coverage queries, and tests pick it up automatically.
    """

    NASA_DAILY_VARS_LONG: ClassVar[TableSchema] = TableSchema(
        table_id="nasa_daily_vars_long",
        merge_keys=("date", "geohash5", "variable_id", "source"),
        description=(
            "Long-format daily NASA POWER variables. One row per "
            "(date, point, variable). Source='NASA'."
        ),
    )
    CAMS_DAILY_VARS_LONG: ClassVar[TableSchema] = TableSchema(
        table_id="cams_daily_vars_long",
        merge_keys=("date", "geohash5", "variable_id", "source"),
        description=(
            "Long-format daily CAMS variables. Mirrors NASA_DAILY_VARS_LONG "
            "but for the CAMS source."
        ),
    )
    MERRA_DAILY_VARS_LONG: ClassVar[TableSchema] = TableSchema(
        table_id="merra_daily_vars_long",
        merge_keys=("date", "geohash5", "variable_id", "source"),
        description=(
            "Long-format daily MERRA-2 variables. Mirrors NASA_DAILY_VARS_LONG. "
            "Daily values are cosine-zenith-weighted means of MERRA-2's "
            "hourly-native data, aggregated client-side."
        ),
    )
    MODIS_OBSERVATIONS: ClassVar[TableSchema] = TableSchema(
        table_id="modis_observations",
        merge_keys=("date", "geohash5", "product_id", "band_id", "source"),
        description=(
            "MODIS satellite observations from ORNL DAAC. Each row is one "
            "(date, point, product, band) measurement at the product's "
            "native composite end-date. Distinguished from the *_long "
            "tables by carrying product_id and band_id as separate columns."
        ),
    )
    IRRADIANCE_DAILY: ClassVar[TableSchema] = TableSchema(
        table_id="irradiance_daily",
        merge_keys=("date", "geohash5", "source"),
        description=(
            "Wide-format daily irradiance with GHI/DHI/DNI per source per "
            "point. Used for feature assembly."
        ),
    )
    GROUND_MEASUREMENTS: ClassVar[TableSchema] = TableSchema(
        table_id="ground_measurements",
        merge_keys=("date", "location"),
        description=(
            "Curated daily ground GHI measurements with QC flag, geohash, "
            "and curation provenance."
        ),
    )
    GROUND_MEASUREMENTS_RAW: ClassVar[TableSchema] = TableSchema(
        table_id="ground_measurements_raw",
        merge_keys=("datetime", "location"),
        description=(
            "Raw uncurated ground GHI measurements as parsed from source "
            "files. Curation reads from this table."
        ),
    )
    DIM_VARIABLE: ClassVar[TableSchema] = TableSchema(
        table_id="dim_variable",
        merge_keys=("variable_id", "source"),
        description=(
            "Catalogue of satellite/ground variables: ID, units, valid range, "
            "spatial resolution, description."
        ),
    )


@dataclass(frozen=True)
class TableRefs:
    """Fully-qualified BigQuery table names for the warehouse.

    Derived from a :class:`WarehouseConfig`; properties return FQTNs so
    callers can use attribute access (``tables.irradiance_daily``) without
    embedding string concatenation.
    """

    config: WarehouseConfig = field(default_factory=WarehouseConfig)

    @property
    def nasa_daily_vars_long(self) -> str:
        return self.config.fqn(TableSchemas.NASA_DAILY_VARS_LONG.table_id)

    @property
    def cams_daily_vars_long(self) -> str:
        return self.config.fqn(TableSchemas.CAMS_DAILY_VARS_LONG.table_id)

    @property
    def merra_daily_vars_long(self) -> str:
        return self.config.fqn(TableSchemas.MERRA_DAILY_VARS_LONG.table_id)

    @property
    def modis_observations(self) -> str:
        return self.config.fqn(TableSchemas.MODIS_OBSERVATIONS.table_id)

    @property
    def irradiance_daily(self) -> str:
        return self.config.fqn(TableSchemas.IRRADIANCE_DAILY.table_id)

    @property
    def ground_measurements(self) -> str:
        return self.config.fqn(TableSchemas.GROUND_MEASUREMENTS.table_id)

    @property
    def ground_measurements_raw(self) -> str:
        return self.config.fqn(TableSchemas.GROUND_MEASUREMENTS_RAW.table_id)

    @property
    def dim_variable(self) -> str:
        return self.config.fqn(TableSchemas.DIM_VARIABLE.table_id)

    @property
    def fn_nearest_point(self) -> str:
        return self.config.fqn("fn_nearest_point")

    @property
    def fn_nearest_var_daily(self) -> str:
        return self.config.fqn("fn_nearest_var_daily")


class MatchStrategy(StrEnum):
    """How to match a ground point to its nearest satellite point."""

    GEOHASH = "geohash"
    NEAREST = "nearest"


@dataclass(frozen=True)
class WarehouseOptions:
    """Behaviour flags for feature assembly and query-time joins."""

    geohash_precision: int = 5
    include_cams: bool = True
    include_nasa: bool = True
    match_strategy: MatchStrategy = MatchStrategy.GEOHASH

    def __post_init__(self) -> None:
        if not (1 <= self.geohash_precision <= 12):
            raise ValueError(
                f"geohash_precision={self.geohash_precision} is outside [1, 12]. "
                f"5 is the warehouse default and matches the existing tables."
            )
