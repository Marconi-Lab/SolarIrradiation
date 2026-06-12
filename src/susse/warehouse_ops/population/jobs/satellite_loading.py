"""Shared BigQuery-load mechanics for satellite ingest jobs.

The per-point :class:`BaseSatelliteJob` and the region-shaped
:class:`MerraRegionJob` / :class:`NasaPowerRegionJob` all land long-format
rows in a ``*_daily_vars_long`` table and — for sources that carry
irradiance — wide rows in ``irradiance_daily``. This module is the single
home of that shared mechanics (the GEOGRAPHY derivation, the long→wide
irradiance pivot, the MERGE-load wrappers) so the three jobs cannot drift
apart.
"""

from __future__ import annotations

import pandas as pd

from ...io.bq import BigQueryClient
from ...io.config import TableSchema, TableSchemas
from ..loaders import DerivedColumn, MergeLoader, MergeSpec
from ..validators import validate_long_format

# Server-side derivation: the GEOGRAPHY column is built from staging lat/lon
# during the MERGE rather than carried across the wire by pandas.
GEOG_DERIVATION: tuple[DerivedColumn, ...] = (
    DerivedColumn(name="geog", sql_expr="ST_GEOGPOINT(longitude, latitude)"),
)

# Long-format irradiance variable_id → wide ``irradiance_daily`` column name.
_IRRADIANCE_COLUMN_BY_VARIABLE: dict[str, str] = {
    "ghi": "ghi_kwh_m2_day",
    "dhi": "dhi_kwh_m2_day",
    "dni": "dni_kwh_m2_day",
}

# Column order each table's MERGE-load expects.
_LONG_LOAD_COLUMNS: tuple[str, ...] = (
    "date",
    "latitude",
    "longitude",
    "geohash5",
    "variable_id",
    "value",
    "source",
)
_IRRADIANCE_LOAD_COLUMNS: tuple[str, ...] = (
    "date",
    "latitude",
    "longitude",
    "geohash5",
    "source",
    "ghi_kwh_m2_day",
    "dhi_kwh_m2_day",
    "dni_kwh_m2_day",
    "reliability",
)


def irradiance_long_to_wide(irradiance_long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot already-filtered long irradiance rows into the wide shape.

    Args:
        irradiance_long_df: Long-format rows whose ``variable_id`` is one of
            ``ghi`` / ``dhi`` / ``dni`` (the caller filters; this function
            does not). Must carry ``date, latitude, longitude, geohash5,
            source, variable_id, value``.

    Returns:
        One row per (date, cell) with named ``*_kwh_m2_day`` columns. Bands
        absent from the input become NULL, as does ``reliability`` — no
        satellite source in the catalogue publishes a reliability series.
    """
    wide = irradiance_long_df.pivot_table(
        index=["date", "latitude", "longitude", "geohash5", "source"],
        columns="variable_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns=_IRRADIANCE_COLUMN_BY_VARIABLE)
    for col in _IRRADIANCE_COLUMN_BY_VARIABLE.values():
        if col not in wide.columns:
            wide[col] = pd.NA
    wide["reliability"] = pd.NA
    return wide


def load_long(
    bq: BigQueryClient,
    *,
    table_fqn: str,
    schema: TableSchema,
    df: pd.DataFrame,
    context: str,
) -> int:
    """Validate and MERGE-load long-format rows into a ``*_daily_vars_long`` table.

    Args:
        bq: BigQuery client.
        table_fqn: Fully-qualified target table name.
        schema: The target table's :class:`TableSchema` (carries merge keys).
        df: Long-format rows; must carry every column in
            :data:`_LONG_LOAD_COLUMNS`.
        context: Label for the validation error message (usually the job
            name).

    Returns:
        Number of rows staged for the MERGE.
    """
    validate_long_format(df, context=context)
    loader = MergeLoader(
        bq=bq,
        table_fqn=table_fqn,
        spec=MergeSpec(schema=schema, derived_columns=GEOG_DERIVATION),
    )
    return loader.load(df[list(_LONG_LOAD_COLUMNS)])


def load_irradiance(bq: BigQueryClient, *, table_fqn: str, df: pd.DataFrame) -> int:
    """MERGE-load wide irradiance rows into ``irradiance_daily``.

    Args:
        bq: BigQuery client.
        table_fqn: Fully-qualified ``irradiance_daily`` table name.
        df: Wide rows; must carry every column in
            :data:`_IRRADIANCE_LOAD_COLUMNS`.

    Returns:
        Number of rows staged for the MERGE.
    """
    loader = MergeLoader(
        bq=bq,
        table_fqn=table_fqn,
        spec=MergeSpec(
            schema=TableSchemas.IRRADIANCE_DAILY,
            derived_columns=GEOG_DERIVATION,
        ),
    )
    return loader.load(df[list(_IRRADIANCE_LOAD_COLUMNS)])
