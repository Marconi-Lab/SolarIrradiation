"""Read-only repositories over the warehouse tables.

Each repository wraps one logical table-family (ground, satellite-irradiance,
per-source long-format aux variables) and exposes typed query methods that
return pandas DataFrames. The :class:`FeatureService` orchestrates these
into the assembled training / inference frame.

These are intentionally thin: each method is one SQL query. Joins live in
the FeatureService layer where pandas can replace BigQuery's per-table
limits and the call sites are easier to read.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

import pandas as pd

from .bq import BigQueryClient
from .config import TableRefs


class GroundRepository:
    """Curated daily ground GHI measurements with QC, location, geohash."""

    def __init__(self, bq: BigQueryClient, tables: TableRefs) -> None:
        self._bq = bq
        self._t = tables

    def fetch(
        self,
        start: date,
        end: date,
        *,
        locations: Optional[Sequence[str]] = None,
        qc_levels: Sequence[str] = ("pass",),
    ) -> pd.DataFrame:
        """Fetch ground measurements between ``start`` and ``end`` (inclusive).

        Args:
            start, end: Date bounds (inclusive).
            locations: Optional list of station names; ``None`` returns all.
            qc_levels: QC labels to keep. Default ``("pass",)`` excludes
                fail-range etc.

        Returns:
            DataFrame with columns ``date, location, lat, lon, geohash5,
            ghi_kwh_m2_day, qc_level``.
        """
        if not qc_levels:
            raise ValueError(
                "qc_levels must be non-empty. To keep all rows, pass every "
                "level explicitly (e.g. ('pass', 'fail_range'))."
            )
        filters = [f"date BETWEEN DATE('{start}') AND DATE('{end}')"]
        if locations:
            quoted = ", ".join(f"'{loc}'" for loc in locations)
            filters.append(f"location IN ({quoted})")
        qc_quoted = ", ".join(f"'{q}'" for q in qc_levels)
        filters.append(f"qc_level IN ({qc_quoted})")
        where = " AND ".join(filters)
        sql = f"""
        SELECT date, location, lat, lon, geohash5,
               ghi_kwh_m2_day, qc_level
        FROM `{self._t.ground_measurements}`
        WHERE {where}
        """
        return self._bq.query(sql)


class SatelliteRepository:
    """Daily satellite irradiance + per-source pivoted aux variables."""

    def __init__(self, bq: BigQueryClient, tables: TableRefs) -> None:
        self._bq = bq
        self._t = tables

    def daily_irradiance_by_geohash(
        self,
        start: date,
        end: date,
        *,
        sources: Sequence[str] = ("NASA", "CAMS"),
        geohash5s: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Per-(date, geohash5) wide irradiance for the requested sources.

        Returns columns ``date, geohash5, sat_ghi_<source>_kwh_m2_day`` —
        one extra column per source. Empty list of sources returns an
        empty frame (no SQL trip).
        """
        if not sources:
            return pd.DataFrame()
        select_cols = [
            f"MAX(IF(source = '{src}', ghi_kwh_m2_day, NULL)) "
            f"AS sat_ghi_{src.lower()}_kwh_m2_day"
            for src in sources
        ]
        filters = [f"date BETWEEN DATE('{start}') AND DATE('{end}')"]
        if geohash5s:
            gh_list = ", ".join(f"'{g}'" for g in geohash5s)
            filters.append(f"geohash5 IN ({gh_list})")
        where = " AND ".join(filters)
        sql = f"""
        SELECT date, geohash5,
               {', '.join(select_cols)}
        FROM `{self._t.irradiance_daily}`
        WHERE {where}
        GROUP BY date, geohash5
        """
        return self._bq.query(sql)

    def long_aux_pivoted(
        self,
        *,
        table_fqn: str,
        column_prefix: str,
        start: date,
        end: date,
        variable_ids: Sequence[str],
        geohash5s: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Pivot a long-format aux table ``(date, geohash5, variable_id, value)``.

        Args:
            table_fqn: Fully-qualified BQ table name (e.g.
                ``project.solar_warehouse.nasa_daily_vars_long``).
            column_prefix: Source tag prepended to every output column to
                avoid collisions across sources (``"nasa"``, ``"cams"``,
                ``"merra"``). E.g. ``T2M`` → ``nasa_T2M``.
            start, end: Inclusive date bounds.
            variable_ids: Variables to pivot. Empty list → empty frame.
            geohash5s: Optional location scope.

        Returns:
            Wide DataFrame with columns ``date, geohash5, <prefix>_<var>...``.
        """
        if not variable_ids:
            return pd.DataFrame()
        # PIVOT clause supports per-pivot-column aliasing via "AS alias".
        pivot_in = ", ".join(
            f"'{vid}' AS {column_prefix}_{vid}" for vid in variable_ids
        )
        var_list = ", ".join(f"'{vid}'" for vid in variable_ids)
        filters = [
            f"date BETWEEN DATE('{start}') AND DATE('{end}')",
            f"variable_id IN ({var_list})",
        ]
        if geohash5s:
            gh_list = ", ".join(f"'{g}'" for g in geohash5s)
            filters.append(f"geohash5 IN ({gh_list})")
        where = " AND ".join(filters)
        sql = f"""
        SELECT * FROM (
          SELECT date, geohash5, variable_id, value
          FROM `{table_fqn}`
          WHERE {where}
        )
        PIVOT (ANY_VALUE(value) FOR variable_id IN ({pivot_in}))
        ORDER BY date, geohash5
        """
        return self._bq.query(sql)

    def warehouse_table_mods(
        self, table_ids: Sequence[str]
    ) -> dict[str, str]:
        """Return ``{table_id: last_modified_time_iso}`` for each table.

        Used by the dataset manifest to capture the warehouse state at
        snapshot-build time. Rows for tables that don't exist are simply
        omitted.
        """
        if not table_ids:
            return {}
        quoted = ", ".join(f"'{t}'" for t in table_ids)
        config = self._bq.config
        sql = f"""
        SELECT table_id,
               TIMESTAMP_MILLIS(last_modified_time) AS last_modified_ts
        FROM `{config.project_id}.{config.dataset}.__TABLES__`
        WHERE table_id IN ({quoted})
        """
        df = self._bq.query(sql)
        return {
            row["table_id"]: pd.Timestamp(row["last_modified_ts"]).isoformat()
            for _, row in df.iterrows()
        }
