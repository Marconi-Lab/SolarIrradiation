from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
from datetime import date

import pandas as pd
from google.cloud import bigquery

from .bq import BQ
from .config import TableRefs

# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class GroundRepository:
    """Access curated ground measurements.

    Returns daily records with (date, location, lat, lon, geog, geohash5,
    ghi_kwh_m2_day, qc_level, ...).
    """

    def __init__(self, bq: BQ, tables: TableRefs) -> None:
        self._bq = bq
        self._t = tables

    def fetch(
        self,
        start: date,
        end: date,
        locations: Optional[Sequence[str]] = None,
        qc_levels: Optional[Sequence[str]] = ("pass",),
    ) -> pd.DataFrame:
        """Fetch ground measurements between start and end (inclusive).

        Parameters
        ----------
        start, end: date bounds (inclusive)
        locations: optional filter by location names
        qc_levels: e.g., ("pass",) to keep only QC-passed rows
        """
        filters = [
            f"date BETWEEN DATE('{start}') AND DATE('{end}')",
        ]
        if locations:
            loc_list = ",".join([f"'{l}'" for l in locations])
            filters.append(f"location IN ({loc_list})")
        if qc_levels:
            lvl_list = ",".join([f"'{q}'" for q in qc_levels])
            filters.append(f"qc_level IN ({lvl_list})")

        where = " AND \n      ".join(filters)
        sql = f"""
        SELECT date, location, lat, lon, geog, geohash5,
               ghi_kwh_m2_day, qc_level
        FROM {self._t.ground_measurements}
        WHERE {where}
        """
        return self._bq.df(sql)


class SatelliteRepository:
    """Access daily satellite irradiance (NASA/CAMS)."""

    def __init__(self, bq: BQ, tables: TableRefs) -> None:
        self._bq = bq
        self._t = tables

    def daily_irradiance_by_geohash(
        self,
        start: date,
        end: date,
        sources: Optional[Sequence[str]] = ("NASA", "CAMS"),
    ) -> pd.DataFrame:
        """Returns per-(date, geohash5) daily irradiance for requested sources.

        Columns: date, geohash5, sat_ghi_cams_kwh_m2_day, sat_ghi_nasa_kwh_m2_day
        """
        src_case = []
        if sources is None:
            sources = []
        if "CAMS" in sources:
            src_case.append(
                "MAX(IF(source = 'CAMS', ghi_kwh_m2_day, NULL)) AS sat_ghi_cams_kwh_m2_day"
            )
        if "NASA" in sources:
            src_case.append(
                "MAX(IF(source = 'NASA', ghi_kwh_m2_day, NULL)) AS sat_ghi_nasa_kwh_m2_day"
            )
        if not src_case:
            # Return empty
            return pd.DataFrame()

        sql = f"""
        SELECT date, geohash5,
               {', '.join(src_case)}
        FROM {self._t.irradiance_daily}
        WHERE date BETWEEN DATE('{start}') AND DATE('{end}')
        GROUP BY date, geohash5
        """
        return self._bq.df(sql)

    def nasa_vars_pivoted(
        self,
        start: date,
        end: date,
        variables: Sequence[str],
    ) -> pd.DataFrame:
        """Fetch selected NASA variables (long) and pivot to wide by (date, geohash5).

        Parameters
        ----------
        variables: list of variable_id (or variable code) to include.
        """
        if not variables:
            return pd.DataFrame()
        var_list = ",".join([f"'{v}'" for v in variables])
        sql = f"""
        WITH base AS (
          SELECT date, geohash5, variable_id, value
          FROM {self._t.nasa_daily_vars_long}
          WHERE date BETWEEN DATE('{start}') AND DATE('{end}')
            AND variable_id IN ({var_list})
        )
        SELECT * FROM base
        PIVOT ( ANY_VALUE(value) FOR variable_id IN ({var_list}) )
        ORDER BY date, geohash5
        """
        return self._bq.df(sql)

