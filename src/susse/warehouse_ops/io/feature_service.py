from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
from datetime import date

import pandas as pd
from google.cloud import bigquery
from .bq import BigQueryClient
from .config import TableRefs, WarehouseOptions
from .repositories import SatelliteRepository, GroundRepository

# ---------------------------------------------------------------------------
# Feature assembler (joins repositories; match by geohash)
# ---------------------------------------------------------------------------


class FeatureService:
    """High-level service that assembles training and inference feature sets.

    This class hides SQL details. By default it matches ground ↔ satellite
    by (date, geohash5). Nearest-distance matching can be added later.
    """

    def __init__(self, bq: BigQueryClient, tables: TableRefs, opts: WarehouseOptions | None = None) -> None:
        self._bq = bq
        self._t = tables
        self._opts = opts or WarehouseOptions()

    def build_training_pairs(
        self,
        start: date,
        end: date,
        locations: Optional[Sequence[str]] = None,
        nasa_variables: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return a model-ready table joining ground ↔ satellite (and NASA vars).

        Columns include:
          - date, location, lat, lon, geohash5
          - y_ghi_kwh_m2_day (ground target)
          - sat_ghi_cams_kwh_m2_day, sat_ghi_nasa_kwh_m2_day (features)
          - optional NASA variables (pivoted): one column per variable_id
        """
        # Base: ground measurements
        if locations:
            quoted = ", ".join([f"'{l}'" for l in locations])
            loc_filter = f"AND location IN ({quoted})"
        else:
            loc_filter = ""

        sql = f"""
        WITH gm AS (
          SELECT date, location, lat, lon, geohash5, ghi_kwh_m2_day AS y_ghi_kwh_m2_day
          FROM {self._t.ground_measurements}
          WHERE date BETWEEN DATE('{start}') AND DATE('{end}')
            AND qc_level = 'pass' {loc_filter}
        ),
        sat AS (
          SELECT date, geohash5,
                 MAX(IF(source='CAMS', ghi_kwh_m2_day, NULL)) AS sat_ghi_cams_kwh_m2_day,
                 MAX(IF(source='NASA', ghi_kwh_m2_day, NULL)) AS sat_ghi_nasa_kwh_m2_day
          FROM {self._t.irradiance_daily}
          WHERE date BETWEEN DATE('{start}') AND DATE('{end}')
          GROUP BY date, geohash5
        )
        SELECT gm.*, sat.sat_ghi_cams_kwh_m2_day, sat.sat_ghi_nasa_kwh_m2_day
        FROM gm
        LEFT JOIN sat USING (date, geohash5)
        """

        df = self._bq.query(sql)

        # Optionally enrich with NASA vars
        if nasa_variables:
            nv = SatelliteRepository(self._bq, self._t).nasa_vars_pivoted(start, end, nasa_variables)
            if not nv.empty:
                # Join on (date, geohash5)
                df = df.merge(nv, on=["date", "geohash5"], how="left")
        return df

    def build_inference_features(
        self,
        target_date: date,
        lat: float,
        lon: float,
        geohash_precision: Optional[int] = None,
        nasa_variables: Optional[Sequence[str]] = None,
        *,
        nearest: bool = False,
        max_km: Optional[float] = None,
        fill_missing_with_nearest: bool = True,
    ) -> pd.DataFrame:
        """Return a single-row feature frame for inference at (lat, lon, date).

        If `nearest=True`, uses the nearest-point table functions; otherwise
        uses geohash-binned joins (original behavior). If `fill_missing_with_nearest`
        is True, a geohash miss auto-falls back to nearest within `max_km`.
        """
        gh_prec = geohash_precision or self._opts.geohash_precision
        sat_repo = SatelliteRepository(self._bq, self._t)

        def _geohash_features() -> pd.DataFrame:
            sql_sat = f"""
            WITH pt AS (
              SELECT DATE('{target_date}') AS date,
                     ST_GEOHASH(ST_GEOGPOINT({lon}, {lat}), {gh_prec}) AS geohash5
            ), sat AS (
              SELECT s.date, s.geohash5,
                     MAX(IF(source='CAMS', ghi_kwh_m2_day, NULL)) AS sat_ghi_cams_kwh_m2_day,
                     MAX(IF(source='NASA', ghi_kwh_m2_day, NULL)) AS sat_ghi_nasa_kwh_m2_day
              FROM {self._t.irradiance_daily} s
              JOIN pt USING (date)
              WHERE s.geohash5 = (SELECT geohash5 FROM pt)
              GROUP BY s.date, s.geohash5
            )
            SELECT * FROM sat
            """
            return self._bq.query(sql_sat)

        def _nearest_features() -> pd.DataFrame:
            return sat_repo.nearest_irradiance(lat=lat, lon=lon, start=target_date, end=target_date, max_km=max_km)

        # Satellite features
        if nearest:
            sat_df = _nearest_features()
        else:
            sat_df = _geohash_features()
            if sat_df.empty and fill_missing_with_nearest:
                sat_df = _nearest_features()

        # Ensure a single-row frame even if nothing found
        if sat_df.empty:
            cols = ["date", "sat_ghi_cams_kwh_m2_day", "sat_ghi_nasa_kwh_m2_day"]
            sat_df = pd.DataFrame([[pd.NaT, None, None]], columns=cols)
            sat_df["date"] = pd.to_datetime(target_date).date()

        # Optional NASA variables. The pivoted result carries `geohash5`,
        # so we can merge on (date, geohash5) directly — that selects the
        # single matching row out of the ~1977 the warehouse-wide pivot
        # returns. (Earlier code joined only on date, which produced a
        # 1977-row Cartesian explosion before falling out of the merge.)
        if nasa_variables:
            if nearest:
                nv = sat_repo.nearest_nasa_vars_daily(
                    lat=lat, lon=lon,
                    start=target_date, end=target_date,
                    variables=nasa_variables, max_km=max_km,
                )
                if not sat_df.empty and not nv.empty:
                    sat_df = sat_df.merge(nv, on=["date"], how="left")
            else:
                nv = sat_repo.nasa_vars_pivoted(
                    start=target_date, end=target_date, variables=nasa_variables,
                )
                if not nv.empty and "geohash5" in sat_df.columns:
                    sat_df = sat_df.merge(nv, on=["date", "geohash5"], how="left")

        # Add lat/lon used for traceability
        sat_df["lat"] = float(lat)
        sat_df["lon"] = float(lon)
        return sat_df