"""High-level feature assembler — joins repositories into a model-ready frame.

Drives one end-to-end query path for both training (ground ↔ satellite at a
named station) and inference (satellite-only at any (lat, lon, date)). The
typed :class:`~susse.datasets.FeatureSelection` config tells it which sources
to include and which auxiliary variables to pivot.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

import pandas as pd

from ..warehouse_ops.io.bq import BigQueryClient
from ..warehouse_ops.io.config import TableRefs, TableSchemas, WarehouseOptions
from ..warehouse_ops.io.repositories import GroundRepository, SatelliteRepository
from ..warehouse_ops.population.types import Source
from .feature_selection import FeatureSelection

# Source-id → table accessor, prefix used when pivoting that source's
# long-format aux table. Single source of truth so adding a new aux
# source is one entry here plus an entry in FeatureSelection.
_LONG_AUX_TABLES: dict[Source, tuple[str, str]] = {
    # Source: (TableRefs attr, column prefix)
    Source.NASA_POWER: ("nasa_daily_vars_long", "nasa"),
    Source.CAMS: ("cams_daily_vars_long", "cams"),
    Source.MERRA_2: ("merra_daily_vars_long", "merra"),
}


class FeatureService:
    """Assembles training pairs and inference feature frames."""

    def __init__(
        self,
        bq: BigQueryClient,
        tables: TableRefs,
        opts: Optional[WarehouseOptions] = None,
    ) -> None:
        self._bq = bq
        self._t = tables
        self._opts = opts or WarehouseOptions()
        self._ground = GroundRepository(bq, tables)
        self._sat = SatelliteRepository(bq, tables)

    @property
    def tables(self) -> TableRefs:
        return self._t

    # ------------------------------------------------------------------
    # Training pairs
    # ------------------------------------------------------------------

    def build_training_pairs(
        self,
        *,
        selection: FeatureSelection,
        date_start: date,
        date_end: date,
        locations: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Assemble (target, features) pairs for training.

        Always returns one row per (date, station) where ground GHI is
        present and passes the selection's QC filter; missing satellite
        / aux columns are NaN.

        Args:
            selection: What to pull, per source.
            date_start, date_end: Inclusive date bounds.
            locations: Optional station-name filter; ``None`` = all.

        Returns:
            DataFrame columns:
              * ``date, location, lat, lon, geohash5`` — base.
              * ``y_ghi_kwh_m2_day`` — ground target.
              * ``sat_ghi_<source>_kwh_m2_day`` per requested source.
              * ``<prefix>_<variable_id>`` per requested aux variable
                (prefix matches source: ``nasa_``, ``cams_``, ``merra_``).
              * ``qc_level`` — kept for traceability.
        """
        ground = self._ground.fetch(
            date_start, date_end,
            locations=locations,
            qc_levels=selection.qc_levels,
        )
        if ground.empty:
            return ground.rename(columns={"ghi_kwh_m2_day": "y_ghi_kwh_m2_day"})
        ground = ground.rename(columns={"ghi_kwh_m2_day": "y_ghi_kwh_m2_day"})
        plan_geohashes = tuple(ground["geohash5"].unique())

        irr = self._sat.daily_irradiance_by_geohash(
            date_start, date_end,
            sources=tuple(s.value for s in selection.include_satellite_irradiance),
            geohash5s=plan_geohashes,
        )
        df = self._merge_left(ground, irr, on=("date", "geohash5"))

        for source, ids in self._aux_requests(selection):
            table_attr, prefix = _LONG_AUX_TABLES[source]
            aux = self._sat.long_aux_pivoted(
                table_fqn=getattr(self._t, table_attr),
                column_prefix=prefix,
                start=date_start, end=date_end,
                variable_ids=ids,
                geohash5s=plan_geohashes,
            )
            df = self._merge_left(df, aux, on=("date", "geohash5"))

        if selection.modis_variable_ids:
            # MODIS lives in modis_observations with composite-cadence
            # rows (date = composite end). The forward-fill / as-of join
            # belongs in preprocessing (NB 03), not in materialisation —
            # that way the snapshot stores only what BigQuery actually
            # has, and the daily upsampling decision is documented in
            # the preprocessing config.
            raise NotImplementedError(
                "FeatureSelection.modis_variable_ids is not yet wired "
                "through FeatureService.build_training_pairs. MODIS data "
                "lives at composite-cadence dates in modis_observations; "
                "the forward-fill semantics will land with NB 03 once C8 "
                "has populated the table. Leave modis_variable_ids empty "
                "for now."
            )
        return df.sort_values(["date", "location"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Inference features
    # ------------------------------------------------------------------

    def build_inference_features(
        self,
        *,
        selection: FeatureSelection,
        target_date: date,
        lat: float,
        lon: float,
    ) -> pd.DataFrame:
        """Single-row feature frame for inference at ``(lat, lon, target_date)``.

        Uses geohash-binned joins (same precision as the warehouse
        default). No ground truth is included — that's what the model is
        going to predict.
        """
        if selection.modis_variable_ids:
            raise NotImplementedError(
                "FeatureSelection.modis_variable_ids is not yet wired "
                "through FeatureService.build_inference_features. See "
                "build_training_pairs for the same constraint."
            )
        import pygeohash
        gh = pygeohash.encode(lat, lon, precision=self._opts.geohash_precision)

        irr = self._sat.daily_irradiance_by_geohash(
            target_date, target_date,
            sources=tuple(s.value for s in selection.include_satellite_irradiance),
            geohash5s=(gh,),
        )
        if irr.empty:
            base = pd.DataFrame([{"date": target_date, "geohash5": gh}])
            for src in selection.include_satellite_irradiance:
                base[f"sat_ghi_{src.value.lower()}_kwh_m2_day"] = pd.NA
            df = base
        else:
            df = irr

        for source, ids in self._aux_requests(selection):
            table_attr, prefix = _LONG_AUX_TABLES[source]
            aux = self._sat.long_aux_pivoted(
                table_fqn=getattr(self._t, table_attr),
                column_prefix=prefix,
                start=target_date, end=target_date,
                variable_ids=ids,
                geohash5s=(gh,),
            )
            df = self._merge_left(df, aux, on=("date", "geohash5"))

        df["lat"] = float(lat)
        df["lon"] = float(lon)
        return df

    # ------------------------------------------------------------------
    # Manifest support
    # ------------------------------------------------------------------

    def warehouse_table_mods(
        self, selection: FeatureSelection
    ) -> dict[str, str]:
        """Per-source-table ``last_modified_time`` for manifest provenance.

        Returns the ISO-8601 modification timestamp of each warehouse
        table the selection would touch. Used by the dataset writer to
        snapshot "what was the warehouse state when this dataset was
        built" alongside the parquet payload.
        """
        ids: list[str] = [TableSchemas.GROUND_MEASUREMENTS.table_id]
        if selection.include_satellite_irradiance:
            ids.append(TableSchemas.IRRADIANCE_DAILY.table_id)
        for source, _ in self._aux_requests(selection):
            attr, _ = _LONG_AUX_TABLES[source]
            ids.append(getattr(TableSchemas, attr.upper()).table_id)
        if selection.modis_variable_ids:
            ids.append(TableSchemas.MODIS_OBSERVATIONS.table_id)
        return self._sat.warehouse_table_mods(ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _aux_requests(
        selection: FeatureSelection,
    ) -> list[tuple[Source, tuple[str, ...]]]:
        """Yield ``(source, variable_ids)`` pairs for the requested aux sources."""
        out: list[tuple[Source, tuple[str, ...]]] = []
        if selection.nasa_variable_ids:
            out.append((Source.NASA_POWER, selection.nasa_variable_ids))
        if selection.cams_variable_ids:
            out.append((Source.CAMS, selection.cams_variable_ids))
        if selection.merra_variable_ids:
            out.append((Source.MERRA_2, selection.merra_variable_ids))
        return out

    @staticmethod
    def _merge_left(
        left: pd.DataFrame, right: pd.DataFrame, *, on: tuple[str, ...]
    ) -> pd.DataFrame:
        """LEFT-join right onto left on ``on``; tolerates an empty right side."""
        if right.empty:
            return left
        return left.merge(right, on=list(on), how="left")
