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

from .. import schema
from ..warehouse_ops.io.bq import BigQueryClient
from ..warehouse_ops.io.config import TableRefs, TableSchemas, WarehouseOptions
from ..warehouse_ops.io.repositories import GroundRepository, SatelliteRepository
from ..warehouse_ops.population.types import Source, satellite_irradiance_column
from .feature_selection import FeatureSelection

# Source-id → (TableRefs attr, column prefix). Single source of truth for
# every consumer (training pairs, inference features, Predictor): they all
# go through :meth:`FeatureService.build_satellite_features`, so no other
# module needs to import this.
_LONG_AUX_TABLES: dict[Source, tuple[str, str]] = {
    Source.NASA_POWER: ("nasa_daily_vars_long", "nasa"),
    Source.CAMS: ("cams_daily_vars_long", "cams"),
    Source.MERRA_2: ("merra_daily_vars_long", "merra"),
}


class FeatureService:
    """Assembles training pairs and inference feature frames."""

    def __init__(
        self,
        bq: BigQueryClient,
        tables: Optional[TableRefs] = None,
        opts: Optional[WarehouseOptions] = None,
    ) -> None:
        # `tables` is uniquely determined by `bq.config` for any production
        # deployment; default to the derived value so callers don't have to
        # construct two parallel objects. Pass explicitly only when running
        # against a sandbox dataset.
        self._bq = bq
        self._t = tables if tables is not None else TableRefs(config=bq.config)
        self._opts = opts or WarehouseOptions()
        self._ground = GroundRepository(bq, self._t)
        self._sat = SatelliteRepository(bq, self._t)

    @property
    def tables(self) -> TableRefs:
        return self._t

    @property
    def geohash_precision(self) -> int:
        """Geohash precision the warehouse + queries use, e.g. ``5``."""
        return self._opts.geohash_precision

    # ------------------------------------------------------------------
    # Satellite features — the one loop every other method composes on top of.
    # ------------------------------------------------------------------

    def build_satellite_features(
        self,
        *,
        selection: FeatureSelection,
        date_start: date,
        date_end: date,
        geohash5s: Sequence[str],
    ) -> pd.DataFrame:
        """Wide ``(date, geohash5)`` frame: irradiance + per-source aux.

        The single place the satellite-side fetch + pivot + merge loop
        lives. :meth:`build_training_pairs`, :meth:`build_inference_features`,
        and :class:`~susse.inference.Predictor` all delegate here.

        Args:
            selection: What to pull, per source.
            date_start, date_end: Inclusive date bounds.
            geohash5s: Geohash5 cells to query. Must be non-empty.

        Returns:
            Wide DataFrame with columns ``date``, ``geohash5``, one
            ``sat_<band>_<source>_kwh_m2_day`` per requested (band, source),
            and ``<prefix>_<variable_id>`` per requested aux variable.
            Rows are present only for (date, geohash5) the warehouse
            actually has — callers that need placeholder rows synthesize
            them after this returns.

        Raises:
            ValueError: If ``geohash5s`` is empty.
            NotImplementedError: If ``selection.modis_variable_ids`` is
                non-empty. MODIS lives at composite cadence in a separate
                table and is not yet wired through this assembler.
        """
        if not geohash5s:
            raise ValueError(
                "build_satellite_features: geohash5s is empty. Pass at "
                "least one geohash5 string."
            )
        if selection.modis_variable_ids:
            raise NotImplementedError(
                "FeatureSelection.modis_variable_ids is not yet wired "
                "through FeatureService. MODIS data lives at composite-"
                "cadence dates in modis_observations; the forward-fill "
                "semantics will land with NB 03 once C8 has populated "
                "the table. Leave modis_variable_ids empty for now."
            )
        gh_tuple = tuple(geohash5s)
        irr = self._sat.daily_irradiance_by_geohash(
            date_start,
            date_end,
            sources=tuple(s.value for s in selection.include_satellite_irradiance),
            bands=selection.include_satellite_bands,
            geohash5s=gh_tuple,
        )
        df = irr
        for source, ids in self._aux_requests(selection):
            table_attr, prefix = _LONG_AUX_TABLES[source]
            aux = self._sat.long_aux_pivoted(
                table_fqn=getattr(self._t, table_attr),
                column_prefix=prefix,
                start=date_start,
                end=date_end,
                variable_ids=ids,
                geohash5s=gh_tuple,
            )
            # LEFT-join: rows of the result are exactly irradiance's
            # (date, geohash5) pairs. If a cell has aux data but no
            # irradiance, callers must treat it as missing — the Predictor
            # cache-miss detector relies on this definition.
            df = self._merge_left(df, aux, on=(schema.DATE, schema.GEOHASH5))
        if df.empty:
            return pd.DataFrame(columns=[schema.DATE, schema.GEOHASH5])
        return df

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
              * ``sat_<band>_<source>_kwh_m2_day`` per requested
                (band, source) pair.
              * ``<prefix>_<variable_id>`` per requested aux variable.
              * ``qc_level`` — kept for traceability.
        """
        ground = self._ground.fetch(
            date_start,
            date_end,
            locations=locations,
            qc_levels=selection.qc_levels,
        )
        rename_map = {schema.GROUND_GHI: schema.TRAINING_TARGET}
        if ground.empty:
            return ground.rename(columns=rename_map)
        ground = ground.rename(columns=rename_map)
        plan_geohashes = tuple(ground[schema.GEOHASH5].unique())
        satellite = self.build_satellite_features(
            selection=selection,
            date_start=date_start,
            date_end=date_end,
            geohash5s=plan_geohashes,
        )
        df = self._merge_left(ground, satellite, on=(schema.DATE, schema.GEOHASH5))
        return df.sort_values([schema.DATE, schema.LOCATION]).reset_index(drop=True)

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

        Uses geohash-binned joins (same precision as the warehouse default).
        If the warehouse has no row for this (geohash, date), a placeholder
        row with NaN satellite columns is returned so callers always get
        exactly one row.
        """
        import pygeohash

        gh = pygeohash.encode(lat, lon, precision=self._opts.geohash_precision)
        satellite = self.build_satellite_features(
            selection=selection,
            date_start=target_date,
            date_end=target_date,
            geohash5s=(gh,),
        )
        if satellite.empty:
            df = self._empty_irradiance_placeholder(
                target_date=target_date, geohash5=gh, selection=selection
            )
        else:
            df = satellite
        df[schema.LAT] = float(lat)
        df[schema.LON] = float(lon)
        return df

    @staticmethod
    def _empty_irradiance_placeholder(
        *,
        target_date: date,
        geohash5: str,
        selection: FeatureSelection,
    ) -> pd.DataFrame:
        """Synthesise a single-row frame with NaN sat columns.

        Used by :meth:`build_inference_features` so the caller always gets
        exactly one row even when the warehouse has nothing for this cell.
        """
        row: dict[str, object] = {schema.DATE: target_date, schema.GEOHASH5: geohash5}
        for band in selection.include_satellite_bands:
            for src in selection.include_satellite_irradiance:
                row[satellite_irradiance_column(src, band)] = pd.NA
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Manifest support
    # ------------------------------------------------------------------

    def warehouse_table_mods(self, selection: FeatureSelection) -> dict[str, str]:
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
