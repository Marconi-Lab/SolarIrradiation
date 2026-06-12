"""High-level feature assembler — joins repositories into a model-ready frame.

Drives one end-to-end query path for both training (ground ↔ satellite at a
named station) and inference (satellite-only at any (lat, lon, date)). The
typed :class:`~susse.datasets.FeatureSelection` config tells it which sources
to include and which auxiliary variables to pivot.

Callers hand the service *query points* — plain ``(lat, lon)`` pairs. Each
satellite source stores its data at its own set of cells, so the service
snaps every query point onto that source's cells independently (via
:class:`~susse.warehouse_ops.snapping.NearestPixelSnapper`) before querying,
then relabels the rows back to the query point's own ``geohash5``. Snapping
is therefore invisible to every caller: a portal developer queries variables
for a coordinate and a date range, and the per-source grid bookkeeping is
handled here and nowhere else.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

import pandas as pd
import pygeohash

from .. import schema
from ..warehouse_ops.io.bq import BigQueryClient
from ..warehouse_ops.io.config import TableRefs, TableSchemas, WarehouseOptions
from ..warehouse_ops.io.repositories import GroundRepository, SatelliteRepository
from ..warehouse_ops.population.types import (
    IrradianceBand,
    Source,
    aux_column_prefix,
    satellite_irradiance_column,
)
from ..warehouse_ops.snapping import NearestPixelSnapper
from .feature_selection import FeatureSelection

# Source-id → TableRefs attr for the long-format aux table. The column
# prefix that pairs with each table is owned by :func:`aux_column_prefix`
# (the single source of truth for the prefix convention) and is *not*
# duplicated here.
_LONG_AUX_TABLES: dict[Source, str] = {
    Source.NASA_POWER: "nasa_daily_vars_long",
    Source.CAMS: "cams_daily_vars_long",
    Source.MERRA_2: "merra_daily_vars_long",
}

# Internal column carrying the query point's own geohash5 through the
# snap/relabel step. Never leaves :meth:`build_satellite_features`.
_QUERY_GEOHASH5: str = "query_geohash5"


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
        # Snappers are built lazily from the warehouse's own cell sets and
        # cached per (table, source) — the distinct-cells scan runs once.
        self._snapper_cache: dict[tuple[str, str], NearestPixelSnapper] = {}

    @property
    def tables(self) -> TableRefs:
        return self._t

    @property
    def geohash_precision(self) -> int:
        """Geohash precision the warehouse + queries use, e.g. ``5``."""
        return self._opts.geohash_precision

    def invalidate_snapper_cache(self) -> None:
        """Drop cached per-source snappers so the next query rebuilds them.

        Snappers are built from the warehouse's native-pixel sets and cached
        for the service's lifetime. A caller that has just ingested new rows
        (e.g. :class:`~susse.inference.Predictor`'s on-demand fetch) must
        call this so the follow-up query snaps against the fresh pixels
        rather than a stale set.
        """
        self._snapper_cache.clear()

    # ------------------------------------------------------------------
    # Satellite features — the one loop every other method composes on top of.
    # ------------------------------------------------------------------

    def build_satellite_features(
        self,
        *,
        selection: FeatureSelection,
        date_start: date,
        date_end: date,
        points: Sequence[tuple[float, float]],
    ) -> pd.DataFrame:
        """Wide ``(date, geohash5)`` frame: irradiance + per-source aux.

        The single place the satellite-side snap + fetch + pivot + merge
        loop lives. :meth:`build_training_pairs`,
        :meth:`build_inference_features`, and
        :class:`~susse.inference.Predictor` all delegate here.

        Each query point is snapped onto every requested source's native
        grid independently; the fetched rows are relabelled to the query
        point's own ``geohash5`` so the result still uses one cell key per
        point regardless of how the sources are gridded.

        Args:
            selection: What to pull, per source.
            date_start, date_end: Inclusive date bounds.
            points: ``(lat, lon)`` query points. Must be non-empty.
                Duplicates (and points sharing a ``geohash5`` cell) collapse
                to one output cell.

        Returns:
            Wide DataFrame with columns ``date``, ``geohash5`` (the query
            point's own geohash5), one ``sat_<band>_<source>_kwh_m2_day``
            per requested (band, source), and ``<prefix>_<variable_id>`` per
            requested aux variable. Rows are present only for (date,
            geohash5) the warehouse actually has — callers that need
            placeholder rows synthesize them after this returns.

        Raises:
            ValueError: If ``points`` is empty.
            NotImplementedError: If ``selection.modis_variable_ids`` is
                non-empty. MODIS lives at composite cadence in a separate
                table and is not yet wired through this assembler.
        """
        if not points:
            raise ValueError(
                "build_satellite_features: points is empty. Pass at least "
                "one (lat, lon) query point."
            )
        if selection.modis_variable_ids:
            raise NotImplementedError(
                "FeatureSelection.modis_variable_ids is not yet wired "
                "through FeatureService. MODIS data lives at composite-"
                "cadence dates in modis_observations; the forward-fill "
                "semantics will land with NB 03 once C8 has populated "
                "the table. Leave modis_variable_ids empty for now."
            )
        query = self._unique_query_points(points)

        # Irradiance spine: union (outer-join) over the requested sources.
        irradiance_frames: list[pd.DataFrame] = []
        for source in selection.include_satellite_irradiance:
            frame = self._irradiance_for_source(
                source,
                query,
                date_start=date_start,
                date_end=date_end,
                bands=selection.include_satellite_bands,
            )
            if frame is not None:
                irradiance_frames.append(frame)
        df = self._outer_join(irradiance_frames, on=(schema.DATE, schema.GEOHASH5))

        # Auxiliary long-format variables: LEFT-joined onto the spine, so
        # the result rows stay exactly the irradiance (date, geohash5) pairs
        # — the Predictor cache-miss detector relies on this definition.
        for source, ids in self._aux_requests(selection):
            table_attr = _LONG_AUX_TABLES[source]
            aux = self._aux_for_source(
                source,
                query,
                table_fqn=getattr(self._t, table_attr),
                prefix=aux_column_prefix(source),
                variable_ids=ids,
                date_start=date_start,
                date_end=date_end,
            )
            if aux is not None:
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
        station_points = [
            (float(lat), float(lon))
            for lat, lon in ground[[schema.LAT, schema.LON]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ]
        satellite = self.build_satellite_features(
            selection=selection,
            date_start=date_start,
            date_end=date_end,
            points=station_points,
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

        The query point is snapped onto each source's native grid internally.
        If the warehouse has no row for this cell/date, a placeholder row
        with NaN satellite columns is returned so callers always get exactly
        one row.
        """
        gh = pygeohash.encode(lat, lon, precision=self._opts.geohash_precision)
        satellite = self.build_satellite_features(
            selection=selection,
            date_start=target_date,
            date_end=target_date,
            points=[(lat, lon)],
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
            attr = _LONG_AUX_TABLES[source]
            ids.append(getattr(TableSchemas, attr.upper()).table_id)
        if selection.modis_variable_ids:
            ids.append(TableSchemas.MODIS_OBSERVATIONS.table_id)
        return self._sat.warehouse_table_mods(ids)

    # ------------------------------------------------------------------
    # Internal — per-source snap + fetch + relabel
    # ------------------------------------------------------------------

    def _unique_query_points(
        self, points: Sequence[tuple[float, float]]
    ) -> pd.DataFrame:
        """Distinct query points keyed by their own ``geohash5``.

        Points sharing a ``geohash5`` cell collapse to one row — they map to
        the same warehouse cell, so fetching once is correct and cheaper.
        """
        precision = self._opts.geohash_precision
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for lat, lon in points:
            gh = pygeohash.encode(lat, lon, precision=precision)
            if gh in seen:
                continue
            seen.add(gh)
            rows.append(
                {
                    schema.GEOHASH5: gh,
                    schema.LAT: float(lat),
                    schema.LON: float(lon),
                }
            )
        return pd.DataFrame(rows)

    def _irradiance_for_source(
        self,
        source: Source,
        query: pd.DataFrame,
        *,
        date_start: date,
        date_end: date,
        bands: Sequence[IrradianceBand],
    ) -> Optional[pd.DataFrame]:
        """Snap → fetch → relabel one source's irradiance, or None if nothing."""
        mapping = self._snap_mapping(source, self._t.irradiance_daily, query)
        if mapping.empty:
            return None
        result = self._sat.daily_irradiance_by_geohash(
            date_start,
            date_end,
            sources=(source.value,),
            bands=bands,
            geohash5s=tuple(mapping[schema.GEOHASH5].unique()),
        )
        if result.empty:
            return None
        return self._relabel(result, mapping)

    def _aux_for_source(
        self,
        source: Source,
        query: pd.DataFrame,
        *,
        table_fqn: str,
        prefix: str,
        variable_ids: tuple[str, ...],
        date_start: date,
        date_end: date,
    ) -> Optional[pd.DataFrame]:
        """Snap → fetch → relabel one source's aux variables, or None."""
        mapping = self._snap_mapping(source, table_fqn, query)
        if mapping.empty:
            return None
        result = self._sat.long_aux_pivoted(
            table_fqn=table_fqn,
            column_prefix=prefix,
            start=date_start,
            end=date_end,
            variable_ids=variable_ids,
            geohash5s=tuple(mapping[schema.GEOHASH5].unique()),
        )
        if result.empty:
            return None
        return self._relabel(result, mapping)

    def _snap_mapping(
        self, source: Source, snapper_table_fqn: str, query: pd.DataFrame
    ) -> pd.DataFrame:
        """Map each query point's ``geohash5`` to the source-pixel ``geohash5``.

        Out-of-coverage query points (no native pixel within range) are
        dropped — they surface downstream as cache misses, never as a snap
        to a far, unrelated pixel.
        """
        snapper = self._snapper_for(source, snapper_table_fqn)
        snapped = snapper.snap_or_none(list(query[schema.LAT]), list(query[schema.LON]))
        mapping = pd.DataFrame(
            {
                schema.GEOHASH5: snapped,
                _QUERY_GEOHASH5: list(query[schema.GEOHASH5]),
            }
        )
        return mapping.dropna(subset=[schema.GEOHASH5]).reset_index(drop=True)

    @staticmethod
    def _relabel(result: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
        """Replace each row's source-pixel ``geohash5`` with the query one.

        The merge fans a source pixel out to every query cell that snapped
        to it, so two nearby query cells sharing a native pixel each get
        their own relabelled copy.
        """
        merged = result.merge(mapping, on=schema.GEOHASH5, how="inner")
        return merged.drop(columns=schema.GEOHASH5).rename(
            columns={_QUERY_GEOHASH5: schema.GEOHASH5}
        )

    def _snapper_for(self, source: Source, table_fqn: str) -> NearestPixelSnapper:
        """Return (and cache) the snapper for one ``(table, source)`` pair.

        Built from the table's own distinct cells for that source, so query
        points snap onto coordinates the warehouse actually holds.
        """
        cache_key = (table_fqn, source.value)
        cached = self._snapper_cache.get(cache_key)
        if cached is not None:
            return cached
        pixels = self._sat.native_pixels(table_fqn=table_fqn, source=source)
        snapper = NearestPixelSnapper.from_dataframe(pixels)
        self._snapper_cache[cache_key] = snapper
        return snapper

    # ------------------------------------------------------------------
    # Internal — generic helpers
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
    def _outer_join(
        frames: Sequence[pd.DataFrame], *, on: tuple[str, ...]
    ) -> pd.DataFrame:
        """Outer-join several wide frames on ``on``; empty input → empty frame."""
        non_empty = [f for f in frames if not f.empty]
        if not non_empty:
            return pd.DataFrame(columns=list(on))
        out = non_empty[0]
        for frame in non_empty[1:]:
            out = out.merge(frame, on=list(on), how="outer")
        return out

    @staticmethod
    def _merge_left(
        left: pd.DataFrame, right: pd.DataFrame, *, on: tuple[str, ...]
    ) -> pd.DataFrame:
        """LEFT-join right onto left on ``on``; tolerates an empty right side."""
        if right.empty:
            return left
        return left.merge(right, on=list(on), how="left")
