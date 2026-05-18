"""Region-shaped NASA POWER ingest job.

Sibling of :class:`BaseSatelliteJob` (per-point) and :class:`MerraRegionJob`:
NASA POWER's ``/regional`` endpoint returns one HTTP call per (tile, variable)
covering a whole bounding box, instead of the per-point pattern. This job
wraps :class:`NASAPowerRegionalFetcher`, then routes the returned rows into
the wide ``irradiance_daily`` table (GHI/DHI/DNI) and the long
``nasa_daily_vars_long`` table (auxiliary variables).

Crucially, the job stores **NASA's own native pixel centres verbatim** — no
densification onto a finer grid. NASA POWER irradiance comes off CERES' 1°
grid and its auxiliary variables off MERRA-2's 0.5°×0.625° grid; both are
stored at those native resolutions. Reads snap arbitrary query coordinates
onto these pixels via :class:`susse.warehouse_ops.snapping.NearestPixelSnapper`.

For a one-year Uganda ingest the whole result is ~1.5M rows (~130 native
pixels × 366 days × ~32 variables), so the BigQuery load is a single fast
MERGE — the cost is entirely the ~130 HTTP calls the fetcher issues.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import pygeohash

from ....api_clients.NASA_Power import NASAPowerRegionalFetcher
from ...io.bq import BigQueryClient
from ...io.config import TableRefs, TableSchemas
from ..base_job import BaseJob, JobResult
from ..dim_variable import VariableCatalog
from ..loaders import DerivedColumn, MergeLoader, MergeSpec
from ..types import FetchPlan, RegionPlan, Source
from ..validators import validate_long_format

_logger = logging.getLogger(__name__)

# Server-side derivation: GEOGRAPHY column built from staging lat/lon.
# Same convention as the per-point satellite jobs and MerraRegionJob.
_GEOG_DERIVATION: tuple[DerivedColumn, ...] = (
    DerivedColumn(name="geog", sql_expr="ST_GEOGPOINT(longitude, latitude)"),
)

# Long-format variable_id → wide irradiance_daily column name.
_IRRADIANCE_COLUMN_BY_VARIABLE: dict[str, str] = {
    "ghi": "ghi_kwh_m2_day",
    "dhi": "dhi_kwh_m2_day",
    "dni": "dni_kwh_m2_day",
}


class NasaPowerRegionJob(BaseJob):
    """Region-shaped NASA POWER ingest, storing native pixels verbatim.

    Args:
        bq: BigQuery client.
        fetcher: Inject a configured :class:`NASAPowerRegionalFetcher` (e.g.
            a stub for tests). Defaults to a fresh fetcher.
        refs: Override the table-ref resolver. Defaults to one built from
            ``bq.config``.
        geohash_precision: Precision for the ``geohash5`` column. Must match
            the rest of the warehouse (5).
    """

    def __init__(
        self,
        bq: BigQueryClient,
        *,
        fetcher: NASAPowerRegionalFetcher | None = None,
        refs: TableRefs | None = None,
        geohash_precision: int = 5,
    ) -> None:
        super().__init__(bq)
        self._fetcher = fetcher or NASAPowerRegionalFetcher()
        self._refs = refs or TableRefs(config=bq.config)
        self._geohash_precision = geohash_precision

    @property
    def source(self) -> Source:
        return Source.NASA_POWER

    @property
    def name(self) -> str:
        return f"nasa_power_region_ingest:{self.source.value}"

    def run(self, plan: FetchPlan) -> JobResult:
        if not isinstance(plan, RegionPlan):
            raise TypeError(
                f"{self.name} accepts RegionPlan, got {type(plan).__name__}. "
                f"Wrap the bbox + variables in a RegionPlan."
            )
        if plan.source is not self.source:
            raise ValueError(
                f"{self.name} requires plan.source == {self.source}, "
                f"got {plan.source}."
            )

        started = self._start_time()
        variables = tuple(plan.variables)
        api_to_var = {v.api_code: v.variable_id for v in variables}
        _logger.info(
            "%s: bbox lat[%.3f..%.3f] lon[%.3f..%.3f], dates %s..%s, %d var(s)",
            self.name,
            plan.bbox.min_lat,
            plan.bbox.max_lat,
            plan.bbox.min_lon,
            plan.bbox.max_lon,
            plan.date_range.start,
            plan.date_range.end,
            len(variables),
        )

        df = self._fetcher.fetch_region_long(
            min_lat=plan.bbox.min_lat,
            max_lat=plan.bbox.max_lat,
            min_lon=plan.bbox.min_lon,
            max_lon=plan.bbox.max_lon,
            date_start=plan.date_range.start,
            date_end=plan.date_range.end,
            api_codes=tuple(v.api_code for v in variables),
        )

        rows_added_long = 0
        rows_added_irr = 0
        n_pixels = 0
        if not df.empty:
            enriched = self._enrich(df, api_to_var)
            n_pixels = int(enriched["geohash5"].nunique())
            is_irradiance = enriched["variable_id"].isin(
                VariableCatalog.IRRADIANCE_VARIABLE_IDS
            )
            irradiance_long = enriched[is_irradiance]
            aux_long = enriched[~is_irradiance]
            if not irradiance_long.empty:
                rows_added_irr = self._load_irradiance(
                    _irradiance_long_to_wide(irradiance_long)
                )
            if not aux_long.empty:
                rows_added_long = self._load_long(aux_long)

        finished = datetime.now(timezone.utc)
        result = JobResult(
            job_name=self.name,
            plan_summary=plan.describe(),
            started_at=started,
            finished_at=finished,
            rows_added=rows_added_long + rows_added_irr,
            # The fetcher fans out internally; one fetch_region_long call.
            api_calls_made=1,
            extra={
                "rows_added_long": rows_added_long,
                "rows_added_irradiance": rows_added_irr,
                "native_pixels": n_pixels,
                "fetched_rows": int(len(df)),
            },
        )
        _logger.info("%s | %s", self.name, result.summary())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich(
        self, df: pd.DataFrame, api_to_var: dict[str, str]
    ) -> pd.DataFrame:
        """Map api_code → variable_id and attach geohash5 + source columns.

        The fetcher emits ``variable_id == api_code``; the warehouse uses the
        catalog-defined ``variable_id``. ``geohash5`` is computed once per
        distinct native pixel (~130 of them) and merged in, rather than once
        per row.
        """
        out = df.copy()
        out["variable_id"] = out["variable_id"].map(api_to_var)
        out = out.dropna(subset=["variable_id"])
        if out.empty:
            return out

        pixels = out[["latitude", "longitude"]].drop_duplicates()
        pixels["geohash5"] = [
            pygeohash.encode(lat, lon, precision=self._geohash_precision)
            for lat, lon in zip(pixels["latitude"], pixels["longitude"])
        ]
        out = out.merge(pixels, on=["latitude", "longitude"], how="left")
        out["source"] = self.source.value
        return out

    def _load_long(self, df: pd.DataFrame) -> int:
        validate_long_format(df, context=f"{self.name} long")
        loader = MergeLoader(
            bq=self._bq,
            table_fqn=self._refs.nasa_daily_vars_long,
            spec=MergeSpec(
                schema=TableSchemas.NASA_DAILY_VARS_LONG,
                derived_columns=_GEOG_DERIVATION,
            ),
        )
        return loader.load(
            df[
                [
                    "date",
                    "latitude",
                    "longitude",
                    "geohash5",
                    "variable_id",
                    "value",
                    "source",
                ]
            ]
        )

    def _load_irradiance(self, df: pd.DataFrame) -> int:
        loader = MergeLoader(
            bq=self._bq,
            table_fqn=self._refs.irradiance_daily,
            spec=MergeSpec(
                schema=TableSchemas.IRRADIANCE_DAILY,
                derived_columns=_GEOG_DERIVATION,
            ),
        )
        return loader.load(
            df[
                [
                    "date",
                    "latitude",
                    "longitude",
                    "geohash5",
                    "source",
                    "ghi_kwh_m2_day",
                    "dhi_kwh_m2_day",
                    "dni_kwh_m2_day",
                    "reliability",
                ]
            ]
        )


def _irradiance_long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format irradiance rows into the wide ``irradiance_daily`` shape.

    Input rows carry ``variable_id`` ∈ {``ghi``, ``dhi``, ``dni``}; the output
    has one row per (date, pixel) with named ``*_kwh_m2_day`` columns. Bands
    absent from the input become NULL, as does ``reliability`` (NASA POWER
    publishes no reliability series).
    """
    wide = long_df.pivot_table(
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
