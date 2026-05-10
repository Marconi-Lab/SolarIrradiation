"""MERRA-2 region-shaped ingest job.

Sibling of :class:`BaseSatelliteJob` rather than a subclass: MERRA-2 is
fetched via region (bbox) OPeNDAP calls — one call per (date, variable)
covering every location in the plan — instead of the per-(point, date)
pattern that NASA POWER and CAMS use. The inner loop of
``BaseSatelliteJob`` is therefore not the right shape for MERRA-2; we
fan out at the fetcher boundary instead.

See :mod:`susse.api_clients.merra_2.merra_daily_fetcher` for the bbox
mechanics and the speedup rationale (~100× over per-point loops for
grid-sized requests).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import pygeohash

from ....api_clients.merra_2 import MerraDailyFetcher
from ..base_job import BaseJob, JobResult
from ..coverage import CoverageRepository
from ..loaders import DerivedColumn, MergeLoader, MergeSpec
from ..types import (
    FetchPlan,
    GridPlan,
    LocationSpec,
    NamedLocationsPlan,
    Source,
    VariableSpec,
)
from ...io.bq import BigQueryClient
from ...io.config import TableRefs, TableSchemas

_logger = logging.getLogger(__name__)

# Server-side derivation: GEOGRAPHY column built from staging lat/lon.
# Same convention as the long-format satellite jobs.
_GEOG_DERIVATION: tuple[DerivedColumn, ...] = (
    DerivedColumn(name="geog", sql_expr="ST_GEOGPOINT(longitude, latitude)"),
)


class MerraRegionJob(BaseJob):
    """Ingest job for MERRA-2 reanalysis using region-shaped fetches.

    Args:
        bq: BigQuery client.
        fetcher: Inject a configured :class:`MerraDailyFetcher` (e.g. with
            a custom ``max_workers`` or a stub for tests). Defaults to a
            fresh fetcher with the package default concurrency.
        refs: Override the table-ref resolver. Defaults to one built from
            ``bq.config``.
        geohash_precision: Precision for the ``geohash5`` column. Must
            match what the rest of the warehouse uses (5).
    """

    def __init__(
        self,
        bq: BigQueryClient,
        *,
        fetcher: MerraDailyFetcher | None = None,
        refs: TableRefs | None = None,
        geohash_precision: int = 5,
    ) -> None:
        super().__init__(bq)
        self._fetcher = fetcher or MerraDailyFetcher()
        self._refs = refs or TableRefs(config=bq.config)
        self._geohash_precision = geohash_precision

    @property
    def source(self) -> Source:
        return Source.MERRA_2

    @property
    def name(self) -> str:
        return f"merra_region_ingest:{self.source.value}"

    @property
    def table_fqn(self) -> str:
        return self._refs.merra_daily_vars_long

    def run(self, plan: FetchPlan) -> JobResult:
        if not isinstance(plan, (NamedLocationsPlan, GridPlan)):
            raise TypeError(
                f"{self.name} accepts NamedLocationsPlan or GridPlan, "
                f"got {type(plan).__name__}."
            )
        if plan.source is not self.source:
            raise ValueError(
                f"{self.name} requires plan.source == {self.source}, "
                f"got {plan.source}."
            )

        started = self._start_time()
        locations = self._resolve_locations(plan)
        date_range = plan.date_range
        variables = tuple(plan.variables)
        api_codes = tuple(v.api_code for v in variables)
        api_to_var = {v.api_code: v.variable_id for v in variables}

        location_geohashes = tuple(
            pygeohash.encode(loc.lat, loc.lon, precision=self._geohash_precision)
            for loc in locations
        )
        _logger.info(
            "%s: %d location(s), date %s..%s, %d var(s)",
            self.name, len(locations), date_range.start, date_range.end,
            len(variables),
        )

        coverage = CoverageRepository(self._bq)
        existing_keys = coverage.existing_long_keys(
            self.table_fqn,
            date_range=date_range,
            source=self.source,
            variable_ids=tuple(v.variable_id for v in variables),
            geohash5s=location_geohashes,
        )

        df = self._fetcher.fetch_region(
            points=tuple((loc.lat, loc.lon) for loc in locations),
            date_start=date_range.start,
            date_end=date_range.end,
            api_codes=api_codes,
        )
        api_calls = 1  # one fetch_region call covers all (date, var) tasks

        rows_added = 0
        if not df.empty:
            df = self._enrich(df, locations, api_to_var)
            df = self._drop_already_cached(df, existing_keys)
            if not df.empty:
                rows_added = self._load(df)

        finished = datetime.now(timezone.utc)
        result = JobResult(
            job_name=self.name,
            plan_summary=plan.describe(),
            started_at=started,
            finished_at=finished,
            rows_added=rows_added,
            api_calls_made=api_calls,
            extra={
                "locations": len(locations),
                "fetched_rows": int(len(df)) + (rows_added if df.empty else 0),
            },
        )
        _logger.info("%s | %s", self.name, result.summary())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_locations(
        plan: NamedLocationsPlan | GridPlan,
    ) -> list[LocationSpec]:
        if isinstance(plan, NamedLocationsPlan):
            return list(plan.locations)
        return list(plan.grid.iter_locations())

    def _enrich(
        self,
        df: pd.DataFrame,
        locations: list[LocationSpec],
        api_to_var: dict[str, str],
    ) -> pd.DataFrame:
        """Map api_code → variable_id, attach geohash5 and source columns.

        The fetcher returns variable_id == api_code; the warehouse uses
        the catalog-defined variable_id. Geohash5 is derived from the
        plan's locations (one geohash per (lat, lon) pair) so we don't
        recompute per-row.
        """
        out = df.copy()
        out["variable_id"] = out["variable_id"].map(api_to_var)
        out = out.dropna(subset=["variable_id"])
        if out.empty:
            return out

        # Per-(lat, lon) geohash lookup, vectorised via merge.
        gh_lookup = pd.DataFrame(
            [
                {
                    "latitude": loc.lat,
                    "longitude": loc.lon,
                    "geohash5": pygeohash.encode(
                        loc.lat, loc.lon, precision=self._geohash_precision
                    ),
                }
                for loc in locations
            ]
        )
        out = out.merge(gh_lookup, on=["latitude", "longitude"], how="left")
        if out["geohash5"].isna().any():
            # Should be impossible: the fetcher emits rows whose lat/lon
            # come straight from the input ``points``, which mirror
            # ``locations``. Surface the assumption explicitly.
            raise RuntimeError(
                "MerraRegionJob._enrich: fetched rows contain (lat, lon) "
                "values not present in the plan's locations. The fetcher "
                "and the plan are out of sync."
            )
        out["source"] = self.source.value
        return out

    @staticmethod
    def _drop_already_cached(
        df: pd.DataFrame, existing_keys: set[tuple]
    ) -> pd.DataFrame:
        """Filter out rows whose (date, geohash5, variable_id) is already
        in the warehouse.

        The MergeLoader is idempotent on the merge keys, so this filter
        is purely an optimisation: it spares us the upload + MERGE round
        trip on full re-runs.
        """
        if not existing_keys or df.empty:
            return df
        keys = pd.Series(
            list(zip(df["date"], df["geohash5"], df["variable_id"])),
            index=df.index,
        )
        return df[~keys.isin(existing_keys)]

    def _load(self, df: pd.DataFrame) -> int:
        loader = MergeLoader(
            bq=self._bq,
            table_fqn=self.table_fqn,
            spec=MergeSpec(
                schema=TableSchemas.MERRA_DAILY_VARS_LONG,
                derived_columns=_GEOG_DERIVATION,
            ),
        )
        cols = [
            "date", "latitude", "longitude", "geohash5",
            "variable_id", "value", "source",
        ]
        return loader.load(df[cols])
