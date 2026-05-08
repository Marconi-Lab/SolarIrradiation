"""MODIS ingest job.

Sibling of :class:`BaseSatelliteJob` rather than a subclass: MODIS rows
land in their own ``modis_observations`` table (keyed on the
``(product_id, band_id)`` pair, not a flat ``variable_id``) and never
flow into ``irradiance_daily``, so the long-format/wide-format split that
``BaseSatelliteJob`` orchestrates does not apply.

The job accepts :class:`NamedLocationsPlan` or :class:`GridPlan` whose
``source`` is :attr:`Source.MODIS` and dispatches to
:class:`~susse.api_clients.modis.ModisLongFetcher` per location.
Variables in the plan are :class:`VariableSpec` entries from the catalog
where ``api_code`` carries the MODIS *product* (e.g. ``"MCD43A4"``) and
``variable_id`` follows the ``"{product}_{band}"`` convention (so the
band is recovered by stripping the ``"{api_code}_"`` prefix).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import pygeohash

from ....api_clients.modis import ModisLongFetcher, product_cadence_days
from ..base_job import BaseJob, JobResult
from ..coverage import CoverageRepository
from ..loaders import DerivedColumn, MergeLoader, MergeSpec
from ..types import (
    DateRange,
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

# When a (product, band) pair already has at least this fraction of the
# expected row count for a location, treat it as cached and skip. The
# slack accommodates composite-tile alignment: MODIS products report on
# fixed Julian-day boundaries, so a date range of N days yields between
# ``floor(N / cadence)`` and ``ceil(N / cadence) + 1`` rows depending on
# where the requested window lands.
_COVERAGE_RATIO_THRESHOLD = 0.9


def _split_variable_id(v: VariableSpec) -> tuple[str, str]:
    """Return ``(product_id, band_id)`` derived from a MODIS VariableSpec.

    The catalog encodes ``variable_id = f"{product}_{band}"`` and stores
    the product in ``api_code``. Stripping the ``"{api_code}_"`` prefix
    gives the band id verbatim, even when the band itself contains
    underscores (e.g. ``"LST_Day_1km"`` or ``"250m_16_days_NDVI"``).
    """
    prefix = f"{v.api_code}_"
    if not v.variable_id.startswith(prefix):
        raise ValueError(
            f"MODIS VariableSpec variable_id={v.variable_id!r} does not "
            f"start with api_code prefix {prefix!r}. Catalog convention "
            f"requires variable_id == f'{{api_code}}_{{band_id}}'. Update "
            f"the catalog entry in dim_variable.py."
        )
    return v.api_code, v.variable_id[len(prefix):]


class ModisJob(BaseJob):
    """Ingest job for MODIS observations from ORNL DAAC.

    Args:
        bq: BigQuery client.
        fetcher: Inject a configured :class:`ModisLongFetcher` (e.g. with
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
        fetcher: ModisLongFetcher | None = None,
        refs: TableRefs | None = None,
        geohash_precision: int = 5,
    ) -> None:
        super().__init__(bq)
        self._fetcher = fetcher or ModisLongFetcher()
        self._refs = refs or TableRefs(config=bq.config)
        self._geohash_precision = geohash_precision

    @property
    def source(self) -> Source:
        return Source.MODIS

    @property
    def name(self) -> str:
        return f"modis_ingest:{self.source.value}"

    @property
    def table_fqn(self) -> str:
        return self._refs.modis_observations

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
        product_band_pairs = tuple(_split_variable_id(v) for v in plan.variables)

        plan_geohashes = tuple(
            pygeohash.encode(loc.lat, loc.lon, precision=self._geohash_precision)
            for loc in locations
        )
        _logger.info(
            "%s: %d location(s), date %s..%s, %d (product, band) pair(s)",
            self.name, len(locations), date_range.start, date_range.end,
            len(product_band_pairs),
        )

        coverage = CoverageRepository(self._bq)
        existing_keys = coverage.existing_modis_keys(
            self.table_fqn,
            date_range=date_range,
            product_band_pairs=product_band_pairs,
            geohash5s=plan_geohashes,
        )

        rows_added = 0
        skipped_locations = 0
        api_calls = 0

        for loc in locations:
            geohash5 = pygeohash.encode(
                loc.lat, loc.lon, precision=self._geohash_precision
            )
            if self._location_fully_cached(
                geohash5,
                date_range=date_range,
                product_band_pairs=product_band_pairs,
                existing_keys=existing_keys,
            ):
                skipped_locations += 1
                continue

            df = self._fetcher.fetch_long_for_location(
                latitude=loc.lat,
                longitude=loc.lon,
                date_start=date_range.start,
                date_end=date_range.end,
                products_and_bands=product_band_pairs,
            )
            api_calls += 1
            if df.empty:
                continue
            df = self._enrich(df, loc, geohash5)
            rows_added += self._load(df)

        finished = datetime.now(timezone.utc)
        result = JobResult(
            job_name=self.name,
            plan_summary=plan.describe(),
            started_at=started,
            finished_at=finished,
            rows_added=rows_added,
            api_calls_made=api_calls,
            extra={
                "skipped_locations": skipped_locations,
                "processed_locations": len(locations) - skipped_locations,
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

    @staticmethod
    def _location_fully_cached(
        geohash5: str,
        *,
        date_range: DateRange,
        product_band_pairs: tuple[tuple[str, str], ...],
        existing_keys: set[tuple],
    ) -> bool:
        """True iff every (product, band) pair has near-complete coverage.

        See :data:`_COVERAGE_RATIO_THRESHOLD` for the threshold rationale.
        """
        for product_id, band_id in product_band_pairs:
            cadence = product_cadence_days(product_id)
            expected = max(1, date_range.n_days // cadence)
            actual = sum(
                1 for k in existing_keys
                if k[1] == geohash5 and k[2] == product_id and k[3] == band_id
            )
            if actual < expected * _COVERAGE_RATIO_THRESHOLD:
                return False
        return True

    def _enrich(
        self, df: pd.DataFrame, loc: LocationSpec, geohash5: str
    ) -> pd.DataFrame:
        out = df.copy()
        out["latitude"] = loc.lat
        out["longitude"] = loc.lon
        out["geohash5"] = geohash5
        out["source"] = self.source.value
        return out

    def _load(self, df: pd.DataFrame) -> int:
        loader = MergeLoader(
            bq=self._bq,
            table_fqn=self.table_fqn,
            spec=MergeSpec(
                schema=TableSchemas.MODIS_OBSERVATIONS,
                derived_columns=_GEOG_DERIVATION,
            ),
        )
        cols = [
            "date", "latitude", "longitude", "geohash5",
            "product_id", "band_id", "value", "source",
        ]
        return loader.load(df[cols])
