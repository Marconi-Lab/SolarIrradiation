"""Ground-truth ingest job: parse → validate → MERGE raw → curate → MERGE curated."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from ..adapters import GroundSourceAdapter
from ..base_job import BaseJob, JobResult
from ..coverage import CoverageRepository
from ..curation import CurationOptions, curate_ground
from ..loaders import DerivedColumn, MergeLoader, MergeSpec
from ..types import FetchPlan, GroundFilePlan
from ..validators import validate_ground_curated, validate_ground_raw
from ...io.bq import BigQueryClient
from ...io.config import TableRefs, TableSchemas

_logger = logging.getLogger(__name__)


# The curated table holds a GEOGRAPHY column built server-side from lat/lon.
_GEOG_DERIVATION: tuple[DerivedColumn, ...] = (
    DerivedColumn(name="geog", sql_expr="ST_GEOGPOINT(lon, lat)"),
)


class GroundIngestJob(BaseJob):
    """Ingest a ground-measurement file end-to-end.

    Pipeline:

    1. Adapter parses the file into the standard raw schema.
    2. Schema validation against ``ground_measurements_raw``.
    3. Coverage check: skip rows whose ``(datetime, location)`` is already
       in the raw table (rate-limit-aware idempotency for the disk-read).
    4. MERGE-load raw rows into ``ground_measurements_raw``.
    5. Curate the new rows (geohash, units, provenance).
    6. MERGE-load curated rows into ``ground_measurements`` (with ``geog``
       derived server-side).

    Re-running on the same file is a no-op (returns rows_added=0).
    """

    def __init__(
        self,
        bq: BigQueryClient,
        *,
        adapter: GroundSourceAdapter,
        refs: TableRefs | None = None,
        curation_options: CurationOptions | None = None,
    ) -> None:
        super().__init__(bq)
        self._adapter = adapter
        self._refs = refs or TableRefs(config=bq.config)
        self._curation_options = curation_options or CurationOptions()

    @property
    def name(self) -> str:
        return f"ground_ingest:{self._adapter.adapter_id}"

    def run(self, plan: FetchPlan) -> JobResult:
        if not isinstance(plan, GroundFilePlan):
            raise TypeError(
                f"{self.name} accepts GroundFilePlan, got {type(plan).__name__}."
            )
        if plan.adapter_id != self._adapter.adapter_id:
            raise ValueError(
                f"Plan declares adapter '{plan.adapter_id}' but this job is "
                f"configured with '{self._adapter.adapter_id}'. "
                f"Construct the GroundIngestJob with the matching adapter."
            )
        started = self._start_time()

        raw = self._adapter.parse(plan.file_path)
        validate_ground_raw(raw, context=f"{self.name}.parse")

        coverage = CoverageRepository(self._bq)
        locations_in_file = sorted(raw["location"].unique().tolist())
        existing_raw: set[tuple] = set()
        for loc in locations_in_file:
            existing_raw |= coverage.existing_ground_raw_keys(
                self._refs.ground_measurements_raw, location=loc,
            )

        # Filter to rows NOT yet in raw.
        raw_keys = list(zip(raw["datetime"], raw["location"]))
        keep_mask = [k not in existing_raw for k in raw_keys]
        new_raw = raw[keep_mask].reset_index(drop=True)
        skipped = len(raw) - len(new_raw)

        rows_added_raw = 0
        rows_added_curated = 0

        if not new_raw.empty:
            rows_added_raw = self._load_raw(new_raw)
            curated = curate_ground(new_raw, self._curation_options)
            validate_ground_curated(curated, context=f"{self.name}.curated")
            rows_added_curated = self._load_curated(curated)
        else:
            _logger.info(
                "%s: all %d rows already present in raw table; nothing to do.",
                self.name, len(raw),
            )

        finished = datetime.now(timezone.utc)
        result = JobResult(
            job_name=self.name,
            plan_summary=plan.describe(),
            started_at=started,
            finished_at=finished,
            rows_added=rows_added_raw + rows_added_curated,
            rows_already_cached=skipped,
            extra={
                "raw_rows_in_file": len(raw),
                "raw_rows_added": rows_added_raw,
                "curated_rows_added": rows_added_curated,
                "locations": len(locations_in_file),
            },
        )
        _logger.info("%s | %s", self.name, result.summary())
        return result

    def _load_raw(self, df: pd.DataFrame) -> int:
        loader = MergeLoader(
            bq=self._bq,
            table_fqn=self._refs.ground_measurements_raw,
            spec=MergeSpec(schema=TableSchemas.GROUND_MEASUREMENTS_RAW),
        )
        return loader.load(df)

    def _load_curated(self, df: pd.DataFrame) -> int:
        loader = MergeLoader(
            bq=self._bq,
            table_fqn=self._refs.ground_measurements,
            spec=MergeSpec(
                schema=TableSchemas.GROUND_MEASUREMENTS,
                derived_columns=_GEOG_DERIVATION,
            ),
        )
        return loader.load(df)
