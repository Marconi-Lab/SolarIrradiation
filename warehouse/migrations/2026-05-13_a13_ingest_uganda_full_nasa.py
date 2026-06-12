"""A13 — re-ingest NASA POWER over Uganda at native resolution (2024).

Status: **optional storage cleanup — not required for the portal.**

What it does
------------
Replaces the 2024 NASA POWER rows inside the Uganda bounding box with NASA's
own *native pixels*: it deletes the existing dense ~0.1° grid for 2024 from
``nasa_daily_vars_long`` and ``irradiance_daily`` (source ``'NASA'`` only),
then re-ingests the bbox via :class:`NasaPowerRegionJob`, which stores one
row per NASA native pixel (CERES' 1° irradiance grid, MERRA-2's 0.5°×0.625°
auxiliary grid) — no densification.

Why this is optional
--------------------
The portal does **not** need this migration. Reads go through
:class:`~susse.datasets.FeatureService`, which snaps every query coordinate
onto the nearest stored cell via
:class:`~susse.warehouse_ops.snapping.NearestPixelSnapper`. The existing
~0.1° grid already covers every 2024 Uganda click. This migration only
*tidies* the warehouse: it swaps ~23M redundant 0.1°-grid rows for ~1.5M
native-pixel rows (the same NASA values, ~25× fewer copies), completing the
native-resolution storage half of the per-source-grid design.

It is therefore **not run automatically**. Apply it deliberately, when you
want the smaller table, with ``--dry-run`` first.

Scope notes
-----------
* Only the **2024** calendar year inside the Uganda bbox is touched. Sparse
  pre-2024 Uganda rows (a handful of cells from earlier migrations) and all
  out-of-bbox training-station rows are left untouched.
* The replacement is delete-then-ingest. If the ingest step fails midway,
  re-running the migration is safe: the delete is idempotent and
  :class:`NasaPowerRegionJob` MERGE-loads.
* ~130 NASA native pixels × 366 days × the full catalog ≈ ~1.5M rows. The
  cost is the fetcher's ~130 HTTP calls (one per tile × variable), ~20–30
  minutes; the BigQuery load is a single fast MERGE.

Prerequisites
-------------
No credentials — NASA POWER is public. Run from a clone of Solar_irradiation
with the susse package installed (``pip install -e .``).
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.jobs.nasa_power_region_job import (
    NasaPowerRegionJob,
)
from susse.warehouse_ops.population.types import (
    BoundingBox,
    DateRange,
    RegionPlan,
    Source,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-13_a13_ingest_uganda_full_nasa"
_log = logging.getLogger(_MIGRATION_ID)

# Uganda bounding box — matches the region gate the portal enforces.
_UGANDA_BBOX: BoundingBox = BoundingBox(
    min_lat=-1.5, max_lat=4.5, min_lon=29.5, max_lon=35.05,
)

# Calendar-year 2024. The portal's PortalConfig.default_year matches.
_DATE_RANGE: DateRange = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))

# SQL fragment scoping a statement to the 2024 Uganda bbox slice.
_BBOX_2024_SQL: str = (
    f"date BETWEEN DATE('{_DATE_RANGE.start}') AND DATE('{_DATE_RANGE.end}') "
    f"AND latitude BETWEEN {_UGANDA_BBOX.min_lat} AND {_UGANDA_BBOX.max_lat} "
    f"AND longitude BETWEEN {_UGANDA_BBOX.min_lon} AND {_UGANDA_BBOX.max_lon}"
)


def _count_rows(bq: BigQueryClient, table_fqn: str, where: str) -> int:
    """COUNT(*) for the scoped slice of a table."""
    df = bq.query(f"SELECT COUNT(*) AS n FROM `{table_fqn}` WHERE {where}")
    return int(df.iloc[0]["n"])


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    catalog = VariableCatalog.for_source(Source.NASA_POWER)
    if not catalog:
        raise RuntimeError(
            "NASA POWER catalog is empty. Check VariableCatalog registration."
        )

    long_where = _BBOX_2024_SQL
    irr_where = f"source = 'NASA' AND {_BBOX_2024_SQL}"
    n_long = _count_rows(bq, refs.nasa_daily_vars_long, long_where)
    n_irr = _count_rows(bq, refs.irradiance_daily, irr_where)

    plan = RegionPlan(
        source=Source.NASA_POWER,
        date_range=_DATE_RANGE,
        bbox=_UGANDA_BBOX,
        variables=catalog,
    )
    _log.info(
        "Uganda NASA native re-ingest: would delete %d long + %d irradiance "
        "rows for 2024, then re-ingest via %s.",
        n_long, n_irr, plan.describe(),
    )

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would delete {n_long} nasa_daily_vars_long + "
                f"{n_irr} irradiance_daily (source=NASA) rows for the 2024 "
                f"Uganda bbox, then re-ingest {len(catalog)} variables as "
                f"NASA native pixels. No deletes, no API calls."
            ),
        )

    # Delete-then-ingest. The 2024 slice spans 366 date partitions — one
    # DELETE per table stays well under BigQuery's 4000-partitions-per-DML
    # cap. A failed ingest leaves the slice empty; re-running is safe.
    _log.info("Deleting %d existing 2024 Uganda rows from nasa_daily_vars_long.", n_long)
    bq.execute_ddl(f"DELETE FROM `{refs.nasa_daily_vars_long}` WHERE {long_where}")
    _log.info("Deleting %d existing 2024 Uganda NASA rows from irradiance_daily.", n_irr)
    bq.execute_ddl(f"DELETE FROM `{refs.irradiance_daily}` WHERE {irr_where}")

    result = NasaPowerRegionJob(bq, refs=refs).run(plan)

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=result.rows_added,
        notes=(
            f"Deleted {n_long}+{n_irr} dense-grid rows; re-ingested "
            f"{result.rows_added} NASA native-pixel rows "
            f"({result.extra.get('native_pixels', 0)} pixels, "
            f"{result.extra.get('rows_added_long', 0)} long + "
            f"{result.extra.get('rows_added_irradiance', 0)} irradiance)."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Optional cleanup: replace the 2024 Uganda dense NASA grid "
                "with NASA native pixels. Not required for the portal."
            ),
        )
    )
