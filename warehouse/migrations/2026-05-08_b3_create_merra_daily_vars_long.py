"""B3 — create the ``merra_daily_vars_long`` table.

What it changes
---------------
Creates an empty ``solar_warehouse.merra_daily_vars_long`` table from
the DDL at ``warehouse/sql/00_schema/merra_daily_vars_long.sql``. The
DDL uses ``CREATE TABLE IF NOT EXISTS`` so re-running this migration
is a no-op once the table exists.

Why
---
Phase B brings MERRA-2 into the warehouse. The schema mirrors
``nasa_daily_vars_long`` and ``cams_daily_vars_long``, with the added
note that values stored here are *daily aggregates* of MERRA-2's
hourly-native data — the aggregation is performed client-side by
``MerraStreamFetcher`` using cosine-zenith-weighting.

Expected effect
---------------
* No API calls.
* Single ``CREATE TABLE IF NOT EXISTS`` statement, partitioned by date,
  clustered by ``(geohash5, variable_id)``.
* Pre: table does not exist.
* Post: empty table ready for B7 (dim_variable population) and B8 (data ingest).

Reversal
--------
``DROP TABLE solar_warehouse.merra_daily_vars_long`` (only safe if no
later migration has loaded data into it).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs, TableSchemas

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_b3_create_merra_daily_vars_long"
_log = logging.getLogger(_MIGRATION_ID)

_DDL_PATH = (
    Path(__file__).resolve().parent.parent
    / "sql"
    / "00_schema"
    / "merra_daily_vars_long.sql"
)


def _table_exists(bq: BigQueryClient) -> bool:
    table_id = TableSchemas.MERRA_DAILY_VARS_LONG.table_id
    df = bq.query(
        f"SELECT COUNT(*) AS n "
        f"FROM `{bq.config.project_id}.{bq.config.dataset}.INFORMATION_SCHEMA.TABLES` "
        f"WHERE table_name = '{table_id}'"
    )
    return int(df.iloc[0]["n"]) > 0


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)  # noqa: F841 (parity with sister migrations)
    pre_exists = _table_exists(bq)
    _log.info(
        "merra_daily_vars_long pre-state: %s",
        "exists" if pre_exists else "missing",
    )

    if pre_exists:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes="merra_daily_vars_long already exists — no-op.",
        )

    ddl = _DDL_PATH.read_text()
    _log.info("Loaded DDL from %s (%d chars).", _DDL_PATH, len(ddl))

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=f"dry-run; would execute DDL from {_DDL_PATH.name}.",
        )

    bq.execute_ddl(ddl)
    if not _table_exists(bq):
        raise RuntimeError(
            "merra_daily_vars_long still does not exist after CREATE TABLE — "
            "DDL execution did not behave as expected."
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=0,
        notes="merra_daily_vars_long created (empty table).",
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description="Create the merra_daily_vars_long table from its DDL.",
        )
    )
