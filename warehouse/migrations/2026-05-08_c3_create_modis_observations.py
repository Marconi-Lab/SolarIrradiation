"""C3 — create the ``modis_observations`` table.

What it changes
---------------
Creates an empty ``solar_warehouse.modis_observations`` table from the
DDL at ``warehouse/sql/00_schema/modis_observations.sql``. The DDL uses
``CREATE TABLE IF NOT EXISTS`` so re-running this migration is a no-op.

Why
---
Phase C brings MODIS into the warehouse. Unlike the long-format tables
(NASA / CAMS / MERRA-2) which carry one (variable_id, value) per row,
MODIS observations carry both ``product_id`` and ``band_id`` because
each MODIS product has multiple bands, and the (product, band) pair is
the natural identifier of a single observation series. We give MODIS
its own table rather than overloading the long-format schema.

Storage uses each product's native composite cadence (daily, 8-day,
16-day) — values are stored at the composite end-date with no fake
daily upsampling at ingest time.

Expected effect
---------------
* No API calls.
* Single ``CREATE TABLE IF NOT EXISTS`` statement, partitioned by date,
  clustered by ``(geohash5, product_id, band_id)``.
* Pre: table does not exist.
* Post: empty table ready for C7 (dim_variable population) and
  C8 (data ingest).

Reversal
--------
``DROP TABLE solar_warehouse.modis_observations`` (only safe if no
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

_MIGRATION_ID = "2026-05-08_c3_create_modis_observations"
_log = logging.getLogger(_MIGRATION_ID)

_DDL_PATH = (
    Path(__file__).resolve().parent.parent
    / "sql"
    / "00_schema"
    / "modis_observations.sql"
)


def _table_exists(bq: BigQueryClient) -> bool:
    table_id = TableSchemas.MODIS_OBSERVATIONS.table_id
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
        "modis_observations pre-state: %s",
        "exists" if pre_exists else "missing",
    )

    if pre_exists:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes="modis_observations already exists — no-op.",
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
            "modis_observations still does not exist after CREATE TABLE — "
            "DDL execution did not behave as expected."
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=0,
        notes="modis_observations created (empty table).",
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description="Create the modis_observations table from its DDL.",
        )
    )
