"""A7 — drop the legacy external CSV-backed tables.

What it changes
---------------
Drops two external tables that were used by the old ingest path:

* ``solar_warehouse.cams_daily_ext`` — pointer at
  ``gs://solar_irradiation_db/raw/cams/radiation/uganda/year=2024/*.csv``.
  Already consumed by A4 to populate ``cams_daily_vars_long``; no
  remaining purpose.
* ``solar_warehouse.nasa_daily_ext`` — pointer at the analogous NASA
  CSV staging path. Equivalent data already lives in
  ``nasa_daily_vars_long`` and ``irradiance_daily``.

Why
---
With Phase A complete, both relics serve no purpose. Keeping them
around is a smell — future readers might assume they're authoritative
or active.

Expected effect
---------------
* No API calls.
* Two ``DROP EXTERNAL TABLE`` statements.
* Idempotent via ``DROP TABLE IF EXISTS``.

Reversal
--------
Each external table was created by a CSV pointer in
``warehouse/sql/`` (or via legacy notebooks). Re-create from those if
ever needed; the underlying GCS CSVs remain untouched by this drop.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import WarehouseConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_a7_drop_external_relics"
_log = logging.getLogger(_MIGRATION_ID)

_RELIC_TABLE_IDS: tuple[str, ...] = ("cams_daily_ext", "nasa_daily_ext")


def _table_exists(bq: BigQueryClient, config: WarehouseConfig, table_id: str) -> bool:
    df = bq.query(
        f"SELECT COUNT(*) AS n "
        f"FROM `{config.project_id}.{config.dataset}.INFORMATION_SCHEMA.TABLES` "
        f"WHERE table_name = '{table_id}'"
    )
    return int(df.iloc[0]["n"]) > 0


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    config = bq.config
    drops = []
    for table_id in _RELIC_TABLE_IDS:
        exists = _table_exists(bq, config, table_id)
        _log.info("%s: %s", table_id, "exists" if exists else "missing")
        if exists:
            drops.append(table_id)

    if not drops:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes="all relic tables already absent — no-op.",
        )

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=f"dry-run; would drop: {drops}.",
        )

    for table_id in drops:
        fqn = config.fqn(table_id)
        bq.execute_ddl(f"DROP TABLE IF EXISTS `{fqn}`;")
        _log.info("Dropped %s.", fqn)

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=len(drops),
        notes=f"dropped {len(drops)} relic table(s): {drops}",
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description="Drop the legacy external CSV tables (cams_daily_ext, nasa_daily_ext).",
        )
    )
