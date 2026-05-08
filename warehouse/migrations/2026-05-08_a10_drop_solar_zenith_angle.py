"""A10 — remove ``solar_zenith_angle`` from the catalog and warehouse.

What it changes
---------------
``DELETE FROM dim_variable WHERE variable_id = 'solar_zenith_angle'``.
The catalog (``VariableCatalog.NASA_POWER_VARIABLES``) has been edited
in the same commit to drop the corresponding ``VariableSpec``.

Why
---
The variable was added to the catalog in A2 because the api_client
exposes ``SOLAR_ZENITH_ANGLE = "SZA"``. Empirically, however, NASA
POWER does *not* serve SZA at daily resolution — every request returns
the ``-999`` fill value, which the loader filters out. The result:
``nasa_daily_vars_long`` ends up with zero rows for SZA, and the
per-station coverage check treats SZA as always-missing for every
day. That triggered a full re-fetch of every station on every A6 run
(none of the existing rows could ever satisfy the SZA part of the
plan), wasting NASA quota and ingest time.

There is no SZA data to delete from ``nasa_daily_vars_long`` (the
loader never wrote any). Only ``dim_variable`` needs cleaning.

Expected effect
---------------
* No API calls.
* Single ``DELETE`` statement.
* Pre: ``dim_variable`` has 1 row with ``variable_id='solar_zenith_angle'``.
* Post: 0 rows.
* Idempotent: re-running with the row already absent is a no-op.

Reversal
--------
Re-add the ``VariableSpec`` to ``VariableCatalog.NASA_POWER_VARIABLES``
and re-run ``populate_dim_variable``. Only do this once you've verified
NASA POWER actually serves daily SZA values for the locations you care
about.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_a10_drop_solar_zenith_angle"
_log = logging.getLogger(_MIGRATION_ID)

_VARIABLE_ID = "solar_zenith_angle"


def _count_rows(bq: BigQueryClient, refs: TableRefs) -> int:
    df = bq.query(
        f"SELECT COUNT(*) AS n FROM `{refs.dim_variable}` "
        f"WHERE variable_id = '{_VARIABLE_ID}'"
    )
    return int(df.iloc[0]["n"])


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    pre = _count_rows(bq, refs)
    _log.info(
        "Pre: %d row(s) in dim_variable with variable_id='%s'.",
        pre, _VARIABLE_ID,
    )
    if pre == 0:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes="No solar_zenith_angle row to delete — already clean.",
        )
    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=f"dry-run; would DELETE {pre} row(s).",
        )
    bq.execute_ddl(
        f"DELETE FROM `{refs.dim_variable}` "
        f"WHERE variable_id = '{_VARIABLE_ID}'"
    )
    post = _count_rows(bq, refs)
    if post != 0:
        raise RuntimeError(
            f"DELETE left {post} row(s) with variable_id='{_VARIABLE_ID}'. "
            f"Investigate before re-running."
        )
    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=pre,
        notes=f"Deleted {pre} '{_VARIABLE_ID}' row(s) from dim_variable.",
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Delete solar_zenith_angle from dim_variable (NASA POWER "
                "doesn't serve it daily; entry triggered spurious re-fetches)."
            ),
        )
    )
