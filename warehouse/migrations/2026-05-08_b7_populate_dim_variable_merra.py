"""B7 — extend ``dim_variable`` with the 4 MERRA-2 catalog entries.

What it changes
---------------
Re-runs ``populate_dim_variable`` against the full catalog. Because the
catalog now includes 4 MERRA-2 ``VariableSpec``s (added in B2), the MERGE
inserts those new rows. NASA POWER and CAMS rows are already there and
are no-ops on the merge keys ``(variable_id, source)``.

Why
---
Phase B mirrors Phase A1 for the MERRA-2 source: every variable that
will land in the long table needs a corresponding ``dim_variable`` row
documenting its unit, valid range, and spatial resolution. Without this,
downstream tools can't introspect the catalog from BigQuery.

Expected effect
---------------
* No API calls.
* Pre: ``dim_variable`` has 42 rows (32 NASA + 9 CAMS + 1 NASA SZA).
* Post: ``dim_variable`` has 46 rows (42 + 4 MERRA-2).
* Idempotent: re-running with the catalog already populated is a no-op.

Reversal
--------
``DELETE FROM dim_variable WHERE source = 'MERRA2'`` removes the four
MERRA-2 rows. Safe before B8 has loaded any data; once B8 has populated
``merra_daily_vars_long`` you'd want to keep the catalog rows so the
schema stays self-documenting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population.dim_variable import (
    VariableCatalog,
    populate_dim_variable,
)
from susse.warehouse_ops.population.types import Source

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_b7_populate_dim_variable_merra"
_log = logging.getLogger(_MIGRATION_ID)


def _read_state(bq: BigQueryClient, refs: TableRefs) -> dict:
    df = bq.query(
        f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNTIF(source = 'NASA') AS n_nasa,
          COUNTIF(source = 'CAMS') AS n_cams,
          COUNTIF(source = 'MERRA2') AS n_merra
        FROM `{refs.dim_variable}`
        """
    )
    return df.iloc[0].to_dict()


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    catalog = VariableCatalog.all_variables()
    n_merra_in_catalog = sum(1 for v in catalog if v.source is Source.MERRA_2)
    _log.info(
        "Catalog has %d total variables, %d of them MERRA-2.",
        len(catalog), n_merra_in_catalog,
    )

    pre = _read_state(bq, refs)
    _log.info(
        "Pre: dim_variable n_rows=%s NASA=%s CAMS=%s MERRA2=%s",
        pre["n_rows"], pre["n_nasa"], pre["n_cams"], pre["n_merra"],
    )

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would MERGE {len(catalog)} catalog rows; expect "
                f"~{n_merra_in_catalog} new MERRA-2 rows."
            ),
        )

    written = populate_dim_variable(
        bq, table_fqn=refs.dim_variable, variables=catalog
    )
    post = _read_state(bq, refs)
    _log.info(
        "Post: dim_variable n_rows=%s NASA=%s CAMS=%s MERRA2=%s",
        post["n_rows"], post["n_nasa"], post["n_cams"], post["n_merra"],
    )

    if post["n_merra"] != n_merra_in_catalog:
        raise RuntimeError(
            f"Expected {n_merra_in_catalog} MERRA-2 rows in dim_variable "
            f"after MERGE; got {post['n_merra']}. Investigate."
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=written,
        notes=(
            f"dim_variable: pre n_rows={pre['n_rows']} → post "
            f"n_rows={post['n_rows']}; MERRA2 rows {pre['n_merra']} → "
            f"{post['n_merra']}."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Extend dim_variable with the 4 MERRA-2 catalog entries."
            ),
        )
    )
