"""A2 — fix variable_id typos in the warehouse and refresh the catalog.

What it changes
---------------
1. Renames ``variable_id = 'arimass'`` rows to ``'airmass'`` in
   ``nasa_daily_vars_long`` (~718,092 rows).
2. Renames the corresponding ``dim_variable`` row from ``'arimass'`` to
   ``'airmass'`` (1 row).
3. Re-MERGES the full ``VariableCatalog`` into ``dim_variable`` so the
   newly-added ``solar_zenith_angle`` entry and any other catalog edits
   land. After this the table holds 42 rows (33 NASA + 9 CAMS).

Why
---
The legacy ingest path used the misspelt ``arimass`` (introduced upstream
when the api_client enum had ``ARIMASS = "AIRMASS"``). That enum was
fixed in code as part of A2; this migration fixes the warehouse data so
the spelling now matches everywhere.

Also adds a previously-missing variable: ``solar_zenith_angle`` (api code
``SZA``) which was already exposed by the api_client but not present in
the catalog.

Expected effect
---------------
* No API calls.
* Two ``UPDATE`` statements + one MERGE (via ``populate_dim_variable``).
* Pre: ``nasa_daily_vars_long`` has rows with ``variable_id = 'arimass'``;
  ``dim_variable`` has 41 rows (no ``solar_zenith_angle``).
* Post: zero ``arimass`` rows in either table; ``airmass`` rows present;
  ``dim_variable`` has 42 rows including ``solar_zenith_angle``.
* Idempotent: re-running with no ``arimass`` rows present is a no-op.

Reversal
--------
Reversible by running an inverse UPDATE (``airmass → arimass``) but the
api_client and catalog have moved on. Only roll back if you also revert
the code changes from the same commit.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_a2_fix_typos"
_log = logging.getLogger(_MIGRATION_ID)

_OLD_VAR_ID = "arimass"
_NEW_VAR_ID = "airmass"


def _count_rows_with_variable_id(
    bq: BigQueryClient, table_fqn: str, variable_id: str
) -> int:
    """Count rows where ``variable_id`` equals the given value."""
    df = bq.query(
        f"SELECT COUNT(*) AS n FROM `{table_fqn}` "
        f"WHERE variable_id = '{variable_id}'"
    )
    return int(df.iloc[0]["n"])


def _rename_variable_id(
    bq: BigQueryClient, table_fqn: str, old: str, new: str, dry_run: bool
) -> int:
    """UPDATE ``variable_id`` from ``old`` to ``new`` and return rows changed."""
    pre = _count_rows_with_variable_id(bq, table_fqn, old)
    if pre == 0:
        _log.info(
            "%s: zero rows with variable_id='%s' — nothing to rename.",
            table_fqn, old,
        )
        return 0
    _log.info(
        "%s: %d rows with variable_id='%s' will be renamed to '%s'.",
        table_fqn, pre, old, new,
    )
    if dry_run:
        return pre
    bq.execute_ddl(
        f"UPDATE `{table_fqn}` "
        f"SET variable_id = '{new}' "
        f"WHERE variable_id = '{old}'"
    )
    post_old = _count_rows_with_variable_id(bq, table_fqn, old)
    post_new = _count_rows_with_variable_id(bq, table_fqn, new)
    if post_old != 0:
        raise RuntimeError(
            f"{table_fqn}: UPDATE left {post_old} rows still labelled '{old}'. "
            f"Aborting; investigate before re-running."
        )
    _log.info(
        "%s: rename complete — '%s'=0 rows, '%s'=%d rows.",
        table_fqn, old, new, post_new,
    )
    return pre


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)

    # 1. Rename arimass → airmass in the long table (the bulk of the rows).
    long_renamed = _rename_variable_id(
        bq, refs.nasa_daily_vars_long, _OLD_VAR_ID, _NEW_VAR_ID, dry_run
    )

    # 2. Rename arimass → airmass in dim_variable (1 row, but for consistency).
    dim_renamed = _rename_variable_id(
        bq, refs.dim_variable, _OLD_VAR_ID, _NEW_VAR_ID, dry_run
    )

    # 3. Re-MERGE the full catalog so the new solar_zenith_angle row lands
    #    and any updated metadata (display names, descriptions) refreshes.
    catalog = VariableCatalog.all_variables()
    pre_dim_n = bq.query(
        f"SELECT COUNT(*) AS n FROM `{refs.dim_variable}`"
    ).iloc[0]["n"]
    _log.info(
        "Catalog has %d variables; dim_variable currently has %d rows.",
        len(catalog), pre_dim_n,
    )

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=long_renamed + dim_renamed,
            notes=(
                f"dry-run; would rename {long_renamed} long-table rows + "
                f"{dim_renamed} dim_variable rows, then MERGE {len(catalog)} "
                f"catalog rows."
            ),
        )

    written = populate_dim_variable(
        bq, table_fqn=refs.dim_variable, variables=catalog
    )
    post_dim_n = int(
        bq.query(f"SELECT COUNT(*) AS n FROM `{refs.dim_variable}`").iloc[0]["n"]
    )
    _log.info(
        "dim_variable n_rows: pre=%s, post=%s (catalog merged %d rows).",
        pre_dim_n, post_dim_n, written,
    )

    if post_dim_n < len(catalog):
        raise RuntimeError(
            f"dim_variable has {post_dim_n} rows after MERGE, expected "
            f">= {len(catalog)}. Investigate."
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=long_renamed + dim_renamed + written,
        notes=(
            f"renamed {long_renamed} '{_OLD_VAR_ID}'→'{_NEW_VAR_ID}' rows in "
            f"nasa_daily_vars_long, {dim_renamed} in dim_variable; "
            f"dim_variable refreshed to {post_dim_n} rows."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description="Rename arimass→airmass in warehouse data; refresh dim_variable.",
        )
    )
