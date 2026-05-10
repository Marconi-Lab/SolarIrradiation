"""A11 — annotate ``dim_variable`` with the ``physical_storage`` column.

What it changes
---------------
1. Adds a new ``physical_storage STRING`` column to
   ``solar_warehouse.dim_variable``.
2. Re-populates the table from :class:`VariableCatalog` so every row
   carries the new column with its tag (``long_format`` /
   ``irradiance_wide`` / ``modis_observations``).

Why
---
Before this migration, the catalog enumerated variables without
distinguishing where they physically live in the warehouse. The
``ghi`` / ``dhi`` / ``dni`` entries (NASA + CAMS) were claimed to
exist alongside the long-format aux variables, but the underlying data
only ever lived in ``irradiance_daily``. ``FeatureSelection`` would
silently accept ``nasa_variable_ids=("ghi",)`` and fetch a 100%-NaN
column. The new tag closes that gap: ``FeatureSelection`` validation
now checks that each requested ``variable_id`` matches the storage the
field reads from. See ``feature_service_dhi_dni_gap.md`` in project
memory for the discovery context.

Expected effect
---------------
* Pre: ``dim_variable`` has 48 rows, no ``physical_storage`` column.
* Post: 48 rows, all with ``physical_storage`` populated. 9 rows tagged
  non-default (3 NASA irradiance + 3 CAMS irradiance + 3 MODIS); the
  other 39 are ``long_format``.
* No row deletions; no API calls. ALTER TABLE + single MERGE. Subsecond
  runtime.

Reversal
--------
``ALTER TABLE … DROP COLUMN physical_storage`` would roll back the
schema change. The catalog code change is independent — to fully
revert, also revert the ``VariableSpec.physical_storage`` field
addition.
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

_MIGRATION_ID = "2026-05-10_a11_annotate_physical_storage"
_log = logging.getLogger(_MIGRATION_ID)


def _column_exists(bq: BigQueryClient, table_fqn: str, column: str) -> bool:
    """True iff ``column`` is already present on ``table_fqn`` in BQ."""
    project, dataset, table = table_fqn.split(".")
    df = bq.query(f"""
    SELECT column_name
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    """)
    return column in set(df["column_name"])


def _ensure_column(bq: BigQueryClient, table_fqn: str) -> bool:
    """Add ``physical_storage`` column if missing. Returns True iff added."""
    if _column_exists(bq, table_fqn, "physical_storage"):
        _log.info("physical_storage column already present; skipping ALTER.")
        return False
    _log.info("Adding physical_storage column to %s …", table_fqn)
    bq.query(f"""
    ALTER TABLE `{table_fqn}`
    ADD COLUMN physical_storage STRING
    """)
    return True


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    table_fqn = refs.dim_variable

    if dry_run:
        already_present = _column_exists(bq, table_fqn, "physical_storage")
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; physical_storage column "
                f"{'already present' if already_present else 'would be added'}; "
                f"would upsert {len(VariableCatalog.all_variables())} rows."
            ),
        )

    column_added = _ensure_column(bq, table_fqn)
    rows_written = populate_dim_variable(bq, table_fqn=table_fqn)

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=rows_written,
        notes=(
            f"{'Added physical_storage column. ' if column_added else ''}"
            f"Upserted {rows_written} catalog rows with the storage tag."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Annotate dim_variable with the physical_storage column so "
                "FeatureSelection can validate that requested variable_ids "
                "match the storage backend the field reads from."
            ),
        )
    )
