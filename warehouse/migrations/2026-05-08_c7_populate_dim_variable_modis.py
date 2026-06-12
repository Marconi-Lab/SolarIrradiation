"""C7 — extend ``dim_variable`` with the MODIS catalog entries.

What it changes
---------------
Re-runs ``populate_dim_variable`` against the full catalog. The catalog
now includes 3 MODIS ``VariableSpec``s (added in C2), so the MERGE
inserts those new rows. NASA POWER, CAMS, and MERRA-2 rows are already
there and are no-ops on the merge keys ``(variable_id, source)``.

The MODIS ``variable_id`` follows the ``"{product}_{band}"`` convention
(e.g. ``MOD13Q1_250m_16_days_NDVI``) so the row in ``dim_variable``
uniquely identifies a (product, band) pair while the storage table
``modis_observations`` keeps ``product_id`` and ``band_id`` as separate
columns for downstream querying.

Why
---
Phase C mirrors A1/B7 for the MODIS source: every variable that will
land in ``modis_observations`` needs a corresponding ``dim_variable``
row documenting its unit, valid range, and spatial resolution. Without
this, downstream tools can't introspect the catalog from BigQuery.

Expected effect
---------------
* No API calls.
* Pre: ``dim_variable`` has 46 rows (NASA + CAMS + MERRA2 from A1/B7).
* Post: ``dim_variable`` has 49 rows (46 + 3 MODIS).
* Idempotent: re-running with the catalog already populated is a no-op.

Reversal
--------
``DELETE FROM dim_variable WHERE source = 'MODIS'`` removes the three
MODIS rows. Safe before C8 has loaded any data; once C8 has populated
``modis_observations`` you'd want to keep the catalog rows so the schema
stays self-documenting.
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

_MIGRATION_ID = "2026-05-08_c7_populate_dim_variable_modis"
_log = logging.getLogger(_MIGRATION_ID)


def _read_state(bq: BigQueryClient, refs: TableRefs) -> dict:
    df = bq.query(
        f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNTIF(source = 'NASA') AS n_nasa,
          COUNTIF(source = 'CAMS') AS n_cams,
          COUNTIF(source = 'MERRA2') AS n_merra,
          COUNTIF(source = 'MODIS') AS n_modis
        FROM `{refs.dim_variable}`
        """
    )
    return df.iloc[0].to_dict()


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    catalog = VariableCatalog.all_variables()
    n_modis_in_catalog = sum(1 for v in catalog if v.source is Source.MODIS)
    _log.info(
        "Catalog has %d total variables, %d of them MODIS.",
        len(catalog), n_modis_in_catalog,
    )

    pre = _read_state(bq, refs)
    _log.info(
        "Pre: dim_variable n_rows=%s NASA=%s CAMS=%s MERRA2=%s MODIS=%s",
        pre["n_rows"], pre["n_nasa"], pre["n_cams"], pre["n_merra"], pre["n_modis"],
    )

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would MERGE {len(catalog)} catalog rows; expect "
                f"~{n_modis_in_catalog} new MODIS rows."
            ),
        )

    written = populate_dim_variable(
        bq, table_fqn=refs.dim_variable, variables=catalog
    )
    post = _read_state(bq, refs)
    _log.info(
        "Post: dim_variable n_rows=%s NASA=%s CAMS=%s MERRA2=%s MODIS=%s",
        post["n_rows"], post["n_nasa"], post["n_cams"], post["n_merra"], post["n_modis"],
    )

    if post["n_modis"] != n_modis_in_catalog:
        raise RuntimeError(
            f"Expected {n_modis_in_catalog} MODIS rows in dim_variable "
            f"after MERGE; got {post['n_modis']}. Investigate."
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=written,
        notes=(
            f"dim_variable: pre n_rows={pre['n_rows']} → post "
            f"n_rows={post['n_rows']}; MODIS rows {pre['n_modis']} → "
            f"{post['n_modis']}."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Extend dim_variable with the MODIS catalog entries."
            ),
        )
    )
