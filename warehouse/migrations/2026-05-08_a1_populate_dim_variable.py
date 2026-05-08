"""A1 — populate ``dim_variable`` with the full NASA POWER + CAMS catalog.

What it changes
---------------
Upserts 41 rows into ``solar_warehouse.dim_variable``:

* 32 NASA POWER rows (29 auxiliary + 3 irradiance), ``source = 'NASA'``.
* 9 CAMS rows (6 auxiliary + 3 irradiance), ``source = 'CAMS'``.

Why
---
Before this migration the table contained only 4 NASA rows, all carrying
the sentinel ``spatial_resolution_km = -51.0``. Phase A1 of the warehouse
consistency plan: bring the catalog into alignment with what's actually
ingested into ``nasa_daily_vars_long`` and what will be ingested into
``cams_daily_vars_long``.

Expected effect
---------------
* Pre: 4 rows, all with bogus ``spatial_resolution_km``.
* Post: 35 rows, correct resolutions (NASA POWER 55.5 km, CAMS 5.5 km).
* The 4 pre-existing variable_ids (``aod_550``, ``relative_humidity``,
  ``surface_pressure``, ``temperature``) are UPDATED in place by the
  MERGE on ``(variable_id, source)``; their sentinel values are replaced.
* No API calls. Single MERGE statement. Subsecond runtime.

Reversal
--------
Not strictly reversible (the pre-existing 4 rows had bogus values that
we would not want to restore). To roll back to a different catalog,
edit ``VariableCatalog`` and re-run; MERGE is idempotent.
"""

from __future__ import annotations

import logging
import sys

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population.dim_variable import (
    VariableCatalog,
    populate_dim_variable,
    variables_to_dataframe,
)

# Allow running this file as a script: ``python warehouse/migrations/<file>.py``.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_a1_populate_dim_variable"
_log = logging.getLogger(_MIGRATION_ID)


def _read_state(bq: BigQueryClient, refs: TableRefs) -> dict:
    """Snapshot ``dim_variable`` for before/after comparison."""
    summary = bq.query(
        f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNTIF(spatial_resolution_km < 0) AS n_sentinel,
          COUNTIF(source = 'NASA') AS n_nasa,
          COUNTIF(source = 'CAMS') AS n_cams
        FROM `{refs.dim_variable}`
        """
    )
    return summary.iloc[0].to_dict()


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    catalog = VariableCatalog.all_variables()
    df = variables_to_dataframe(catalog)
    _log.info(
        "Catalog has %d variables (%d NASA, %d CAMS).",
        len(catalog),
        sum(1 for v in catalog if v.source.value == "NASA"),
        sum(1 for v in catalog if v.source.value == "CAMS"),
    )

    pre = _read_state(bq, refs)
    _log.info(
        "Pre: dim_variable n_rows=%s sentinel=%s NASA=%s CAMS=%s",
        pre["n_rows"], pre["n_sentinel"], pre["n_nasa"], pre["n_cams"],
    )

    if dry_run:
        _log.info(
            "Dry run — would MERGE %d rows into %s. Sample:\n%s",
            len(df), refs.dim_variable, df.head(3).to_string(index=False),
        )
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=f"dry-run; would have written {len(df)} rows.",
        )

    written = populate_dim_variable(
        bq, table_fqn=refs.dim_variable, variables=catalog
    )

    post = _read_state(bq, refs)
    _log.info(
        "Post: dim_variable n_rows=%s sentinel=%s NASA=%s CAMS=%s",
        post["n_rows"], post["n_sentinel"], post["n_nasa"], post["n_cams"],
    )

    if post["n_sentinel"] != 0:
        raise RuntimeError(
            f"dim_variable still has {post['n_sentinel']} rows with "
            f"spatial_resolution_km < 0 after the MERGE. Investigate before "
            f"continuing — the catalog rows should have replaced the sentinels."
        )
    if post["n_rows"] < len(df):
        raise RuntimeError(
            f"dim_variable has {post['n_rows']} rows after MERGE, expected "
            f">= {len(df)}. Something went wrong with the upsert."
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=written,
        notes=(
            f"dim_variable populated: pre n_rows={pre['n_rows']} → "
            f"post n_rows={post['n_rows']}; sentinel rows cleared "
            f"({pre['n_sentinel']} → 0)."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description="Populate dim_variable with the full NASA POWER + CAMS catalog.",
        )
    )
