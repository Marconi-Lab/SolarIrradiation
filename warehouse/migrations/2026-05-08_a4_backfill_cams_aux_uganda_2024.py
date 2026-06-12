"""A4 — backfill CAMS auxiliary variables for Uganda 2024.

What it changes
---------------
Reads the 6 auxiliary CAMS variables (``ghi_extra``, ``ghi_clear``,
``bhi``, ``bhi_clear``, ``dhi_clear``, ``dni_clear``) from the external
``cams_daily_ext`` CSV-backed table, converts them from W/m² mean-power
to kWh/m²/day total (factor of 0.024), unpivots into long format, and
MERGEs the result into ``cams_daily_vars_long``.

Why
---
``irradiance_daily`` already holds the CAMS GHI/DHI/DNI for the Uganda
2024 grid. The auxiliary CAMS variables (clear-sky irradiance and
extraterrestrial GHI) live in the same external CSV but were never
loaded into the warehouse. They're useful as ML features (the kt
clearness index = GHI/GHI_clear is a key bias-correction signal).

Geohash5 is reused from ``irradiance_daily`` (joined on lat/lon) so the
keys remain consistent with the rest of the warehouse.

Expected effect
---------------
* No API calls.
* Single MERGE statement.
* Pre: ``cams_daily_vars_long`` is empty (created in A3).
* Post: ``cams_daily_vars_long`` holds ~4.31M rows
  (1,962 grid points × 366 days × 6 aux variables).
* Idempotent: re-running matches existing keys via the schema's
  ``(date, geohash5, variable_id, source)`` MERGE key, no duplicates.

Reversal
--------
``DELETE FROM cams_daily_vars_long WHERE source = 'CAMS'``. Safe to
re-run A4 afterwards.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs, TableSchemas
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.types import Source

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_a4_backfill_cams_aux_uganda_2024"
_log = logging.getLogger(_MIGRATION_ID)

# Conversion factor: external CSV stores values as W/m² averaged over a
# 24-hour observation period; target column is kWh/m²/day total energy.
# kWh/m²/day = (W/m²) × 24 hours × (1 kW / 1000 W) = (W/m²) × 0.024.
_W_M2_TO_KWH_M2_DAY = 0.024

# External table source name.
_EXT_TABLE = "cams_daily_ext"


def _build_merge_sql(
    cams_long_fqn: str,
    irradiance_fqn: str,
    ext_fqn: str,
    aux_variable_ids: tuple[str, ...],
) -> str:
    """Construct the MERGE that unpivots ``cams_daily_ext`` into long format.

    The unpivot column list is the 6 aux variable_ids; their string
    spellings in the catalog match the column names in the external
    table (verified A4).
    """
    unpivot_cols = ", ".join(aux_variable_ids)
    # GEOGRAPHY columns can't be used in SELECT DISTINCT, so we group by
    # the (lat, lon) pair and pick ANY_VALUE for geohash5 + geog (both are
    # functionally determined by lat/lon, so any row works).
    return f"""
    MERGE `{cams_long_fqn}` t
    USING (
        WITH points AS (
            SELECT
                latitude,
                longitude,
                ANY_VALUE(geohash5) AS geohash5,
                ANY_VALUE(geog) AS geog
            FROM `{irradiance_fqn}`
            WHERE source = 'CAMS'
            GROUP BY latitude, longitude
        ),
        ext_aux AS (
            SELECT
                DATE(ts) AS date,
                latitude,
                longitude,
                ghi_extra, ghi_clear, bhi, bhi_clear, dhi_clear, dni_clear
            FROM `{ext_fqn}`
        ),
        long_unpivoted AS (
            SELECT * FROM ext_aux
            UNPIVOT(raw_value FOR variable_id IN ({unpivot_cols}))
        )
        SELECT
            u.date,
            u.latitude,
            u.longitude,
            p.geog,
            p.geohash5,
            u.variable_id,
            u.raw_value * {_W_M2_TO_KWH_M2_DAY} AS value,
            '{Source.CAMS.value}' AS source
        FROM long_unpivoted u
        JOIN points p
          ON u.latitude = p.latitude
         AND u.longitude = p.longitude
    ) s
    ON  t.date = s.date
    AND t.geohash5 = s.geohash5
    AND t.variable_id = s.variable_id
    AND t.source = s.source
    WHEN NOT MATCHED THEN
        INSERT (date, latitude, longitude, geog, geohash5,
                variable_id, value, source)
        VALUES (s.date, s.latitude, s.longitude, s.geog, s.geohash5,
                s.variable_id, s.value, s.source)
    """


def _count_rows(bq: BigQueryClient, table_fqn: str) -> int:
    df = bq.query(f"SELECT COUNT(*) AS n FROM `{table_fqn}`")
    return int(df.iloc[0]["n"])


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    cams_long_fqn = refs.cams_daily_vars_long
    irr_fqn = refs.irradiance_daily
    ext_fqn = bq.config.fqn(_EXT_TABLE)

    aux_specs = VariableCatalog.auxiliary_for_source(Source.CAMS)
    aux_ids = tuple(spec.variable_id for spec in aux_specs)
    if not aux_ids:
        raise RuntimeError(
            "No auxiliary CAMS variables in the catalog. Aborting before MERGE."
        )
    _log.info("CAMS auxiliary variables to backfill: %s", aux_ids)

    pre_n = _count_rows(bq, cams_long_fqn)
    _log.info("%s pre-state: %d rows.", TableSchemas.CAMS_DAILY_VARS_LONG.table_id, pre_n)

    sql = _build_merge_sql(cams_long_fqn, irr_fqn, ext_fqn, aux_ids)
    _log.debug("MERGE SQL:\n%s", sql)

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would MERGE 6 unpivoted CAMS aux vars from "
                f"{_EXT_TABLE} into cams_daily_vars_long. Pre-rows={pre_n}."
            ),
        )

    bq.execute_ddl(sql)
    post_n = _count_rows(bq, cams_long_fqn)
    inserted = post_n - pre_n
    _log.info("%s post-state: %d rows (+%d inserted).",
              TableSchemas.CAMS_DAILY_VARS_LONG.table_id, post_n, inserted)

    expected = 1962 * 366 * len(aux_ids)
    if inserted not in (0, expected):
        _log.warning(
            "Inserted row count %d is neither 0 (already-applied) nor %d "
            "(1962 grid × 366 days × %d aux vars). Investigate the join "
            "or unpivot logic.",
            inserted, expected, len(aux_ids),
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=inserted,
        notes=(
            f"cams_daily_vars_long: {pre_n} → {post_n} rows (+{inserted})."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "MERGE CAMS auxiliary variables (Uganda 2024) from "
                "cams_daily_ext into cams_daily_vars_long."
            ),
        )
    )
