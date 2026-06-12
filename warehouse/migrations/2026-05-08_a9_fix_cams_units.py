"""A9 — fix CAMS unit-conversion bug introduced by A6.

What it changes
---------------
For every CAMS row in ``irradiance_daily`` and ``cams_daily_vars_long``
that was written without the W/m² → kWh/m²/day conversion, multiplies
the value by 0.024 in place. Affected rows are identified by magnitude:
CAMS values stored without the conversion are ~100–400 (W/m² mean over
the day), while correctly-converted values are ~0–10 (kWh/m²/day).
A threshold of 30 is comfortably above any plausible CAMS daily energy
total but well below any unconverted W/m² mean.

Why
---
``CamsSatelliteJob._cams_dataframe_to_long`` was missing the unit
conversion that the legacy ingest path (and migration A4) applied. Every
CAMS row inserted by A6 went into the warehouse 41.7× too large
(``ghi_kwh_m2_day`` columns containing W/m² values). The code is now
fixed in the same commit; this migration corrects the data already in
BigQuery.

Affected:

* ``irradiance_daily`` rows where ``source='CAMS'`` and any of the
  ``ghi_kwh_m2_day`` / ``dhi_kwh_m2_day`` / ``dni_kwh_m2_day`` columns
  exceeds 30 (the magnitude threshold).
* ``cams_daily_vars_long`` rows where ``source='CAMS'`` and ``value``
  exceeds 30.

Unaffected (already correct):

* The Uganda 2024 grid CAMS rows in ``irradiance_daily`` (legacy ingest
  applied the conversion).
* The Uganda 2024 grid CAMS aux rows in ``cams_daily_vars_long`` written
  by migration A4 (which applied the conversion explicitly).

Expected effect
---------------
* No API calls.
* Two ``UPDATE`` statements (one per table).
* Idempotent: a re-run sees only already-converted (small-magnitude)
  rows and updates nothing.

Reversal
--------
``UPDATE … SET col = col / 0.024 WHERE source='CAMS' AND col < 30`` would
undo this, but you'd lose the distinction between A6-and-A9-corrected
rows and rows that were always correct. Practically irreversible; if
something goes wrong, the safest path is to delete CAMS rows from A6
and re-run A6 with the fixed code.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population.jobs.satellite_job import CamsSatelliteJob

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_a9_fix_cams_units"
_log = logging.getLogger(_MIGRATION_ID)

# Magnitude threshold separating buggy (W/m² mean) values from correct
# (kWh/m²/day) values. Real CAMS daily energy is bounded by the solar
# constant × 24 / 1000 ≈ 33 kWh/m²/day in the absolute extreme; in
# practice values are below 10. Anything > 30 was written without the
# 0.024 conversion.
_BAD_VALUE_THRESHOLD = 30.0

# BigQuery rejects DML statements that touch more than 4,000 partitions.
# Bad CAMS rows span ~2011 → 2024, so we chunk the UPDATE by date range
# the same way the MergeLoader does.
_PARTITION_CHUNK_DAYS = 3500


def _date_chunks_for(
    bq: BigQueryClient, table_fqn: str
) -> list[tuple[date, date]]:
    """Compute date chunks for the bad CAMS rows in ``table_fqn``.

    Returns an empty list when the table has no bad rows (skip the UPDATE
    entirely). Otherwise returns chunks each spanning at most
    ``_PARTITION_CHUNK_DAYS`` calendar days, covering [min_date, max_date]
    of the bad-row population in that table.
    """
    if "irradiance_daily" in table_fqn:
        bad_filter = (
            f"(ghi_kwh_m2_day > {_BAD_VALUE_THRESHOLD} "
            f"OR dhi_kwh_m2_day > {_BAD_VALUE_THRESHOLD} "
            f"OR dni_kwh_m2_day > {_BAD_VALUE_THRESHOLD})"
        )
    else:
        bad_filter = f"value > {_BAD_VALUE_THRESHOLD}"
    df = bq.query(
        f"""
        SELECT MIN(date) AS min_d, MAX(date) AS max_d
        FROM `{table_fqn}`
        WHERE source = 'CAMS' AND {bad_filter}
        """
    )
    row = df.iloc[0]
    if row["min_d"] is None or (hasattr(row["min_d"], "is_nat") and row["min_d"] is None):
        return []
    min_d = row["min_d"]
    max_d = row["max_d"]
    if hasattr(min_d, "date"):
        min_d = min_d.date()
    if hasattr(max_d, "date"):
        max_d = max_d.date()
    chunks: list[tuple[date, date]] = []
    cursor = min_d
    while cursor <= max_d:
        chunk_end = min(
            cursor + timedelta(days=_PARTITION_CHUNK_DAYS - 1), max_d
        )
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _count_bad_rows(bq: BigQueryClient, refs: TableRefs) -> dict[str, int]:
    irr = bq.query(
        f"""
        SELECT COUNT(*) AS n
        FROM `{refs.irradiance_daily}`
        WHERE source = 'CAMS'
          AND (ghi_kwh_m2_day > {_BAD_VALUE_THRESHOLD}
               OR dhi_kwh_m2_day > {_BAD_VALUE_THRESHOLD}
               OR dni_kwh_m2_day > {_BAD_VALUE_THRESHOLD})
        """
    ).iloc[0]["n"]
    long = bq.query(
        f"""
        SELECT COUNT(*) AS n
        FROM `{refs.cams_daily_vars_long}`
        WHERE source = 'CAMS' AND value > {_BAD_VALUE_THRESHOLD}
        """
    ).iloc[0]["n"]
    return {"irradiance_daily": int(irr), "cams_daily_vars_long": int(long)}


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    factor = CamsSatelliteJob._W_M2_TO_KWH_M2_DAY
    pre = _count_bad_rows(bq, refs)
    _log.info(
        "Pre: irradiance_daily bad rows=%d, cams_daily_vars_long bad rows=%d.",
        pre["irradiance_daily"], pre["cams_daily_vars_long"],
    )
    if pre["irradiance_daily"] == 0 and pre["cams_daily_vars_long"] == 0:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes="No CAMS rows above the magnitude threshold — nothing to fix.",
        )

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would correct {pre['irradiance_daily']} irradiance "
                f"rows and {pre['cams_daily_vars_long']} long-table rows by "
                f"multiplying by {factor}."
            ),
        )

    irr_chunks = _date_chunks_for(bq, refs.irradiance_daily)
    long_chunks = _date_chunks_for(bq, refs.cams_daily_vars_long)
    _log.info(
        "Date-chunking the UPDATEs: irradiance_daily into %d chunk(s), "
        "cams_daily_vars_long into %d chunk(s).",
        len(irr_chunks), len(long_chunks),
    )

    # irradiance_daily: ghi/dhi/dni columns. UPDATE only the columns that
    # actually exceed the threshold so a single bad column doesn't
    # double-correct the others.
    for start, end in irr_chunks:
        bq.execute_ddl(
            f"""
            UPDATE `{refs.irradiance_daily}`
            SET
              ghi_kwh_m2_day = IF(ghi_kwh_m2_day > {_BAD_VALUE_THRESHOLD},
                                  ghi_kwh_m2_day * {factor}, ghi_kwh_m2_day),
              dhi_kwh_m2_day = IF(dhi_kwh_m2_day > {_BAD_VALUE_THRESHOLD},
                                  dhi_kwh_m2_day * {factor}, dhi_kwh_m2_day),
              dni_kwh_m2_day = IF(dni_kwh_m2_day > {_BAD_VALUE_THRESHOLD},
                                  dni_kwh_m2_day * {factor}, dni_kwh_m2_day)
            WHERE source = 'CAMS'
              AND date BETWEEN DATE('{start.isoformat()}')
                          AND DATE('{end.isoformat()}')
              AND (ghi_kwh_m2_day > {_BAD_VALUE_THRESHOLD}
                   OR dhi_kwh_m2_day > {_BAD_VALUE_THRESHOLD}
                   OR dni_kwh_m2_day > {_BAD_VALUE_THRESHOLD})
            """
        )
    for start, end in long_chunks:
        bq.execute_ddl(
            f"""
            UPDATE `{refs.cams_daily_vars_long}`
            SET value = value * {factor}
            WHERE source = 'CAMS'
              AND date BETWEEN DATE('{start.isoformat()}')
                          AND DATE('{end.isoformat()}')
              AND value > {_BAD_VALUE_THRESHOLD}
            """
        )

    post = _count_bad_rows(bq, refs)
    _log.info(
        "Post: irradiance_daily bad rows=%d, cams_daily_vars_long bad rows=%d.",
        post["irradiance_daily"], post["cams_daily_vars_long"],
    )
    if post["irradiance_daily"] != 0 or post["cams_daily_vars_long"] != 0:
        raise RuntimeError(
            f"After UPDATE, {post['irradiance_daily']} irradiance rows and "
            f"{post['cams_daily_vars_long']} long-table rows still exceed the "
            f"threshold. Investigate before re-running."
        )
    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=pre["irradiance_daily"] + pre["cams_daily_vars_long"],
        notes=(
            f"Corrected {pre['irradiance_daily']} irradiance_daily rows and "
            f"{pre['cams_daily_vars_long']} cams_daily_vars_long rows "
            f"(× {factor})."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Fix CAMS unit conversion: multiply rows above magnitude "
                "threshold by 0.024 in irradiance_daily and "
                "cams_daily_vars_long."
            ),
        )
    )
