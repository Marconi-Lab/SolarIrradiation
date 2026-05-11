"""A6 — ingest NASA POWER + CAMS for the 28 ground-measurement stations.

What it changes
---------------
For each of the 28 distinct locations in ``ground_measurements``:

* Pull NASA POWER for **every** catalog NASA variable
  (``VariableCatalog.for_source(Source.NASA_POWER)``) over the station's
  full ``[min(date), max(date)]`` ground-data range.
* Pull CAMS radiation (all catalog CAMS variables) over the same range.

The fetched values land in ``nasa_daily_vars_long``,
``cams_daily_vars_long``, and ``irradiance_daily`` according to the
existing :class:`SatelliteJob` routing (irradiance subset → wide table,
everything else → the long companion).

Why
---
Without this step the warehouse has zero satellite/ground training pairs.
The 28 stations sit outside the existing Uganda-2024 grid footprint, so we
must pull satellite data at each station's exact (lat, lon) for its ground
date range. After this migration the pair-availability check in
``warehouse/extending_the_warehouse.ipynb`` must show
``n_nasa_pairs > 0`` and ``n_cams_pairs > 0`` for every station.

Expected effect
---------------
* 28 NASA POWER API calls (one per station, multi-variable per call).
* 28 CAMS API calls (one per station, multi-variable per call).
* Combined rate-limit footprint is well under both providers' daily
  quotas (CAMS ~40/day, NASA POWER no published limit but generally
  generous).
* Runtime: several minutes; depends on station date ranges. Long-history
  stations like ``kampala`` (~12 years) take longer than recent
  ``kenya_locationN`` stations.
* Idempotent: ``SatelliteJob._location_fully_cached`` checks coverage
  per (geohash5, date) before issuing the API call, so a second run
  skips stations whose data is already loaded.

Prerequisites
-------------
* CAMS requires a registered email. Set ``CAMS_EMAIL`` in ``.env``
  before running, or the first invocation will prompt interactively
  (which fails inside the migration runner).
* NASA POWER requires no credentials.

Reversal
--------
Possible but tedious — would need to delete rows by ``(date, geohash5)``
matching each station's footprint. Easier to just leave the data in
place; rerunning is idempotent.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.jobs.satellite_job import (
    CamsSatelliteJob,
    NasaPowerSatelliteJob,
)
from susse.warehouse_ops.population.types import (
    DateRange,
    LocationSpec,
    NamedLocationsPlan,
    Source,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_a6_ingest_28_ground_stations"
_log = logging.getLogger(_MIGRATION_ID)


def _load_stations(bq: BigQueryClient, refs: TableRefs):
    """Read per-station (location, lat, lon, min_date, max_date) from BQ."""
    df = bq.query(
        f"""
        SELECT
          location,
          MIN(lat) AS lat,
          MIN(lon) AS lon,
          MIN(date) AS min_date,
          MAX(date) AS max_date,
          COUNT(*) AS n_ground_days
        FROM `{refs.ground_measurements}`
        GROUP BY location
        ORDER BY location
        """
    )
    if df.empty:
        raise RuntimeError(
            "ground_measurements is empty — there are no stations to ingest."
        )
    _log.info("Loaded %d stations from ground_measurements.", len(df))
    return df


def _check_cams_email_present() -> None:
    # Load .env so a value placed there is visible without the user also
    # exporting it in the shell. The CAMSClient does this lazily at request
    # time; we do it eagerly so the pre-flight check matches.
    load_dotenv()
    if not os.getenv("CAMS_EMAIL"):
        raise RuntimeError(
            "CAMS_EMAIL is not set in the environment or .env. CAMS requests "
            "would prompt interactively, which breaks the migration runner. "
            "Add CAMS_EMAIL=<registered email> to .env before running."
        )


def _run_for_source(
    *,
    job,
    source: Source,
    stations_df,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Iterate stations, run a NamedLocationsPlan per station for ``source``.

    Returns ``(rows_added, api_calls_made, stations_processed)``.
    """
    catalog_vars = VariableCatalog.for_source(source)
    if not catalog_vars:
        raise RuntimeError(f"Empty catalog for source={source}.")
    _log.info(
        "Source=%s: %d catalog variables (%d irradiance, %d aux).",
        source.value,
        len(catalog_vars),
        len(VariableCatalog.irradiance_for_source(source)),
        len(VariableCatalog.auxiliary_for_source(source)),
    )

    rows_added = 0
    api_calls = 0
    stations_processed = 0

    for row in stations_df.itertuples(index=False):
        loc = LocationSpec(name=row.location, lat=float(row.lat), lon=float(row.lon))
        date_range = DateRange(start=row.min_date, end=row.max_date)
        plan = NamedLocationsPlan(
            source=source,
            date_range=date_range,
            locations=(loc,),
            variables=catalog_vars,
        )
        _log.info(
            "[%s] %s — %d days, %d vars",
            source.value, row.location, date_range.n_days, len(catalog_vars),
        )
        if dry_run:
            stations_processed += 1
            continue
        try:
            result = job.run(plan)
        except Exception:
            _log.exception(
                "[%s] %s — fetch failed; continuing with remaining stations.",
                source.value, row.location,
            )
            continue
        rows_added += result.rows_added
        api_calls += result.api_calls_made
        stations_processed += 1
        _log.info(
            "[%s] %s — rows_added=%d, api_calls=%d, skipped=%s",
            source.value, row.location,
            result.rows_added, result.api_calls_made,
            result.extra.get("skipped_locations"),
        )

    return rows_added, api_calls, stations_processed


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    stations_df = _load_stations(bq, refs)

    if not dry_run:
        _check_cams_email_present()

    nasa_job = NasaPowerSatelliteJob(bq, refs=refs)
    cams_job = CamsSatelliteJob(bq, refs=refs)

    nasa_rows, nasa_calls, nasa_n = _run_for_source(
        job=nasa_job, source=Source.NASA_POWER,
        stations_df=stations_df, dry_run=dry_run,
    )
    cams_rows, cams_calls, cams_n = _run_for_source(
        job=cams_job, source=Source.CAMS,
        stations_df=stations_df, dry_run=dry_run,
    )

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would issue {nasa_n} NASA + {cams_n} CAMS plans "
                f"(one per station). No API calls or writes."
            ),
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=nasa_rows + cams_rows,
        notes=(
            f"NASA: +{nasa_rows} rows in {nasa_calls} api calls across "
            f"{nasa_n} stations. CAMS: +{cams_rows} rows in {cams_calls} "
            f"api calls across {cams_n} stations."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Ingest NASA POWER + CAMS at the 28 ground-measurement "
                "stations over each station's ground-data date range."
            ),
        )
    )
