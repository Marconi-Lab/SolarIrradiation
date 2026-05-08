"""B8 — ingest MERRA-2 for the 28 ground stations and the Uganda 2024 grid.

What it changes
---------------
For each of the 28 distinct locations in ``ground_measurements``, plus
every point on the existing Uganda 2024 NASA POWER grid (1962 points
× 366 days), pulls the 4 MERRA-2 catalog variables and lands them in
``merra_daily_vars_long`` after cosine-zenith-weighted daily aggregation.

Why
---
This is the Phase-B counterpart of A6: bring MERRA-2 into the warehouse
for the same training-corpus and inference-cache slices we covered with
NASA POWER and CAMS. After this run, the bias-correction model has
access to MERRA-2's aerosol decomposition and direct precipitable water
fields at every (station, date) and every Uganda 2024 grid point.

Expected effect
---------------
* **Per-station-day cost**: 3 OPeNDAP queries (one per MERRA-2 collection
  — TOTSCATAU and TOTEXTTAU share ``tavg1_2d_aer_Nx``, AODANA is in
  ``inst3_2d_gas_Nx``, TQV is in ``tavg1_2d_slv_Nx``).
* Per-station total: ~5 years × 365 days × 3 queries ≈ 5,500 queries.
* **Total ingest cost**: 28 stations + 1962 grid points ≈ 11M queries
  cumulative. Slow but bounded; the per-(geohash5, date) coverage check
  makes interruptions safe — re-running picks up where it left off.
* Per-row volume: small — one row per (station, day, variable),
  similarly for grid points.

Prerequisites
-------------
* ``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD`` in ``.env``. Register
  at https://urs.earthdata.nasa.gov/ if you don't have an account, then
  approve the *NASA GESDISC DATA ARCHIVE* application via the Earthdata
  profile page (one-time, free).
* B3 (table created) and B7 (catalog populated) must have run first.

Reversal
--------
Possible but tedious. Easier to leave the data in place; rerunning is
idempotent.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.jobs.satellite_job import MerraSatelliteJob
from susse.warehouse_ops.population.types import (
    BoundingBox,
    DateRange,
    GridPlan,
    GridSpec,
    LocationSpec,
    NamedLocationsPlan,
    Source,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MigrationResult, run_migration  # noqa: E402

_MIGRATION_ID = "2026-05-08_b8_ingest_merra_stations_and_grid"
_log = logging.getLogger(_MIGRATION_ID)

# Uganda inference-cache footprint, mirroring the existing NASA POWER /
# CAMS coverage for 2024. NASA POWER native resolution is 0.5°.
_UGANDA_BBOX = BoundingBox(
    min_lat=-1.5, max_lat=4.5, min_lon=29.5, max_lon=35.0,
)
_UGANDA_GRID_STEP = 0.5
_UGANDA_GRID_DATES = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))


def _check_credentials_present() -> None:
    load_dotenv()
    missing = [
        name for name in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not set in environment or .env. "
            f"Register at https://urs.earthdata.nasa.gov/ if needed, then "
            f"approve the 'NASA GESDISC DATA ARCHIVE' application before "
            f"re-running."
        )


def _load_stations(bq: BigQueryClient, refs: TableRefs):
    df = bq.query(
        f"""
        SELECT
          location, MIN(lat) AS lat, MIN(lon) AS lon,
          MIN(date) AS min_date, MAX(date) AS max_date
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


def _run_named_locations(
    job: MerraSatelliteJob, stations_df, dry_run: bool
) -> tuple[int, int, int]:
    catalog_vars = VariableCatalog.for_source(Source.MERRA_2)
    if not catalog_vars:
        raise RuntimeError("Empty MERRA-2 catalog — check VariableCatalog.")
    rows_added = 0
    api_calls = 0
    processed = 0

    for row in stations_df.itertuples(index=False):
        loc = LocationSpec(
            name=row.location, lat=float(row.lat), lon=float(row.lon)
        )
        date_range = DateRange(start=row.min_date, end=row.max_date)
        plan = NamedLocationsPlan(
            source=Source.MERRA_2,
            date_range=date_range,
            locations=(loc,),
            variables=catalog_vars,
        )
        _log.info(
            "[MERRA2] %s — %d days, %d vars",
            row.location, date_range.n_days, len(catalog_vars),
        )
        if dry_run:
            processed += 1
            continue
        try:
            result = job.run(plan)
        except Exception:
            _log.exception(
                "[MERRA2] %s — fetch failed; continuing with next station.",
                row.location,
            )
            continue
        rows_added += result.rows_added
        api_calls += result.api_calls_made
        processed += 1
        _log.info(
            "[MERRA2] %s — rows_added=%d, api_calls=%d",
            row.location, result.rows_added, result.api_calls_made,
        )
    return rows_added, api_calls, processed


def _run_grid(
    job: MerraSatelliteJob, dry_run: bool
) -> tuple[int, int, int]:
    grid = GridSpec(
        bbox=_UGANDA_BBOX,
        lat_step=_UGANDA_GRID_STEP,
        lon_step=_UGANDA_GRID_STEP,
    )
    plan = GridPlan(
        source=Source.MERRA_2,
        date_range=_UGANDA_GRID_DATES,
        grid=grid,
        variables=VariableCatalog.for_source(Source.MERRA_2),
    )
    _log.info(
        "[MERRA2-grid] %d points, %d days, %d vars",
        grid.n_points, _UGANDA_GRID_DATES.n_days,
        len(plan.variables),
    )
    if dry_run:
        return 0, 0, grid.n_points
    result = job.run(plan)
    _log.info(
        "[MERRA2-grid] rows_added=%d, api_calls=%d",
        result.rows_added, result.api_calls_made,
    )
    return result.rows_added, result.api_calls_made, grid.n_points


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    if not dry_run:
        _check_credentials_present()
    job = MerraSatelliteJob(bq, refs=refs)
    stations_df = _load_stations(bq, refs)

    station_rows, station_calls, n_stations = _run_named_locations(
        job, stations_df, dry_run
    )
    grid_rows, grid_calls, n_grid_points = _run_grid(job, dry_run)

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would issue MERRA-2 plans for {n_stations} stations "
                f"and a {n_grid_points}-point Uganda 2024 grid."
            ),
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=station_rows + grid_rows,
        notes=(
            f"Stations: +{station_rows} rows, {station_calls} api calls, "
            f"{n_stations} stations. "
            f"Grid: +{grid_rows} rows, {grid_calls} api calls, "
            f"{n_grid_points} points."
        ),
    )


if __name__ == "__main__":
    sys.exit(
        run_migration(
            migration_id=_MIGRATION_ID,
            fn=migrate,
            description=(
                "Ingest MERRA-2 for the 28 ground stations and the Uganda "
                "2024 NASA-POWER-resolution grid."
            ),
        )
    )
