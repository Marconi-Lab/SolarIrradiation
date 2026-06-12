"""B9 — region-shaped MERRA-2 ingest for stations + Uganda 2024 grid.

Replaces b8 wholesale. The behaviour is the same set of rows (28 ground
stations + the Uganda 2024 NASA-POWER-resolution grid, 4 MERRA-2 catalog
variables) but the fetch pattern is now region-shaped: one OPeNDAP bbox
call per (date, variable) covering all locations in the plan, instead of
one call per (date, variable, point). See the b9 design memo and
:mod:`susse.api_clients.merra_2.merra_daily_fetcher` for the rationale.

Why a new migration ID rather than editing b8
----------------------------------------------
* b8 used :class:`MerraSatelliteJob` (now removed) which fanned out per
  station with each station's own date range. b9 uses
  :class:`MerraRegionJob` and bundles all stations into one plan over
  the union date range.
* The bundled date range over-fetches some station-date combinations
  (a station with one year of ground truth gets MERRA rows for the full
  union range). Storage cost is trivial (~0.5M rows in BQ) and the
  cached rows accelerate later inference. Net win.
* Keeping the b8 file around as historical record would be commented-out
  code; it lives in git history instead.

Expected effect
---------------
* **Per-(date, variable) cost**: 1 OPeNDAP call returning all points'
  sub-daily values. Wall time ~3 s regardless of point count.
* **Stations bundle**: 1 plan, union(min_date, max_date) across all 28
  stations, 4 variables. ~13 yr × 365 d × 4 vars / 8 workers × ~3 s
  ≈ 2 hours wall time.
* **Uganda 2024 grid bundle**: 1 plan, 156 points, 4 variables, 366 days.
  ~10 minutes wall time.
* Re-runs are cheap: the coverage layer's existing-keys filter spares
  any (date, geohash, variable_id) tuples already in the warehouse from
  re-upload, even though every (date, variable) bbox call is still made.

Prerequisites
-------------
* ``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD`` in ``.env``. Register
  at https://urs.earthdata.nasa.gov/ if needed; approve the *NASA GESDISC
  DATA ARCHIVE* application via the Earthdata profile.
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
from susse.warehouse_ops.population.jobs.merra_region_job import MerraRegionJob
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

_MIGRATION_ID = "2026-05-10_b9_ingest_merra_region"
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


def _run_stations_bundle(
    job: MerraRegionJob, stations_df, dry_run: bool
) -> tuple[int, int]:
    """One bundled NamedLocationsPlan over all stations × union date range.

    Returns (rows_added, api_calls).
    """
    catalog_vars = VariableCatalog.for_source(Source.MERRA_2)
    if not catalog_vars:
        raise RuntimeError("Empty MERRA-2 catalog — check VariableCatalog.")
    locations = tuple(
        LocationSpec(name=row.location, lat=float(row.lat), lon=float(row.lon))
        for row in stations_df.itertuples(index=False)
    )
    union_start = min(stations_df["min_date"])
    union_end = max(stations_df["max_date"])
    date_range = DateRange(start=union_start, end=union_end)

    plan = NamedLocationsPlan(
        source=Source.MERRA_2,
        date_range=date_range,
        locations=locations,
        variables=catalog_vars,
    )
    _log.info(
        "[MERRA2-stations] %d stations, %d days (union %s..%s), %d vars",
        len(locations), date_range.n_days,
        date_range.start, date_range.end, len(catalog_vars),
    )
    if dry_run:
        return 0, 0
    result = job.run(plan)
    _log.info(
        "[MERRA2-stations] rows_added=%d, api_calls=%d, duration=%.0fs",
        result.rows_added, result.api_calls_made, result.duration_seconds,
    )
    return result.rows_added, result.api_calls_made


def _run_grid(
    job: MerraRegionJob, dry_run: bool
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
        grid.n_points, _UGANDA_GRID_DATES.n_days, len(plan.variables),
    )
    if dry_run:
        return 0, 0, grid.n_points
    result = job.run(plan)
    _log.info(
        "[MERRA2-grid] rows_added=%d, api_calls=%d, duration=%.0fs",
        result.rows_added, result.api_calls_made, result.duration_seconds,
    )
    return result.rows_added, result.api_calls_made, grid.n_points


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    if not dry_run:
        _check_credentials_present()
    job = MerraRegionJob(bq, refs=refs)
    stations_df = _load_stations(bq, refs)

    station_rows, station_calls = _run_stations_bundle(
        job, stations_df, dry_run
    )
    grid_rows, grid_calls, n_grid_points = _run_grid(job, dry_run)

    if dry_run:
        return MigrationResult(
            migration_id=_MIGRATION_ID,
            rows_affected=0,
            notes=(
                f"dry-run; would issue 1 stations plan ({len(stations_df)} "
                f"stations) and 1 grid plan ({n_grid_points} points)."
            ),
        )

    return MigrationResult(
        migration_id=_MIGRATION_ID,
        rows_affected=station_rows + grid_rows,
        notes=(
            f"Stations: +{station_rows} rows, {station_calls} api call(s). "
            f"Grid: +{grid_rows} rows, {grid_calls} api call(s), "
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
                "2024 NASA-POWER-resolution grid via region-shaped fetches."
            ),
        )
    )
