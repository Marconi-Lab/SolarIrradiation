"""C8 — ingest MODIS for the 28 ground stations and the Uganda 2024 grid.

What it changes
---------------
For each of the 28 distinct locations in ``ground_measurements``, plus
every point on the existing Uganda 2024 NASA-POWER-resolution grid
(1962 points × 366 days), pulls the 3 MODIS catalog (product, band)
pairs from ORNL DAAC and lands them in ``modis_observations``.

Why
---
This is the Phase-C counterpart of A6 / B8: bring MODIS into the
warehouse for the same training-corpus and inference-cache slices we
covered with NASA POWER, CAMS, and MERRA-2. After this run, the
bias-correction model has access to MODIS surface reflectance, land
surface temperature, and NDVI at every (station, date) and every Uganda
2024 grid point.

Because MODIS values are stored at each product's *native composite
end-date* (1-day for MCD43A4, 8-day for MOD11A2, 16-day for MOD13Q1),
the row count per (station, date-range) is much smaller than for the
daily satellite sources — but each row is sampled, not interpolated, so
the data quality is honest about its temporal resolution.

Expected effect
---------------
* **Per-station cost**: one ORNL DAAC subset call per (product, chunk).
  For a 5-year window: MCD43A4 daily-cadence × 10-day-chunks ≈ 183
  chunks; MOD11A2 8-day × 10-tile chunks ≈ 23 chunks; MOD13Q1 16-day
  × 10-tile chunks ≈ 12 chunks. Total ~220 calls per station, ~6,200
  cumulative for 28 stations.
* **Grid cost**: ~220 calls × 1962 points ≈ 430k cumulative calls. ORNL
  DAAC is unauthenticated and unrate-limited but slow per call; expect
  hours per station and *days* for the full grid. The per-(geohash5,
  product, band) coverage check makes interruptions safe — re-running
  picks up where it left off.

Prerequisites
-------------
* C3 (table created) and C7 (catalog populated) must have run first.
* No credentials needed — ORNL DAAC's RST subset endpoint is public.

Reversal
--------
Possible but tedious. Easier to leave the data in place; rerunning is
idempotent.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.jobs import ModisJob
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

_MIGRATION_ID = "2026-05-08_c8_ingest_modis_stations_and_grid"
_log = logging.getLogger(_MIGRATION_ID)

# Uganda inference-cache footprint, mirroring the existing NASA POWER /
# CAMS / MERRA-2 coverage for 2024. NASA POWER native resolution is 0.5°.
_UGANDA_BBOX = BoundingBox(
    min_lat=-1.5, max_lat=4.5, min_lon=29.5, max_lon=35.0,
)
_UGANDA_GRID_STEP = 0.5
_UGANDA_GRID_DATES = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))


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
    job: ModisJob, stations_df, dry_run: bool
) -> tuple[int, int, int]:
    catalog_vars = VariableCatalog.for_source(Source.MODIS)
    if not catalog_vars:
        raise RuntimeError("Empty MODIS catalog — check VariableCatalog.")
    rows_added = 0
    api_calls = 0
    processed = 0

    for row in stations_df.itertuples(index=False):
        loc = LocationSpec(
            name=row.location, lat=float(row.lat), lon=float(row.lon)
        )
        date_range = DateRange(start=row.min_date, end=row.max_date)
        plan = NamedLocationsPlan(
            source=Source.MODIS,
            date_range=date_range,
            locations=(loc,),
            variables=catalog_vars,
        )
        _log.info(
            "[MODIS] %s — %d days, %d (product, band) pair(s)",
            row.location, date_range.n_days, len(catalog_vars),
        )
        if dry_run:
            processed += 1
            continue
        try:
            result = job.run(plan)
        except Exception:
            _log.exception(
                "[MODIS] %s — fetch failed; continuing with next station.",
                row.location,
            )
            continue
        rows_added += result.rows_added
        api_calls += result.api_calls_made
        processed += 1
        _log.info(
            "[MODIS] %s — rows_added=%d, api_calls=%d",
            row.location, result.rows_added, result.api_calls_made,
        )
    return rows_added, api_calls, processed


def _run_grid(job: ModisJob, dry_run: bool) -> tuple[int, int, int]:
    grid = GridSpec(
        bbox=_UGANDA_BBOX,
        lat_step=_UGANDA_GRID_STEP,
        lon_step=_UGANDA_GRID_STEP,
    )
    plan = GridPlan(
        source=Source.MODIS,
        date_range=_UGANDA_GRID_DATES,
        grid=grid,
        variables=VariableCatalog.for_source(Source.MODIS),
    )
    _log.info(
        "[MODIS-grid] %d points, %d days, %d (product, band) pair(s)",
        grid.n_points, _UGANDA_GRID_DATES.n_days,
        len(plan.variables),
    )
    if dry_run:
        return 0, 0, grid.n_points
    result = job.run(plan)
    _log.info(
        "[MODIS-grid] rows_added=%d, api_calls=%d",
        result.rows_added, result.api_calls_made,
    )
    return result.rows_added, result.api_calls_made, grid.n_points


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    job = ModisJob(bq, refs=refs)
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
                f"dry-run; would issue MODIS plans for {n_stations} stations "
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
                "Ingest MODIS for the 28 ground stations and the Uganda "
                "2024 NASA-POWER-resolution grid."
            ),
        )
    )
