"""A12 — ingest NASA POWER + CAMS at the 54 Katongole 2023 station coordinates over 2017-2022.

What it changes
---------------
For each of the 54 stations listed in
``data/external_references/katongole_2023_monthly.csv``:

* Pull NASA POWER for every catalog NASA variable over the fixed window
  ``2017-01-01 .. 2022-12-31`` — the same 7-year window Katongole et al.
  (2023) average over to produce their published monthly climatology.
* Pull CAMS for every catalog CAMS variable over the same window.

The fetched values land in ``nasa_daily_vars_long``,
``cams_daily_vars_long``, and ``irradiance_daily`` according to the
existing :class:`SatelliteJob` routing.

Why
---
The recomputation notebook validates the bias-corrected model against
the Katongole 2017–2022 monthly climatology. Before this migration, the
warehouse only had 2017–2022 NASA + CAMS coverage at the 5 cells that
fell inside Uganda for the 28 training-station ingest (A6). Mapping
each Katongole site to its nearest of those 5 cells produced a median
snap distance of 62 km (max 220 km) — coarse enough that the
satellite-feature column was effectively unrelated to where the station
actually sits.

After this migration, every Katongole station has NASA + CAMS at its
own coordinates over the full 2017–2022 window, so the model's
2017–2022 monthly climatology can be computed and compared to
Katongole's climatology apples-to-apples.

Expected effect
---------------
* 54 NASA POWER API calls (one per station, multi-variable per call).
* 54 CAMS API calls (one per station, multi-variable per call).
* Combined rate-limit footprint sits within both providers' daily
  quotas (CAMS ~40/day soft cap may chunk this across 2 days for
  free-tier users; NASA POWER has no published limit and generally
  serves 54 calls in a few minutes).
* Runtime: a few minutes for NASA, up to a couple of hours for CAMS
  depending on rate-limit interactions.
* Idempotent: ``SatelliteJob._location_fully_cached`` checks coverage
  per (geohash5, date) before issuing the API call, so a second run
  skips stations whose data is already loaded.

Prerequisites
-------------
* ``CAMS_EMAIL`` in ``.env`` (a registered SoDa-Pro account).
* NASA POWER requires no credentials.

Reversal
--------
Possible but tedious — would need to delete rows by ``(date, geohash5)``
matching each station's footprint. Easier to leave the data in place;
rerunning is idempotent.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
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

_MIGRATION_ID = "2026-05-11_a12_ingest_katongole_2017_2022"
_log = logging.getLogger(_MIGRATION_ID)

# Hard-coded to the Katongole 2023 climatology window.
_VAL_START = date(2017, 1, 1)
_VAL_END = date(2022, 12, 31)

# Path to the Katongole CSV, relative to the repository root.
_KATONGOLE_CSV = (
    Path(__file__).resolve().parents[2]
    / "data" / "external_references" / "katongole_2023_monthly.csv"
)


def _load_katongole_stations() -> pd.DataFrame:
    """Load station (name, lat, lon) triples from the Katongole reference CSV."""
    if not _KATONGOLE_CSV.exists():
        raise RuntimeError(
            f"Katongole reference CSV not found at {_KATONGOLE_CSV}. "
            f"Run from a clean clone of the repository — the file is tracked "
            f"in git under data/external_references/."
        )
    df = pd.read_csv(_KATONGOLE_CSV)
    if df.empty:
        raise RuntimeError(f"{_KATONGOLE_CSV} is empty.")
    expected_cols = {"location", "latitude", "longitude"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{_KATONGOLE_CSV} is missing required columns: {sorted(missing)}."
        )
    _log.info("Loaded %d Katongole stations from %s", len(df), _KATONGOLE_CSV.name)
    return df[["location", "latitude", "longitude"]].copy()


def _check_cams_email_present() -> None:
    """Verify CAMS_EMAIL is set; CamsClient would otherwise prompt interactively."""
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
    stations_df: pd.DataFrame,
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
        loc = LocationSpec(
            name=str(row.location),
            lat=float(row.latitude),
            lon=float(row.longitude),
        )
        plan = NamedLocationsPlan(
            source=source,
            date_range=DateRange(start=_VAL_START, end=_VAL_END),
            locations=(loc,),
            variables=catalog_vars,
        )
        _log.info(
            "[%s] %s — %s..%s, %d vars",
            source.value, loc.name, _VAL_START, _VAL_END, len(catalog_vars),
        )
        if dry_run:
            stations_processed += 1
            continue
        try:
            result = job.run(plan)
        except Exception:
            _log.exception(
                "[%s] %s — fetch failed; continuing with remaining stations.",
                source.value, loc.name,
            )
            continue
        rows_added += result.rows_added
        api_calls += result.api_calls_made
        stations_processed += 1
        _log.info(
            "[%s] %s — rows_added=%d, api_calls=%d",
            source.value, loc.name, result.rows_added, result.api_calls_made,
        )

    return rows_added, api_calls, stations_processed


def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult:
    refs = TableRefs(config=bq.config)
    stations_df = _load_katongole_stations()

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
                f"covering {_VAL_START}..{_VAL_END}. No API calls or writes."
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
                "Ingest NASA POWER + CAMS at the 54 Katongole 2023 stations "
                "over the 2017-2022 climatology window."
            ),
        )
    )
