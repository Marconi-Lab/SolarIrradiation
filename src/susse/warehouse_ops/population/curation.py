"""Pure transformation from the raw ground schema to the curated schema.

The curated schema adds geohash, geog (computed server-side at load),
units in both Wh/m²/day and kWh/m²/day, a QC level, and provenance
columns. ``curate_ground`` is a deterministic function: same input
produces the same output, no I/O, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pygeohash

from .validators import validate_ground_curated, validate_ground_raw


@dataclass(frozen=True)
class CurationOptions:
    """Configurable defaults stamped on every curated row."""

    qc_level: str = "auto"
    version: str = "v1"
    geohash_precision: int = 5

    def __post_init__(self) -> None:
        if not (1 <= self.geohash_precision <= 12):
            raise ValueError(
                f"geohash_precision={self.geohash_precision} is outside [1, 12]."
            )
        if not self.qc_level:
            raise ValueError("CurationOptions.qc_level must be non-empty.")
        if not self.version:
            raise ValueError("CurationOptions.version must be non-empty.")


def curate_ground(
    raw: pd.DataFrame, options: CurationOptions | None = None
) -> pd.DataFrame:
    """Transform a raw ground DataFrame into the curated schema.

    Schema mapping:

    * ``datetime`` → ``date`` (DATE)
    * (derived)   → ``month`` (first-of-month DATE)
    * ``location`` → ``location`` (unchanged)
    * ``latitude``  → ``lat`` (FLOAT64)
    * ``longitude`` → ``lon`` (FLOAT64)
    * (derived)   → ``geohash5`` (STRING, precision from options)
    * ``ghi``     → ``ghi_wh_m2_day`` (FLOAT64)
    * (derived)   → ``ghi_kwh_m2_day`` = ``ghi_wh_m2_day`` / 1000
    * (derived)   → ``qc_level`` (constant from options)
    * (derived)   → ``_version`` (constant from options)
    * (derived)   → ``_curated_at`` (current UTC timestamp)

    The ``geog`` GEOGRAPHY column is intentionally NOT produced here — it is
    computed server-side at load time via ``ST_GEOGPOINT(lon, lat)`` so the
    DataFrame stays JSON-serialisable and cross-environment-portable.
    """
    options = options or CurationOptions()
    validate_ground_raw(raw, context="curate_ground.input")

    out = pd.DataFrame()
    parsed_datetime = pd.to_datetime(raw["datetime"])
    out["date"] = parsed_datetime.dt.date
    out["month"] = parsed_datetime.dt.to_period("M").dt.start_time.dt.date
    out["location"] = raw["location"].astype(str)
    out["lat"] = raw["latitude"].astype(float)
    out["lon"] = raw["longitude"].astype(float)
    out["geohash5"] = [
        pygeohash.encode(lat, lon, precision=options.geohash_precision)
        for lat, lon in zip(out["lat"], out["lon"])
    ]
    out["ghi_wh_m2_day"] = raw["ghi"].astype(float)
    out["ghi_kwh_m2_day"] = out["ghi_wh_m2_day"] / 1000.0
    out["qc_level"] = options.qc_level
    out["_version"] = options.version
    out["_curated_at"] = pd.Timestamp.utcnow()

    validate_ground_curated(out, context="curate_ground.output")
    return out
