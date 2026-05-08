"""
BigQuery warehouse access layer for the Solar Irradiation project.
-----------------------------------------------------------------

Goal
====
Provide a thin, typed, *no-SQL-at-callsite* interface to fetch:
- Ground measurements (curated)
- Satellite irradiance (NASA/CAMS) daily aggregates
- NASA daily auxiliary variables (long format → wide)
- Assembled training pairs (ground ↔ satellite) with optional NASA vars
- Inference features for a given (lat, lon, date[, window])

Design
======
- Users interact with small repository/service classes; SQL is encapsulated.
- Config objects (dataclasses) declare project/dataset/table names and options.
- Convenience methods accept high-level filters (date ranges, locations, sources).
- Returns pandas.DataFrame; internal code uses the BigQuery Python client.

Notes
=====
- This module is self-contained; you can later split it into packages
  (io/config.py, io/bq.py, io/repositories.py, io/feature_service.py).
- Assumes tables already created as discussed:
  - `solar_warehouse.ground_measurements` (curated, month-partitioned)
  - `solar_warehouse.irradiance_daily` (existing)
  - `solar_warehouse.nasa_daily_vars_long` (existing long-format NASA vars)

Author: (c) 2025
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Final
from datetime import date

import pandas as pd
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ID: Final[str] = "solar-irradiation-estimation"
DATASET: Final[str] = "solar_warehouse"

@dataclass(frozen=True)
class TableRefs:
    nasa_daily_vars_long: str = f"{PROJECT_ID}.{DATASET}.nasa_daily_vars_long"
    cams_daily_vars_long: str = f"{PROJECT_ID}.{DATASET}.cams_daily_vars_long"
    irradiance_daily: str = f"{PROJECT_ID}.{DATASET}.irradiance_daily"
    ground_measurements: str = f"{PROJECT_ID}.{DATASET}.ground_measurements"
    ground_measurements_raw: str = f"{PROJECT_ID}.{DATASET}.ground_measurements_raw"

    # Nearest-point helper table functions
    fn_nearest_point: str = f"{PROJECT_ID}.{DATASET}.fn_nearest_point"
    fn_nearest_var_daily: str = f"{PROJECT_ID}.{DATASET}.fn_nearest_var_daily"



@dataclass(frozen=True)
class WarehouseOptions:
    """Behavior flags and defaults for assembling features."""

    geohash_precision: int = 5  # matches existing tables
    include_cams: bool = True
    include_nasa: bool = True
    # how to match ground↔satellite: 'geohash' or 'nearest' (TODO: implement nearest later)
    match_strategy: str = "geohash"
