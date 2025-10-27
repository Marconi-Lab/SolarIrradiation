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
from typing import Any, Iterable, Optional, Sequence
from datetime import date

import pandas as pd
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableRefs:
    """Holds fully-qualified table names.

    Customize these if your dataset names differ.
    """

    project: str = "solar-irradiation-estimation"
    dataset: str = "solar_warehouse"

    # Curated ground measurements (month-partitioned)
    ground_measurements: str = (
        "`solar-irradiation-estimation.solar_warehouse.ground_measurements`"
    )

    # Daily irradiance (NASA/CAMS), already present in your warehouse
    irradiance_daily: str = (
        "`solar-irradiation-estimation.solar_warehouse.irradiance_daily`"
    )

    # NASA daily variables (long format: date, lat, lon, geohash5, variable_id, value)
    nasa_daily_vars_long: str = (
        "`solar-irradiation-estimation.solar_warehouse.nasa_daily_vars_long`"
    )

    # Optional dimension table for variable metadata
    dim_variable: str = (
        "`solar-irradiation-estimation.solar_warehouse.dim_variable`"
    )


@dataclass(frozen=True)
class WarehouseOptions:
    """Behavior flags and defaults for assembling features."""

    geohash_precision: int = 5  # matches existing tables
    include_cams: bool = True
    include_nasa: bool = True
    # how to match ground↔satellite: 'geohash' or 'nearest' (TODO: implement nearest later)
    match_strategy: str = "geohash"

# ---------------------------------------------------------------------------
# Example (manual test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke test; adapt dates/coords as needed.
    bq = BQ(project="solar-irradiation-estimation")
    tables = TableRefs()
    svc = FeatureService(bq, tables)

    # Training pairs for a short range
    df_pairs = svc.build_training_pairs(
        start=date(2024, 2, 1), end=date(2024, 2, 15),
        locations=None,
        nasa_variables=["T2M", "CLRSKY_DNI"],  # example variable_ids
    )
    print(df_pairs.head())

    # Inference features for one point/time
    features = svc.build_inference_features(
        target_date=date(2024, 2, 15), lat=-1.38214, lon=29.671499,
        nasa_variables=["T2M"],
    )
    print(features)
