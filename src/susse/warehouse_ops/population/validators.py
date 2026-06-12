"""Schema validators for ingest pipelines.

Each validator raises :class:`ValueError` with a remediation message when a
DataFrame violates the contract its target table expects, so callers see the
exact fix at the call site (per CLAUDE.md "validation errors must contain
the remediation").
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def _check_required_columns(
    df: pd.DataFrame, required: Sequence[str], context: str
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: DataFrame is missing required columns {missing}. "
            f"Required schema: {list(required)}. "
            f"Add the missing columns before loading."
        )


def validate_long_format(df: pd.DataFrame, *, context: str = "long-format") -> None:
    """Validate the long-format satellite schema.

    Required columns: ``date, latitude, longitude, geohash5, variable_id,
    value, source``. The ``geog`` GEOGRAPHY column is intentionally NOT
    required client-side — it is computed server-side during MERGE from
    ``ST_GEOGPOINT(longitude, latitude)``.
    """

    required = (
        "date",
        "latitude",
        "longitude",
        "geohash5",
        "variable_id",
        "value",
        "source",
    )
    _check_required_columns(df, required, context)
    if df.empty:
        raise ValueError(
            f"{context}: DataFrame is empty. The fetch step returned no rows; "
            f"upstream API or grid configuration produced nothing."
        )
    if df["value"].isna().all():
        raise ValueError(
            f"{context}: every row has value=NaN. The API likely returned a "
            f"sentinel for an out-of-range query (e.g. requesting a date the "
            f"source does not yet cover)."
        )


def validate_ground_raw(df: pd.DataFrame, *, context: str = "ground-raw") -> None:
    """Validate the raw ground-measurement schema.

    Required columns: ``datetime, ghi, location, latitude, longitude``.
    Matches the existing :class:`StandardCsvAdapter` output and the
    ``ground_measurements_raw`` BQ table.
    """

    required = ("datetime", "ghi", "location", "latitude", "longitude")
    _check_required_columns(df, required, context)
    if df.empty:
        raise ValueError(f"{context}: DataFrame is empty.")
    bad_lat = df["latitude"].dropna()
    bad_lat = bad_lat[(bad_lat < -90) | (bad_lat > 90)]
    if len(bad_lat) > 0:
        raise ValueError(
            f"{context}: {len(bad_lat)} row(s) have latitude outside [-90, 90]. "
            f"Examples: {bad_lat.head(3).tolist()}. Check column ordering — "
            f"lat and lon may be swapped."
        )
    bad_lon = df["longitude"].dropna()
    bad_lon = bad_lon[(bad_lon < -180) | (bad_lon > 180)]
    if len(bad_lon) > 0:
        raise ValueError(
            f"{context}: {len(bad_lon)} row(s) have longitude outside [-180, 180]. "
            f"Examples: {bad_lon.head(3).tolist()}."
        )


def validate_ground_curated(
    df: pd.DataFrame, *, context: str = "ground-curated"
) -> None:
    """Validate the curated ground-measurement schema.

    Required columns: ``date, month, location, lat, lon, geohash5,
    ghi_wh_m2_day, ghi_kwh_m2_day, qc_level, _version, _curated_at``. The
    ``geog`` GEOGRAPHY column is computed server-side at load time.
    """

    required = (
        "date",
        "month",
        "location",
        "lat",
        "lon",
        "geohash5",
        "ghi_wh_m2_day",
        "ghi_kwh_m2_day",
        "qc_level",
        "_version",
        "_curated_at",
    )
    _check_required_columns(df, required, context)
    if df.empty:
        raise ValueError(f"{context}: DataFrame is empty.")
    if (df["ghi_kwh_m2_day"] * 1000.0 - df["ghi_wh_m2_day"]).abs().max() > 1e-3:
        raise ValueError(
            f"{context}: ghi_kwh_m2_day and ghi_wh_m2_day disagree beyond "
            f"floating-point tolerance. They must be the same value in "
            f"different units."
        )
