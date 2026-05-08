"""Tests for warehouse_ops.population.validators."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from susse.warehouse_ops.population.validators import (
    validate_ground_curated,
    validate_ground_raw,
    validate_long_format,
)


def _valid_long_row() -> dict:
    return {
        "date": date(2025, 1, 1),
        "latitude": 0.0,
        "longitude": 32.0,
        "geohash5": "s8p1v",
        "variable_id": "temperature",
        "value": 25.0,
        "source": "NASA",
    }


class TestValidateLongFormat:
    def test_valid_frame_passes(self) -> None:
        df = pd.DataFrame([_valid_long_row()])
        validate_long_format(df)  # should not raise

    @pytest.mark.parametrize(
        "missing_col",
        ["date", "latitude", "longitude", "geohash5", "variable_id", "value", "source"],
    )
    def test_rejects_missing_required_column(self, missing_col: str) -> None:
        row = _valid_long_row()
        df = pd.DataFrame([row]).drop(columns=[missing_col])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_long_format(df)

    def test_rejects_empty_frame(self) -> None:
        df = pd.DataFrame(columns=list(_valid_long_row().keys()))
        with pytest.raises(ValueError, match="empty"):
            validate_long_format(df)

    def test_rejects_all_nan_values(self) -> None:
        row = _valid_long_row()
        row["value"] = np.nan
        df = pd.DataFrame([row])
        with pytest.raises(ValueError, match="value=NaN"):
            validate_long_format(df)


def _valid_raw_row() -> dict:
    return {
        "datetime": date(2024, 1, 28),
        "ghi": 5550.759,
        "location": "somalia_location1",
        "latitude": 3.107191,
        "longitude": 43.637153,
    }


class TestValidateGroundRaw:
    def test_valid_frame_passes(self) -> None:
        df = pd.DataFrame([_valid_raw_row()])
        validate_ground_raw(df)

    def test_rejects_missing_columns(self) -> None:
        df = pd.DataFrame([_valid_raw_row()]).drop(columns=["latitude"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_ground_raw(df)

    def test_rejects_lat_out_of_range(self) -> None:
        row = _valid_raw_row()
        row["latitude"] = 95.0  # over the pole
        df = pd.DataFrame([row])
        with pytest.raises(ValueError, match="latitude outside"):
            validate_ground_raw(df)

    def test_rejects_lon_out_of_range(self) -> None:
        row = _valid_raw_row()
        row["longitude"] = -250.0
        df = pd.DataFrame([row])
        with pytest.raises(ValueError, match="longitude outside"):
            validate_ground_raw(df)


def _valid_curated_row() -> dict:
    return {
        "date": date(2024, 1, 28),
        "month": date(2024, 1, 1),
        "location": "somalia_location1",
        "lat": 3.107191,
        "lon": 43.637153,
        "geohash5": "sbx18",
        "ghi_wh_m2_day": 5550.759,
        "ghi_kwh_m2_day": 5.550759,
        "qc_level": "auto",
        "_version": "v1",
        "_curated_at": pd.Timestamp.utcnow(),
    }


class TestValidateGroundCurated:
    def test_valid_frame_passes(self) -> None:
        df = pd.DataFrame([_valid_curated_row()])
        validate_ground_curated(df)

    def test_rejects_unit_mismatch(self) -> None:
        row = _valid_curated_row()
        row["ghi_kwh_m2_day"] = 999.0  # disagrees with ghi_wh_m2_day / 1000
        df = pd.DataFrame([row])
        with pytest.raises(ValueError, match="ghi_kwh_m2_day and ghi_wh_m2_day disagree"):
            validate_ground_curated(df)

    def test_rejects_missing_provenance_column(self) -> None:
        df = pd.DataFrame([_valid_curated_row()]).drop(columns=["_curated_at"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_ground_curated(df)
