"""Tests for the shared satellite-load helpers.

The MERGE-load wrappers are exercised end-to-end by the per-job tests
(with stubbed BigQuery); this file pins the one pure, bug-prone helper —
the long → wide irradiance pivot — in isolation.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pygeohash

from susse.warehouse_ops.population.jobs.satellite_loading import (
    irradiance_long_to_wide,
)


def _irradiance_row(variable_id: str, value: float) -> dict:
    gh = pygeohash.encode(0.5, 32.5, 5)
    return {
        "date": date(2024, 1, 1),
        "latitude": 0.5,
        "longitude": 32.5,
        "geohash5": gh,
        "source": "NASA",
        "variable_id": variable_id,
        "value": value,
    }


class TestIrradianceLongToWide:
    def test_pivots_all_three_bands(self) -> None:
        long_df = pd.DataFrame(
            [
                _irradiance_row("ghi", 5.0),
                _irradiance_row("dhi", 2.0),
                _irradiance_row("dni", 6.5),
            ]
        )
        wide = irradiance_long_to_wide(long_df)
        assert len(wide) == 1
        row = wide.iloc[0]
        assert row["ghi_kwh_m2_day"] == 5.0
        assert row["dhi_kwh_m2_day"] == 2.0
        assert row["dni_kwh_m2_day"] == 6.5

    def test_absent_band_and_reliability_are_null(self) -> None:
        # Only ghi present → dhi/dni columns must still exist (NULL), as must
        # reliability (no catalogue source publishes a reliability series).
        wide = irradiance_long_to_wide(pd.DataFrame([_irradiance_row("ghi", 5.0)]))
        row = wide.iloc[0]
        assert row["ghi_kwh_m2_day"] == 5.0
        assert pd.isna(row["dhi_kwh_m2_day"])
        assert pd.isna(row["dni_kwh_m2_day"])
        assert pd.isna(row["reliability"])

    def test_one_row_per_date_cell(self) -> None:
        # Two days, one cell → two wide rows.
        rows = []
        for day in (date(2024, 1, 1), date(2024, 1, 2)):
            r = _irradiance_row("ghi", 5.0)
            r["date"] = day
            rows.append(r)
        wide = irradiance_long_to_wide(pd.DataFrame(rows))
        assert len(wide) == 2
