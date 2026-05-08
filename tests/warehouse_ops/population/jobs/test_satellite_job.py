"""Tests for the source-agnostic helpers on BaseSatelliteJob.

End-to-end ``run()`` requires real CAMS/NASA API access and is exercised
by the notebook against the live warehouse. These tests cover the
internal contracts that don't.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from susse.warehouse_ops.population.jobs.satellite_job import (
    BaseSatelliteJob,
    NasaPowerSatelliteJob,
)
from susse.warehouse_ops.population.types import (
    DateRange,
    LocationSpec,
    Source,
    VariableSpec,
)


class _StubBQ:
    """Stand-in for BigQueryClient — only attribute accesses on `config`."""

    @property
    def config(self):
        from susse.warehouse_ops.io.config import WarehouseConfig
        return WarehouseConfig()


def _ghi_var(source: Source) -> VariableSpec:
    api = "ALLSKY_SFC_SW_DWN" if source is Source.NASA_POWER else "ghi"
    return VariableSpec(
        variable_id="ghi", source=source, api_code=api,
        display_name="GHI", unit="kWh/m^2/day", native_unit="kWh/m^2/day",
        description="x",
    )


def _temp_var() -> VariableSpec:
    return VariableSpec(
        variable_id="temperature", source=Source.NASA_POWER, api_code="T2M",
        display_name="Temp", unit="degC", native_unit="degC", description="x",
    )


class TestExtractIrradiance:
    """``_extract_irradiance`` pivots long → wide for irradiance_daily."""

    def test_pivots_ghi_dhi_dni(self) -> None:
        job = NasaPowerSatelliteJob(_StubBQ())  # type: ignore[arg-type]
        irradiance_vars = (
            _ghi_var(Source.NASA_POWER),
            VariableSpec(
                variable_id="dhi", source=Source.NASA_POWER,
                api_code="ALLSKY_SFC_SW_DIFF", display_name="DHI",
                unit="kWh/m^2/day", native_unit="kWh/m^2/day", description="x",
            ),
        )
        long = pd.DataFrame([
            {
                "date": date(2025, 1, 1),
                "latitude": 0.333,
                "longitude": 32.568,
                "geohash5": "s8p1v",
                "source": "NASA_POWER",
                "variable_id": "ghi",
                "value": 5.5,
            },
            {
                "date": date(2025, 1, 1),
                "latitude": 0.333,
                "longitude": 32.568,
                "geohash5": "s8p1v",
                "source": "NASA_POWER",
                "variable_id": "dhi",
                "value": 2.5,
            },
        ])
        wide = job._extract_irradiance(long, irradiance_vars)
        assert len(wide) == 1
        assert wide.loc[0, "ghi_kwh_m2_day"] == 5.5
        assert wide.loc[0, "dhi_kwh_m2_day"] == 2.5
        # DNI was not in the long input → column exists but is NA.
        assert "dni_kwh_m2_day" in wide.columns
        assert pd.isna(wide.loc[0, "dni_kwh_m2_day"])
        # Reliability is wide-only (CAMS-specific) and stays NA for NASA.
        assert pd.isna(wide.loc[0, "reliability"])


class TestLocationFullyCached:
    """The pre-fetch coverage check determines whether we skip the API call."""

    def _job(self) -> BaseSatelliteJob:
        return NasaPowerSatelliteJob(_StubBQ())  # type: ignore[arg-type]

    def test_returns_true_when_all_keys_present(self) -> None:
        job = self._job()
        date_range = DateRange(start=date(2025, 1, 1), end=date(2025, 1, 2))
        long_vars = (_temp_var(),)
        irr_vars = (_ghi_var(Source.NASA_POWER),)

        existing_long = {
            (date(2025, 1, 1), "s8p1v", "temperature"),
            (date(2025, 1, 2), "s8p1v", "temperature"),
        }
        existing_irr = {
            (date(2025, 1, 1), "s8p1v"),
            (date(2025, 1, 2), "s8p1v"),
        }
        assert job._location_fully_cached(
            "s8p1v",
            date_range=date_range,
            long_vars=long_vars,
            irradiance_vars=irr_vars,
            existing_long=existing_long,
            existing_irr=existing_irr,
        ) is True

    def test_returns_false_when_a_long_key_missing(self) -> None:
        job = self._job()
        date_range = DateRange(start=date(2025, 1, 1), end=date(2025, 1, 2))
        long_vars = (_temp_var(),)
        irr_vars = ()

        existing_long = {(date(2025, 1, 1), "s8p1v", "temperature")}  # day 2 missing
        assert job._location_fully_cached(
            "s8p1v",
            date_range=date_range,
            long_vars=long_vars,
            irradiance_vars=irr_vars,
            existing_long=existing_long,
            existing_irr=set(),
        ) is False

    def test_returns_false_when_irradiance_key_missing(self) -> None:
        job = self._job()
        date_range = DateRange(start=date(2025, 1, 1), end=date(2025, 1, 1))
        irr_vars = (_ghi_var(Source.NASA_POWER),)
        assert job._location_fully_cached(
            "s8p1v",
            date_range=date_range,
            long_vars=(),
            irradiance_vars=irr_vars,
            existing_long=set(),
            existing_irr=set(),  # missing
        ) is False


class TestRunRejectsBadPlan:
    def test_run_raises_on_wrong_plan_type(self) -> None:
        job = NasaPowerSatelliteJob(_StubBQ())  # type: ignore[arg-type]

        class FakePlan:
            def describe(self) -> str:
                return "fake"

        with pytest.raises(TypeError, match="NamedLocationsPlan or GridPlan"):
            job.run(FakePlan())  # type: ignore[arg-type]

    def test_run_raises_on_source_mismatch(self) -> None:
        from susse.warehouse_ops.population.types import NamedLocationsPlan
        job = NasaPowerSatelliteJob(_StubBQ())  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.CAMS,  # wrong source for NASA job
            date_range=DateRange(start=date(2025, 1, 1), end=date(2025, 1, 1)),
            locations=(LocationSpec(name="x", lat=0.0, lon=0.0),),
            variables=(_ghi_var(Source.CAMS),),
        )
        with pytest.raises(ValueError, match="plan.source"):
            job.run(plan)
