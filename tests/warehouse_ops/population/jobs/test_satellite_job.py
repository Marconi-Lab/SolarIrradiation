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
    CamsSatelliteJob,
    MerraSatelliteJob,
    NasaPowerSatelliteJob,
    _location_spec_to_geopy,
)
from susse.warehouse_ops.population.types import (
    DateRange,
    LocationSpec,
    NamedLocationsPlan,
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
                "source": "NASA",
                "variable_id": "ghi",
                "value": 5.5,
            },
            {
                "date": date(2025, 1, 1),
                "latitude": 0.333,
                "longitude": 32.568,
                "geohash5": "s8p1v",
                "source": "NASA",
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


class TestLocationSpecToGeopy:
    """The NASA POWER fetcher only reads ``.latitude`` / ``.longitude``.

    Earlier the adapter wrapped a ``Point`` in ``geopy.location.Location``,
    which broke at runtime in geopy >=2.4 (``Location.__init__`` requires
    ``address`` and ``raw``). This test pins the contract: the adapter
    must return *something* with usable lat/lon attrs that round-trip the
    input values, and it must not raise.
    """

    def test_returns_object_with_lat_lon_attrs(self) -> None:
        spec = LocationSpec(name="kampala", lat=0.333542, lon=32.568630)
        adapted = _location_spec_to_geopy(spec)
        assert hasattr(adapted, "latitude"), (
            "adapter must return an object with a .latitude attribute "
            "(NASA POWER fetcher reads it)"
        )
        assert hasattr(adapted, "longitude")
        assert adapted.latitude == pytest.approx(spec.lat)
        assert adapted.longitude == pytest.approx(spec.lon)


class _RecordingBQForCoverage:
    """Stub BigQueryClient that records calls coverage helpers make.

    Returns a ``set`` populated densely enough that
    ``_location_fully_cached`` is satisfied for every location in the plan,
    so the test never reaches the API-fetch stage.
    """

    def __init__(self, full_coverage_keys_long, full_coverage_keys_irr):
        self._long_keys = full_coverage_keys_long
        self._irr_keys = full_coverage_keys_irr
        self.coverage_calls: list[dict] = []

    @property
    def config(self):
        from susse.warehouse_ops.io.config import WarehouseConfig
        return WarehouseConfig()

    def existing_keys(self, table_fqn, key_columns, *, where_filters=()):
        self.coverage_calls.append({
            "table_fqn": table_fqn,
            "key_columns": tuple(key_columns),
            "where_filters": tuple(where_filters),
        })
        # Distinguish long-table vs irradiance-table call by key tuple length.
        if len(key_columns) == 3:
            return self._long_keys
        return self._irr_keys


class TestRunScopesCoverageByLocation:
    """``BaseSatelliteJob.run`` must pass plan geohashes to the coverage
    queries. Without this scope, named-location plans whose date range
    overlaps the existing warehouse footprint pull millions of irrelevant
    rows back from BigQuery before any API call is even issued.

    Regression guard for the A6 hang.
    """

    def test_named_locations_plan_includes_geohash_filter(self) -> None:
        import pygeohash

        loc1 = LocationSpec(name="kampala", lat=0.333542, lon=32.568630)
        loc2 = LocationSpec(name="lira", lat=2.295190, lon=32.921370)
        gh1 = pygeohash.encode(loc1.lat, loc1.lon, precision=5)
        gh2 = pygeohash.encode(loc2.lat, loc2.lon, precision=5)
        date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 2))
        long_var = _temp_var()
        irr_var = _ghi_var(Source.NASA_POWER)

        # Pre-populate "warehouse" with every (date, geohash, variable) tuple
        # that the locations would need, so _location_fully_cached returns
        # True for both and the run skips fetching.
        all_dates = (date(2024, 1, 1), date(2024, 1, 2))
        full_long = {
            (d, gh, v.variable_id)
            for d in all_dates for gh in (gh1, gh2) for v in (long_var,)
        }
        full_irr = {(d, gh) for d in all_dates for gh in (gh1, gh2)}
        bq = _RecordingBQForCoverage(full_long, full_irr)

        job = NasaPowerSatelliteJob(bq)  # type: ignore[arg-type]
        plan = NamedLocationsPlan(
            source=Source.NASA_POWER,
            date_range=date_range,
            locations=(loc1, loc2),
            variables=(long_var, irr_var),
        )
        result = job.run(plan)

        # The job should have made coverage calls (one for long, one for
        # irradiance) and skipped both locations.
        assert result.api_calls_made == 0, (
            "fully-cached locations must skip the API call"
        )
        assert result.extra["skipped_locations"] == 2

        # Both coverage calls must include a geohash5 IN (...) filter
        # with the plan's geohashes.
        assert len(bq.coverage_calls) == 2
        for call in bq.coverage_calls:
            joined = " ".join(call["where_filters"])
            assert "geohash5 IN" in joined, (
                f"coverage call missing geohash5 scope: {call!r}"
            )
            assert gh1 in joined and gh2 in joined, (
                f"coverage filter must include both plan geohashes: {call!r}"
            )


class TestCamsUnitConversion:
    """CAMS via pvlib returns daily irradiance as W/m² (mean over the
    24-hour period). The warehouse stores kWh/m²/day, so the loader must
    multiply by 0.024 = 24 hours / 1000 W/kW.

    A6's first run hit production with the conversion missing, polluting
    the warehouse with values 41.7× too large. This test pins the
    contract so it can't regress.
    """

    def test_pvlib_w_per_m2_converted_to_kwh_per_m2_day(self) -> None:
        # A typical clear-sky GHI in West Africa is ~250 W/m² mean (over
        # the 24h period); the corresponding daily energy is ~6 kWh/m²/day.
        cams_var = VariableSpec(
            variable_id="ghi", source=Source.CAMS, api_code="ghi",
            display_name="GHI", unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)", description="x",
        )
        df = pd.DataFrame([
            {"timestamp": "2024-06-15T00:00:00Z", "ghi": 250.0},
            {"timestamp": "2024-06-16T00:00:00Z", "ghi": 200.0},
        ])
        long = CamsSatelliteJob._cams_dataframe_to_long(df, (cams_var,))
        assert len(long) == 2
        # 250 W/m² × 0.024 = 6.0 kWh/m²/day
        assert long["value"].iloc[0] == pytest.approx(6.0, abs=1e-9)
        assert long["value"].iloc[1] == pytest.approx(4.8, abs=1e-9)

    def test_conversion_factor_is_explicit(self) -> None:
        # Pin the factor as a class constant so other code can reference
        # it (e.g. the data-fix migration). 24 hours ÷ 1000 W/kW = 0.024.
        assert CamsSatelliteJob._W_M2_TO_KWH_M2_DAY == pytest.approx(0.024)

    def test_zero_passes_through_unchanged(self) -> None:
        # 0 W/m² → 0 kWh/m²/day; the conversion shouldn't introduce an
        # offset. (Defensive: previous bugs in similar pipelines have
        # added/subtracted constants.)
        cams_var = VariableSpec(
            variable_id="ghi_clear", source=Source.CAMS, api_code="ghi_clear",
            display_name="GHI clear", unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)", description="x",
        )
        df = pd.DataFrame([{"timestamp": "2024-06-15T00:00:00Z", "ghi_clear": 0.0}])
        long = CamsSatelliteJob._cams_dataframe_to_long(df, (cams_var,))
        assert long["value"].iloc[0] == 0.0


class TestMerraSatelliteJob:
    """The MERRA-2 job's only novel logic is api_code → variable_id mapping.

    Coverage / MERGE / idempotency are inherited from BaseSatelliteJob
    and exercised by the existing TestRunScopesCoverageByLocation test.
    """

    def _merra_var(self, variable_id: str, api_code: str) -> VariableSpec:
        return VariableSpec(
            variable_id=variable_id, source=Source.MERRA_2, api_code=api_code,
            display_name=variable_id, unit="x", native_unit="x", description="x",
        )

    def test_fetch_remaps_api_code_to_warehouse_variable_id(self) -> None:
        # Stub fetcher returns rows keyed by api_code; the job must
        # remap them to the variable_id declared in the catalog before
        # the BaseSatelliteJob loader sees the frame.
        class _StubFetcher:
            def fetch_long_for_location(self, **kwargs):
                return pd.DataFrame([
                    {"date": date(2024, 6, 21), "variable_id": "TOTEXTTAU",
                     "value": 0.42},
                    {"date": date(2024, 6, 21), "variable_id": "TOTSCATAU",
                     "value": 0.13},
                ])

        job = MerraSatelliteJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]
        variables = (
            self._merra_var("aod_550_extinction", "TOTEXTTAU"),
            self._merra_var("aod_550_scattering", "TOTSCATAU"),
        )
        df = job._fetch_long_for_location(
            location=LocationSpec(name="kampala", lat=0.333, lon=32.568),
            date_range=DateRange(start=date(2024, 6, 21), end=date(2024, 6, 21)),
            variables=variables,
        )
        assert set(df["variable_id"]) == {
            "aod_550_extinction", "aod_550_scattering"
        }, "the job must translate api codes back to warehouse variable_ids"

    def test_fetch_drops_rows_for_unknown_api_codes(self) -> None:
        # If the fetcher returns an api_code we didn't ask for (shouldn't
        # happen in practice but defensive), the row must be dropped
        # rather than written to the warehouse with a NaN variable_id.
        class _StubFetcher:
            def fetch_long_for_location(self, **kwargs):
                return pd.DataFrame([
                    {"date": date(2024, 6, 21), "variable_id": "TOTEXTTAU",
                     "value": 0.42},
                    {"date": date(2024, 6, 21), "variable_id": "GHOST_VAR",
                     "value": 999.0},
                ])

        job = MerraSatelliteJob(_StubBQ(), fetcher=_StubFetcher())  # type: ignore[arg-type]
        df = job._fetch_long_for_location(
            location=LocationSpec(name="kampala", lat=0.333, lon=32.568),
            date_range=DateRange(start=date(2024, 6, 21), end=date(2024, 6, 21)),
            variables=(self._merra_var("aod_550_extinction", "TOTEXTTAU"),),
        )
        assert list(df["variable_id"]) == ["aod_550_extinction"]
        assert 999.0 not in df["value"].values


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
