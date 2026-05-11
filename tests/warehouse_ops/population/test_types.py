"""Tests for the typed value objects in warehouse_ops.population.types."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from susse.warehouse_ops.population.types import (
    BoundingBox,
    DateRange,
    GridPlan,
    GridSpec,
    GroundFilePlan,
    LocationSpec,
    NamedLocationsPlan,
    Source,
    VariableSpec,
)

# ---------------------------------------------------------------------------
# DateRange
# ---------------------------------------------------------------------------


class TestDateRange:
    def test_n_days_inclusive(self) -> None:
        dr = DateRange(start=date(2025, 1, 1), end=date(2025, 1, 7))
        assert dr.n_days == 7  # both endpoints inclusive

    def test_single_day(self) -> None:
        dr = DateRange(start=date(2025, 1, 1), end=date(2025, 1, 1))
        assert dr.n_days == 1

    def test_rejects_start_after_end(self) -> None:
        with pytest.raises(ValueError, match="is after end"):
            DateRange(start=date(2025, 1, 7), end=date(2025, 1, 1))


# ---------------------------------------------------------------------------
# LocationSpec
# ---------------------------------------------------------------------------


class TestLocationSpec:
    def test_valid_construction(self) -> None:
        loc = LocationSpec(name="kampala", lat=0.333542, lon=32.56863)
        assert loc.name == "kampala"

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            LocationSpec(name="", lat=0.0, lon=0.0)

    @pytest.mark.parametrize("lat", [-91.0, 91.0, 200.0, -200.0])
    def test_rejects_out_of_range_latitude(self, lat: float) -> None:
        with pytest.raises(ValueError, match=r"outside \[-90, 90\]"):
            LocationSpec(name="x", lat=lat, lon=0.0)

    @pytest.mark.parametrize("lon", [-181.0, 181.0, 360.0])
    def test_rejects_out_of_range_longitude(self, lon: float) -> None:
        with pytest.raises(ValueError, match=r"outside \[-180, 180\]"):
            LocationSpec(name="x", lat=0.0, lon=lon)


# ---------------------------------------------------------------------------
# BoundingBox + GridSpec
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_valid_box(self) -> None:
        bbox = BoundingBox(min_lat=0.0, max_lat=1.0, min_lon=32.0, max_lon=33.0)
        assert bbox.min_lat == 0.0

    def test_rejects_inverted_lat(self) -> None:
        with pytest.raises(ValueError, match="latitude range"):
            BoundingBox(min_lat=10.0, max_lat=0.0, min_lon=0.0, max_lon=1.0)

    def test_rejects_inverted_lon(self) -> None:
        with pytest.raises(ValueError, match="longitude range"):
            BoundingBox(min_lat=0.0, max_lat=1.0, min_lon=33.0, max_lon=32.0)


class TestGridSpec:
    def test_n_points_matches_iter_count(self) -> None:
        bbox = BoundingBox(min_lat=0.0, max_lat=0.5, min_lon=32.0, max_lon=32.5)
        grid = GridSpec(bbox=bbox, lat_step=0.05, lon_step=0.05)
        assert grid.n_points == len(list(grid.iter_locations()))

    def test_grid_endpoints_inclusive(self) -> None:
        bbox = BoundingBox(min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0)
        grid = GridSpec(bbox=bbox, lat_step=0.5, lon_step=0.5)
        # min and max lat/lon both produce points → 3×3 = 9 points
        locations = list(grid.iter_locations())
        assert len(locations) == 9
        lats = sorted({loc.lat for loc in locations})
        lons = sorted({loc.lon for loc in locations})
        assert lats == [0.0, 0.5, 1.0]
        assert lons == [0.0, 0.5, 1.0]

    def test_rejects_zero_step(self) -> None:
        bbox = BoundingBox(min_lat=0.0, max_lat=1.0, min_lon=0.0, max_lon=1.0)
        with pytest.raises(ValueError, match="must be positive"):
            GridSpec(bbox=bbox, lat_step=0.0, lon_step=0.5)

    def test_grid_locations_have_unique_names(self) -> None:
        bbox = BoundingBox(min_lat=0.0, max_lat=0.1, min_lon=0.0, max_lon=0.1)
        grid = GridSpec(bbox=bbox, lat_step=0.05, lon_step=0.05)
        names = [loc.name for loc in grid.iter_locations()]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def _kampala_loc() -> LocationSpec:
    return LocationSpec(name="kampala", lat=0.333542, lon=32.56863)


def _nasa_var() -> VariableSpec:
    return VariableSpec(
        variable_id="temperature",
        source=Source.NASA_POWER,
        api_code="T2M",
        display_name="2m Temperature",
        unit="degC",
        native_unit="degC",
        description="Test",
    )


def _cams_var() -> VariableSpec:
    return VariableSpec(
        variable_id="ghi",
        source=Source.CAMS,
        api_code="ghi",
        display_name="GHI",
        unit="kWh/m^2/day",
        native_unit="Wh/m^2",
        description="Test",
    )


class TestNamedLocationsPlan:
    def test_valid_plan(self) -> None:
        plan = NamedLocationsPlan(
            source=Source.NASA_POWER,
            date_range=DateRange(start=date(2025, 1, 1), end=date(2025, 1, 7)),
            locations=(_kampala_loc(),),
            variables=(_nasa_var(),),
        )
        assert f"source={Source.NASA_POWER.value}" in plan.describe()
        assert "locations=1" in plan.describe()

    def test_rejects_empty_locations(self) -> None:
        with pytest.raises(ValueError, match="at least one location"):
            NamedLocationsPlan(
                source=Source.NASA_POWER,
                date_range=DateRange(start=date(2025, 1, 1), end=date(2025, 1, 7)),
                locations=(),
                variables=(_nasa_var(),),
            )

    def test_rejects_empty_variables(self) -> None:
        with pytest.raises(ValueError, match="at least one variable"):
            NamedLocationsPlan(
                source=Source.NASA_POWER,
                date_range=DateRange(start=date(2025, 1, 1), end=date(2025, 1, 7)),
                locations=(_kampala_loc(),),
                variables=(),
            )

    def test_rejects_source_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match the plan's source"):
            NamedLocationsPlan(
                source=Source.NASA_POWER,
                date_range=DateRange(start=date(2025, 1, 1), end=date(2025, 1, 7)),
                locations=(_kampala_loc(),),
                variables=(_cams_var(),),  # CAMS variable in NASA plan
            )


class TestGridPlan:
    def test_valid_plan(self) -> None:
        bbox = BoundingBox(min_lat=0.0, max_lat=0.5, min_lon=32.0, max_lon=32.5)
        plan = GridPlan(
            source=Source.CAMS,
            date_range=DateRange(start=date(2025, 1, 1), end=date(2025, 1, 7)),
            grid=GridSpec(bbox=bbox, lat_step=0.05, lon_step=0.05),
            variables=(_cams_var(),),
        )
        assert "CAMS" in plan.describe()
        assert "points=" in plan.describe()


class TestGroundFilePlan:
    def test_valid_plan(self, tmp_path: Path) -> None:
        # A real (empty) file on disk satisfies the existence check; the
        # adapter handles content parsing separately.
        f = tmp_path / "x.csv"
        f.write_text("datetime,ghi,location,latitude,longitude\n")
        plan = GroundFilePlan(
            source_id="test", file_path=f, adapter_id="standard-csv:test"
        )
        assert "test" in plan.describe()

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            GroundFilePlan(
                source_id="x",
                file_path=tmp_path / "nope.csv",
                adapter_id="x",
            )
