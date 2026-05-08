"""Tests for the variable catalogue."""

from __future__ import annotations

import pytest

from susse.warehouse_ops.population.dim_variable import (
    VariableCatalog,
    variables_to_dataframe,
)
from susse.warehouse_ops.population.types import Source


class TestVariableCatalogPartitioning:
    def test_irradiance_and_auxiliary_partition_for_nasa(self) -> None:
        all_nasa = set(VariableCatalog.for_source(Source.NASA_POWER))
        irr = set(VariableCatalog.irradiance_for_source(Source.NASA_POWER))
        aux = set(VariableCatalog.auxiliary_for_source(Source.NASA_POWER))
        assert irr & aux == set()  # disjoint
        assert irr | aux == all_nasa  # cover everything

    def test_irradiance_and_auxiliary_partition_for_cams(self) -> None:
        all_cams = set(VariableCatalog.for_source(Source.CAMS))
        irr = set(VariableCatalog.irradiance_for_source(Source.CAMS))
        aux = set(VariableCatalog.auxiliary_for_source(Source.CAMS))
        assert irr & aux == set()
        assert irr | aux == all_cams

    def test_irradiance_variables_are_ghi_dhi_dni(self) -> None:
        for source in (Source.NASA_POWER, Source.CAMS):
            ids = {v.variable_id for v in VariableCatalog.irradiance_for_source(source)}
            assert ids == {"ghi", "dhi", "dni"}


class TestVariableCatalogGet:
    def test_finds_known_variable(self) -> None:
        v = VariableCatalog.get(variable_id="temperature", source=Source.NASA_POWER)
        assert v.api_code == "T2M"
        assert v.unit == "degC"

    def test_raises_keyerror_on_unknown(self) -> None:
        with pytest.raises(KeyError, match="No variable"):
            VariableCatalog.get(variable_id="not_real", source=Source.NASA_POWER)

    def test_distinguishes_same_id_across_sources(self) -> None:
        nasa_ghi = VariableCatalog.get(variable_id="ghi", source=Source.NASA_POWER)
        cams_ghi = VariableCatalog.get(variable_id="ghi", source=Source.CAMS)
        # Same variable_id, different source → different entries.
        assert nasa_ghi.api_code != cams_ghi.api_code
        assert nasa_ghi is not cams_ghi


class TestVariableSpatialResolution:
    def test_nasa_power_native_grid_is_positive(self) -> None:
        # Replaces the warehouse's existing -51.0 sentinel bug.
        for v in VariableCatalog.NASA_POWER_VARIABLES:
            assert v.spatial_resolution_km is not None
            assert v.spatial_resolution_km > 0

    def test_cams_native_grid_is_positive(self) -> None:
        for v in VariableCatalog.CAMS_VARIABLES:
            assert v.spatial_resolution_km is not None
            assert v.spatial_resolution_km > 0


class TestVariablesToDataframe:
    def test_columns_match_dim_variable_schema(self) -> None:
        df = variables_to_dataframe(VariableCatalog.all_variables())
        expected = {
            "variable_id", "source", "display_name", "unit", "native_unit",
            "description", "temporal_granularity", "spatial_resolution_km",
            "valid_min", "valid_max",
        }
        assert set(df.columns) == expected

    def test_one_row_per_variable(self) -> None:
        df = variables_to_dataframe(VariableCatalog.all_variables())
        assert len(df) == len(VariableCatalog.all_variables())

    def test_source_values_are_string(self) -> None:
        # dim_variable.source is STRING in BQ; the dataframe must serialise
        # the StrEnum members as plain strings, not enum objects.
        df = variables_to_dataframe(VariableCatalog.all_variables())
        assert df["source"].dtype == object
        assert all(isinstance(v, str) for v in df["source"])
