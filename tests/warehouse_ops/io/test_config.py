"""Tests for the warehouse-config + table-schema registry."""

from __future__ import annotations

import pytest

from susse.warehouse_ops.io.config import (
    MatchStrategy,
    TableRefs,
    TableSchema,
    TableSchemas,
    WarehouseConfig,
    WarehouseOptions,
)


class TestWarehouseConfig:
    def test_default_construction_uses_production_values(self) -> None:
        cfg = WarehouseConfig()
        assert cfg.project_id == "solar-irradiation-estimation"
        assert cfg.dataset == "solar_warehouse"

    def test_fqn_concatenates_project_dataset_table(self) -> None:
        cfg = WarehouseConfig(project_id="proj", dataset="ds")
        assert cfg.fqn("foo") == "proj.ds.foo"


class TestTableSchema:
    def test_rejects_empty_table_id(self) -> None:
        with pytest.raises(ValueError, match="table_id must be non-empty"):
            TableSchema(table_id="", merge_keys=("date",), description="x")

    def test_rejects_empty_merge_keys(self) -> None:
        with pytest.raises(ValueError, match="must declare at least one"):
            TableSchema(table_id="t", merge_keys=(), description="x")


class TestTableSchemasRegistry:
    @pytest.mark.parametrize(
        "schema, expected_id",
        [
            (TableSchemas.NASA_DAILY_VARS_LONG, "nasa_daily_vars_long"),
            (TableSchemas.CAMS_DAILY_VARS_LONG, "cams_daily_vars_long"),
            (TableSchemas.IRRADIANCE_DAILY, "irradiance_daily"),
            (TableSchemas.GROUND_MEASUREMENTS, "ground_measurements"),
            (TableSchemas.GROUND_MEASUREMENTS_RAW, "ground_measurements_raw"),
            (TableSchemas.DIM_VARIABLE, "dim_variable"),
        ],
    )
    def test_known_table_id(self, schema: TableSchema, expected_id: str) -> None:
        assert schema.table_id == expected_id

    def test_long_format_tables_share_merge_key_shape(self) -> None:
        # Both long-format tables key on the same four columns; this property
        # is what lets one MergeLoader contract serve both.
        assert (
            TableSchemas.NASA_DAILY_VARS_LONG.merge_keys
            == TableSchemas.CAMS_DAILY_VARS_LONG.merge_keys
            == ("date", "geohash5", "variable_id", "source")
        )


class TestTableRefs:
    def test_default_refs_match_warehouse_layout(self) -> None:
        refs = TableRefs()
        assert (
            refs.nasa_daily_vars_long
            == "solar-irradiation-estimation.solar_warehouse.nasa_daily_vars_long"
        )
        assert (
            refs.cams_daily_vars_long
            == "solar-irradiation-estimation.solar_warehouse.cams_daily_vars_long"
        )
        assert (
            refs.irradiance_daily
            == "solar-irradiation-estimation.solar_warehouse.irradiance_daily"
        )

    def test_custom_config_propagates(self) -> None:
        cfg = WarehouseConfig(project_id="my-proj", dataset="my-ds")
        refs = TableRefs(config=cfg)
        assert refs.ground_measurements == "my-proj.my-ds.ground_measurements"


class TestWarehouseOptions:
    def test_default_uses_geohash_strategy(self) -> None:
        opts = WarehouseOptions()
        assert opts.match_strategy is MatchStrategy.GEOHASH
        assert opts.geohash_precision == 5

    @pytest.mark.parametrize("precision", [0, -1, 13, 100])
    def test_rejects_out_of_range_geohash_precision(self, precision: int) -> None:
        with pytest.raises(ValueError, match=r"outside \[1, 12\]"):
            WarehouseOptions(geohash_precision=precision)
