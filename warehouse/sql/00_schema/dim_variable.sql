-- Variable dimension table.
-- Created idempotently. Population is handled by
-- `susse.warehouse_ops.population.dim_variable.populate_dim_variable`,
-- which inserts/updates one row per (variable_id, source) tuple.

CREATE TABLE IF NOT EXISTS `solar-irradiation-estimation.solar_warehouse.dim_variable` (
    variable_id           STRING NOT NULL,
    source                STRING NOT NULL,
    display_name          STRING,
    unit                  STRING,
    native_unit           STRING,
    description           STRING,
    temporal_granularity  STRING,
    spatial_resolution_km FLOAT64,
    valid_min             FLOAT64,
    valid_max             FLOAT64
);
