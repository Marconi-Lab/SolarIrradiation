-- Long-format daily CAMS variables.
-- Mirrors nasa_daily_vars_long: one row per (date, point, variable, source).
-- Created idempotently so this DDL is safe to re-run during deployment.

CREATE TABLE IF NOT EXISTS `solar-irradiation-estimation.solar_warehouse.cams_daily_vars_long` (
    date         DATE,
    latitude     FLOAT64,
    longitude    FLOAT64,
    geog         GEOGRAPHY,
    geohash5     STRING,
    variable_id  STRING,
    value        FLOAT64,
    source       STRING
)
PARTITION BY date
CLUSTER BY geohash5, variable_id;
