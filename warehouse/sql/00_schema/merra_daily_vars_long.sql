-- Long-format daily MERRA-2 variables.
-- Mirrors nasa_daily_vars_long and cams_daily_vars_long: one row per
-- (date, point, variable, source). MERRA-2's native temporal resolution
-- is hourly; daily values stored here are cosine-zenith-weighted means
-- aggregated client-side before MERGE.
-- Created idempotently so this DDL is safe to re-run during deployment.

CREATE TABLE IF NOT EXISTS `solar-irradiation-estimation.solar_warehouse.merra_daily_vars_long` (
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
