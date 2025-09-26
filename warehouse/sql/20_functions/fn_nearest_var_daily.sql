CREATE OR REPLACE TABLE FUNCTION `solar-irradiation-estimation.solar_warehouse.fn_nearest_var_daily`(
  p_var_id STRING,
  p_lat FLOAT64,
  p_lon FLOAT64,
  p_start DATE,
  p_end DATE,
  p_max_km FLOAT64
)
AS (
  WITH candidates AS (
    SELECT date, variable_id, value, geog
    FROM `solar-irradiation-estimation.solar_warehouse.nasa_daily_vars_long`
    WHERE variable_id = p_var_id
      AND date BETWEEN p_start AND p_end
      AND (p_max_km IS NULL OR ST_DWITHIN(geog, ST_GEOGPOINT(p_lon, p_lat), p_max_km * 1000))
  )
  SELECT *
  FROM (
    SELECT c.*, ST_DISTANCE(c.geog, ST_GEOGPOINT(p_lon, p_lat)) AS distance_m
    FROM candidates c
  )
  QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY distance_m) = 1
);
