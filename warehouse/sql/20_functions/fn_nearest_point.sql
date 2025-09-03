CREATE OR REPLACE TABLE FUNCTION `solar-irradiation-estimation.solar_warehouse.fn_nearest_point`(
  lat FLOAT64,
  lon FLOAT64,
  start_date DATE,
  end_date DATE,
  max_km FLOAT64  -- pass NULL to disable radius filter
)
AS (
  WITH candidates AS (
    SELECT
      source, date, latitude, longitude,
      ghi_kwh_m2_day, dhi_kwh_m2_day, dni_kwh_m2_day, reliability,
      ST_DISTANCE(geog, ST_GEOGPOINT(lon, lat)) AS distance_m
    FROM `solar-irradiation-estimation.solar_warehouse.irradiance_daily`
    WHERE date BETWEEN start_date AND end_date
      AND (max_km IS NULL OR ST_DWITHIN(geog, ST_GEOGPOINT(lon, lat), max_km * 1000))
  ),
  nearest AS (
    SELECT *
    FROM candidates
    QUALIFY ROW_NUMBER() OVER (PARTITION BY source, date ORDER BY distance_m) = 1
  )
  SELECT *
  FROM nearest
  ORDER BY date, source
);
