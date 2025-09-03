CREATE OR REPLACE VIEW `solar-irradiation-estimation.solar_warehouse.v_irr_weekly_by_point` AS
SELECT
  source,
  latitude,
  longitude,
  DATE_TRUNC(date, WEEK(MONDAY)) AS week_start,
  AVG(ghi_kwh_m2_day) AS ghi_mean,
  STDDEV_SAMP(ghi_kwh_m2_day) AS ghi_std,
  COUNT(*) AS days,
  AVG(reliability) AS reliability_mean
FROM `solar-irradiation-estimation.solar_warehouse.irradiance_daily`
GROUP BY source, latitude, longitude, week_start;
