-- Daily MODIS observations from ORNL DAAC (modis.ornl.gov).
-- Each row is one (date, point, product, band) measurement. Unlike the
-- *_daily_vars_long tables, MODIS rows carry both product_id and band_id
-- because each MODIS product has multiple bands and the (product, band)
-- pair is the natural identifier of a single observation series.
--
-- Storage uses each product's native composite cadence — values appear
-- at the composite end-date (daily for MOD11A1/MOD10A1, 16-day for
-- MCD43A3/MOD13Q1). Daily upsampling is a downstream join concern; we
-- don't fabricate dates here.
--
-- Created idempotently so this DDL is safe to re-run during deployment.

CREATE TABLE IF NOT EXISTS `solar-irradiation-estimation.solar_warehouse.modis_observations` (
    date         DATE,
    latitude     FLOAT64,
    longitude    FLOAT64,
    geog         GEOGRAPHY,
    geohash5     STRING,
    product_id   STRING,
    band_id      STRING,
    value        FLOAT64,
    source       STRING
)
PARTITION BY date
CLUSTER BY geohash5, product_id, band_id;
