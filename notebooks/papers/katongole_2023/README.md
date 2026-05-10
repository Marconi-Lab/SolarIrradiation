# Katongole 2023 — external validation dataset

Monthly-mean GHI climatology for 55 stations in Sub-Saharan Africa, used as
**out-of-distribution validation** for the SuSSE bias-correction model. This
data is never used for training — see
[`memory/katongole_validation_design.md`](../../../) for the design rationale.

## Source

> Katongole, J. et al. (2023). *Solar Irradiance Estimation in East Africa.*
> Tanzania Journal of Science.
> <https://www.ajol.info/index.php/tjs/article/view/244784>

The original publication does **not** distribute the underlying tabular data.
The values in `monthly_ground_truth_df.csv` were extracted from the bar charts
in figures 3a–3d of the paper (one figure panel per region of East Africa).

## What's in `monthly_ground_truth_df.csv`

55 rows × 16 columns. One row per ground station.

| Column | Notes |
|---|---|
| `location` | Station name as given in the paper. |
| `latitude`, `longitude` | Decimal degrees. |
| `altitude` | Metres above sea level, **as reported in the paper** — independent of the SRTM-based DEM lookup our model uses. Useful as a free cross-check on the `ElevationProvider` accuracy. |
| `Jan` … `Dec` | Monthly-mean GHI in **kWh/m²/day**, **averaged over 2017–2022** (7-year climatology, NOT a single year). |
| `Av` | Annual mean GHI (mean of the 12 monthly values). |

## Why it's not in the warehouse

This is a 4.6 KB static file representing a 7-year climatology — it doesn't
fit the warehouse's role of caching prefetched satellite/aux data, and there
is no honest `date` value to assign to a monthly mean of seven different
years. It lives here, alongside the notebook that consumes it, so the
provenance of the comparison is obvious.

## Validation strategy (phased)

**Phase 1** (`validation.ipynb`) — score the model's 2024 monthly-mean
predictions against this 7-year climatology. The absolute error is inflated
by inter-annual variability (~0.1–0.3 kWh/m²/day), so the headline metric is
the *relative improvement* over the raw NASA / CAMS satellites. That ratio is
robust to the year-mismatch.

**Phase 2** (deferred) — only if Phase 1 says the comparison is publishable:
run a targeted 7-year × 55-station ingest of the model's input variables, and
compute true 7-year monthly climatology of model predictions for an
apples-to-apples comparison.

## Geohash-overlap caveat

Some Katongole stations sit in geohash5 cells already covered by our 28
training stations. The model has seen the satellite features for those cells
during training, even though the targets and years differ — so its scores on
those particular stations carry a mild advantage. The validation notebook
tags each station with `seen_in_training_geohash5: bool` and reports the
metrics with and without those points.
