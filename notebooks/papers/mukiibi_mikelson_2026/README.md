# Mukiibi & Mikelson (2026) — recomputation

This subfolder will host the notebook(s) that **recompute** the analysis
published in:

> Mukiibi, R. & Mikelson, J. (2026). *A Machine Learning Approach for GHI
> Bias Correction: Validation of Random Forest Performance Across
> Sub-Saharan Africa.* IEEE [conference forthcoming].

The paper was developed against an earlier ad-hoc codebase. The goal here
is to reproduce its headline numbers (RMSE 0.57 kWh/m²/day, 53% reduction
over CERES; IOA 0.60) on top of the refactored SuSSE library, so we can
keep iterating on the model without losing the ability to point back at
the published baseline.

## Layout

```
mukiibi_mikelson_2026/
├── README.md                          # this file
├── reference_data/
│   └── katongole_2023_monthly.csv     # 55-station × 12-month climatology
└── 01_recomputation.ipynb             # (to be added)
```

## Plan

The recomputation notebook will:

1. Load the 24 training stations from `ground_measurements`. The paper
   holds out East Africa Sites A and B as the test set (the remaining 22
   sites train the model).
2. Build the 39-predictor `FeatureSpec` matching the paper's Table II.
   Most predictors come from NASA POWER (which serves CERES SYN1deg +
   GMAO MERRA-2 fields) and CAMS — both already in the warehouse.
   Derived features: `kt_cams` (`ClearSkyIndexFeature`), day-of-year
   sin/cos (`CyclicalDayOfYearFeature`), altitude (`AltitudeFeature`),
   and `LongitudeFeature` (paper-faithful, see its docstring re the
   ~28-station memorisation risk).
3. Fit `RandomForest` with the paper's hyperparameters
   (`n_estimators=200, min_samples_leaf=5, random_state=42`) and report
   the daily-scale held-out test metrics — should match the paper's
   Table III row.
4. Validate against the 54 Katongole stations in
   `reference_data/katongole_2023_monthly.csv`. Aggregate daily
   predictions to monthly means; compute RMSE / nRMSE / MAE / nMAE / MBE
   / R² / IOA per station and report the cross-station means as the
   paper's Table IV. **Validation year deviation:** the paper used 2021
   model predictions; we use 2024 (already in the warehouse). Both years
   sit inside the Katongole 2017–2022 climatology window, so the
   year-mismatch is symmetric — but the absolute numbers will differ
   slightly from the paper. This is documented at the top of the notebook.
5. Reproduce the paper's Figure 2 (16-site monthly comparison) and
   Figure 3 (RF feature importance plot).

## Reference data

`reference_data/katongole_2023_monthly.csv` (55 stations × 16 columns):
the **2017–2022 monthly-mean GHI climatology** extracted from figures 3a–3d
of Katongole et al. (2023, *Tanzania Journal of Science*). Used as
out-of-distribution validation for the bias-corrected model. The paper
itself does not distribute the underlying data; values were transcribed
from the figures.

| Column | Notes |
|---|---|
| `location` | Station name as given in the paper. |
| `latitude`, `longitude` | Decimal degrees. |
| `altitude` | Metres above sea level, **as reported in the paper**. Useful as a free cross-check on the SRTM-based `ElevationProvider`. |
| `Jan` … `Dec` | Monthly-mean GHI in **kWh/m²/day**, **averaged over 2017–2022** (7-year climatology, NOT a single year). |
| `Av` | Annual mean (mean of the 12 monthly values). |

A few of the 55 stations geohash5-collide with our 28 training stations —
the recomputation notebook flags those and reports metrics with and
without them.
