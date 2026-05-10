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

1. Load **all 28 training stations** from `ground_measurements`.
   **Deviation:** the paper held out 2 of 24 stations spatially to
   produce Table III; we don't, because the deployment bundle going to
   the portal benefits more from the extra ~7% data than from a
   paper-internal Table III replica.
2. Build the `FeatureSpec` matching the paper's Table II predictor set.
   Most predictors come from NASA POWER (which serves CERES SYN1deg +
   GMAO MERRA-2 fields) and CAMS — both already in the warehouse.
   Derived features: `kt_cams` (`ClearSkyIndexFeature`), day-of-year
   sin/cos (`CyclicalDayOfYearFeature`), altitude (`AltitudeFeature`),
   and `LongitudeFeature` (paper-faithful, see its docstring re the
   ~28-station memorisation risk). Cleaning rules (`>12 kWh/m²/day`
   upper bound, IQR lower fence, 5% missing-year exclusion, kNN gap
   imputation) attach as `cleaners` on the same `FeatureSpec`.
3. Fit `RandomForest` with the paper's hyperparameters
   (`n_estimators=200, min_samples_leaf=5, random_state=42`). The
   `Trainer` uses a small random in-distribution holdout to populate
   `val_metrics` as a sanity check — the headline validation is §4.
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

## Sharing the W&B run

The notebook's `WANDB_PROJECT = "susse-mukiibi-mikelson-2026"` lives
under Jan's personal entity. The first run with `LOG_TO_WANDB=True`
creates the project automatically. To share with the co-author:

1. After the first run lands, open the project on wandb.ai.
2. Go to **Project settings → Privacy** and switch to **Public**.
3. Send the project URL to the co-author. They get full read access to
   runs, metrics, charts, system traces, and artifact lineage with no
   account needed.

A free Team plan would give the co-author parallel write access too,
but it requires a paid tier in our region — not worth it for an
academic recomputation. The single-writer / public-reader pattern
matches our intended workflow (Jan trains, co-author reviews).
