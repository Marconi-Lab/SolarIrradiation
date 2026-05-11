# Mukiibi & Mikelson (2026) — recomputation

This folder reproduces the analysis published in:

> Mukiibi, R. & Mikelson, J. (2026). *A Machine Learning Approach for GHI
> Bias Correction: Validation of Random Forest Performance Across
> Sub-Saharan Africa.* IEEE (forthcoming).

The paper was developed against an earlier ad-hoc codebase. This
recomputation rebuilds the full pipeline on top of the refactored SuSSE
library, so the model can keep iterating without losing the ability to
point back at the published baseline.

The fitted bundle saved by the notebook is also the model the companion
`Irradiation_Portal` Flask app loads at startup.

## Layout

```
mukiibi_mikelson_2026/
├── README.md                       # this file
├── 01_recomputation.ipynb          # full pipeline + Katongole validation
├── _build_notebook.py              # source of truth — regenerates the .ipynb
└── reference_data/
    └── katongole_2023_monthly.csv  # 54-station × 12-month climatology
```

The notebook is regenerated from `_build_notebook.py` — direct edits to
the `.ipynb` will be overwritten the next time the script runs. The
underscore prefix marks the builder as internal; ordinary readers should
just open the notebook.

## How to read it

The notebook is self-contained: open it, run top to bottom (~5 minutes
on a workstation), and the metric tables / figures appear inline. Two
methodological deviations from the paper are called out in the
notebook's intro and again in its final section (§9 "Deviations from
the paper"):

1. **Residual target.** The model is trained on `y - sat_ghi_cams`, not
   on absolute GHI. See §3 for the rationale.
2. **Calibration ratio on validation.** The Katongole 2023 dataset uses
   a different pyranometer network from our training data; sections
   6.3–6.4 measure the cross-network offset and apply a single annual
   rescaling before computing the final metrics.

## Reference data

`reference_data/katongole_2023_monthly.csv` (54 stations × 16 columns):
the **2017–2022 monthly-mean GHI climatology** extracted from figures
3a–3d of Katongole et al. (2023, *Tanzania Journal of Science*). Used
as out-of-distribution validation for the bias-corrected model.
Katongole et al. do not distribute the underlying values; the numbers
were transcribed from the published figures.

| Column | Notes |
|---|---|
| `location` | Station name as given in the paper |
| `latitude`, `longitude` | Decimal degrees |
| `altitude` | Metres above sea level, **as reported in the paper** — a free cross-check on the SRTM-based `ElevationProvider` |
| `Jan` … `Dec` | Monthly-mean GHI in **kWh/m²/day**, averaged over 2017–2022 (7-year climatology, not a single year) |
| `Av` | Annual mean (mean of the 12 monthly values) |

A small number of Katongole stations share a 5-character geohash cell
with one of our 28 training stations. The notebook flags those rows and
reports metrics with and without them.

## Sharing W&B runs

`LOG_TO_WANDB=True` in the notebook creates a run under your default
W&B entity in the project `susse-mukiibi-mikelson-2026`. To share with
external readers, open the project on wandb.ai → **Project settings →
Privacy** and switch to **Public**. The project URL alone is then
sufficient to share — no account needed on the reader's side.
