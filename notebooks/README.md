# Notebooks

Three lanes, distinct audiences:

```
notebooks/
  tutorial/      ← system overview; read 01 → 07 to learn the codebase
  inspection/    ← day-to-day model & data inspection (one question per file)
  papers/        ← per-publication validation analyses, with data co-located
```

## `tutorial/` — the 7-step refactor track

Sequential acceptance gates. Each notebook delivers one slice of the
end-to-end pipeline and consumes the previous one's output. Read in order;
re-run in order if you want to regenerate the canonical artifacts.

| # | Notebook | Status |
|---|---|---|
| 01 | `01_warehouse_population.ipynb` | shipped |
| 02 | `02_warehouse_access_and_dataset.ipynb` | shipped |
| 03 | `03_preprocessing.ipynb` | shipped |
| 04 | `04_models.ipynb` | shipped |
| 05 | `05_training.ipynb` | shipped |
| 06 | `06_evaluation.ipynb` | TBD — proper splitter abstraction + metrics class |
| 07 | `07_inference.ipynb` | TBD — portal-facing `load_bundle` + `predict(lat, lon, date)` |

The tutorial notebooks **do** train models and write artifacts (under
`data/training_snapshots/`, `data/bundles/`); long-term, training will move
into a `scripts/train.py` and the tutorial track will be slimmed down to
demonstrate the public API rather than driving production runs.

## `inspection/` — analysis surface

Day-to-day investigation: variable distributions, residual analysis, feature
importances, model-vs-model comparison. Two firm rules:

- **Read-only with respect to artifacts.** Inspection notebooks consume
  `load_bundle(...)` / `load_snapshot(...)` and `FeatureService` for
  inference. They do NOT call `Trainer.train()` and do NOT write models or
  datasets. If a notebook here needs a new fitted model, training has to
  move out of the tutorial track first.
- **One notebook = one question.** No catch-all `analysis.ipynb`.

Initial inventory (build on demand, not pre-populated):

- `variable_distributions.ipynb` — per-source histograms, missingness, station coverage
- `residual_analysis.ipynb` — model errors vs AOD / cloud / month / altitude / station
- `per_station_performance.ipynb` — MAE/R² per station, regional choropleth
- `feature_importance.ipynb` — RF importances + partial dependence
- `model_comparison.ipynb` — side-by-side scoring of multiple bundles

## `papers/` — per-publication validation

Per-paper validation analyses. Each subfolder is self-contained: data
file(s), the validation notebook, and a `README.md` documenting the source
and extraction method.

External validation datasets live **here** alongside the analysis that
consumes them — NOT in the BigQuery warehouse. The warehouse is reserved
for prefetched satellite/auxiliary data and curated training ground-truth.

Current papers:

- `katongole_2023/` — Katongole et al. 2023, *Tanzania Journal of Science*.
  55-station 7-year monthly GHI climatology, used as out-of-distribution
  validation. See the subfolder's `README.md` for the validation strategy.

## Path conventions

All notebooks (tutorial, inspection, papers) sit two levels below the repo
root, so relative paths take the shape `../../data/...`,
`../../src/...`, `../../docs/...`. When creating a new notebook, follow
that convention rather than absolute paths so the repo stays portable.
