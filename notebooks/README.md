# Notebooks

Three lanes, distinct audiences:

```
notebooks/
  tutorial/      ← system overview; read 01 → 05 in order to learn the codebase
  inspection/    ← day-to-day model & data inspection (one question per file)
  papers/        ← per-publication validation analyses, with data co-located
```

## `tutorial/` — end-to-end pipeline walkthrough

Sequential. Each notebook delivers one slice of the pipeline and consumes
the previous one's output; read in order, re-run in order if you want to
regenerate the canonical artifacts.

| # | Notebook | Status |
|---|---|---|
| 01 | `01_data_overview.ipynb` — what's in the warehouse (tables, catalog, interactive map of every coordinate) | shipped |
| 02 | `02_query_the_warehouse.ipynb` — `FeatureSelection` → `FeatureService` → `TrainingDataset` snapshot, plus raw-SQL examples | shipped |
| 03 | `03_preprocessing.ipynb` — `FeatureSpec` → `PreprocessedDataset` | shipped |
| 04 | `04_models.ipynb` — model factory (mean baseline / RF / linear) and the `Params` pattern | shipped |
| 05 | `05_training.ipynb` — `Trainer`, `TrainedBundle`, W&B integration | shipped |
| 06 | `06_evaluation.ipynb` — `Splitter` ABC (random / temporal / station-LOSO / spatial-spread / spatial-block) + `Metric` ABC + `Evaluator` + generic eval plots | shipped |
| 07 | `07_inference.ipynb` — portal-facing `load_bundle` + `predict(lat, lon, date)` | **planned** |

The tutorial notebooks **do** train models and write artifacts (under
`data/training_snapshots/`, `data/bundles/`); they're the canonical way
to regenerate those gitignored artifacts after a fresh clone.

The warehouse-extension content (ingest patterns, RUN_INGEST=True
walkthroughs, contributor recipes) lives separately at
[`warehouse/extending_the_warehouse.ipynb`](../warehouse/extending_the_warehouse.ipynb)
because most readers will only query the warehouse, not modify it.

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

- [`mukiibi_mikelson_2026/`](papers/mukiibi_mikelson_2026/) — Mukiibi & Mikelson
  (2026, IEEE, forthcoming). Reproduces the Random Forest GHI bias-correction
  model on the refactored SuSSE library and validates against Katongole et al.
  (2023, *Tanzania Journal of Science*) 54-station monthly climatology
  (the reference CSV lives co-located inside the paper folder). The fitted
  bundle is also the model the companion `Irradiation_Portal` Flask app loads.

## Path conventions

All notebooks (tutorial, inspection, papers) sit two levels below the repo
root, so relative paths take the shape `../../data/...`,
`../../src/...`, `../../docs/...`. When creating a new notebook, follow
that convention rather than absolute paths so the repo stays portable.
