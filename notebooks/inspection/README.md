# Inspection notebooks

Day-to-day model & data inspection. See [../README.md](../README.md) for
the broader notebook structure.

## Current notebooks

*No notebooks here yet.* The interactive coverage map that used to live
here is now [tutorial 01](../tutorial/01_data_overview.ipynb) — it's
the first thing every reader sees, so the tutorial track is the better
home for it.

## On the backlog

Built on demand alongside actual analysis needs (in rough priority order):

- `variable_distributions.ipynb` — per-source histograms, missingness, station coverage
- `residual_analysis.ipynb` — model errors vs AOD / cloud / month / altitude / station
- `per_station_performance.ipynb` — MAE / R² per station, regional choropleth
- `feature_importance.ipynb` — RF importances + partial dependence
- `model_comparison.ipynb` — side-by-side scoring of multiple bundles

## Rules

- **Read-only with respect to artifacts.** Consume `load_bundle(...)` /
  `load_snapshot(...)` and `FeatureService` for inference. Do NOT call
  `Trainer.train()` and do NOT write models or datasets.
- **One notebook = one question.** No catch-all `analysis.ipynb`.
- **Path convention.** Relative paths take the shape `../../data/...`,
  `../../src/...`. The notebook sits at `notebooks/inspection/<name>.ipynb`,
  two levels below the repo root.
