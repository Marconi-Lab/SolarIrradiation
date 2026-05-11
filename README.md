# SuSSE — Sub-Saharan Solar Estimation

SuSSE is a Python library for **correcting satellite estimates of global
horizontal irradiance (GHI) against ground-station measurements over
Sub-Saharan Africa**. It packages the model from Mukiibi & Mikelson (2026,
IEEE, forthcoming) together with the data infrastructure needed to keep
the model retrainable, deployable, and re-validatable end-to-end.

## What it does

| Stage | Input | Output |
|---|---|---|
| **Ingest** | Daily NASA POWER + CAMS + MERRA-2 satellite/reanalysis fields and ground GHI from partner station networks | Curated BigQuery warehouse (`solar_warehouse`) |
| **Assemble** | A `FeatureSelection` (which variables, which sources, which QC level) + a date range | A `TrainingDataset` snapshot — a parquet artifact tied to a specific warehouse state |
| **Preprocess** | `TrainingDataset` + a `FeatureSpec` (cleaners + derived features) | A `PreprocessedDataset` ready to fit |
| **Train** | `PreprocessedDataset` + model `Params` | A `TrainedBundle` — a self-contained directory the inference layer can load |
| **Validate** | `TrainedBundle` + an external reference (e.g. Katongole et al. 2023) | Paper-style metric tables (RMSE / MBE / IOA / …) |

The companion Flask app `Irradiation_Portal` (separate repository) consumes
a `TrainedBundle` and serves bias-corrected GHI predictions at any
`(lat, lon, date)` over Sub-Saharan Africa.

## Repository layout

```
src/susse/                Library code (warehouse_ops, datasets, preprocessing,
                          models, training, api_clients).
tests/                    pytest suite — mirrors src/susse/ one-to-one.
notebooks/
  tutorial/               01–05 end-to-end walk-through of the pipeline.
  inspection/             Day-to-day analysis (one notebook = one question).
  papers/<paper>/         Per-publication validation, data co-located.
warehouse/
  sql/                    DDL + functions + gold views.
  migrations/             One-shot, idempotent scripts that mutate the warehouse.
data/
  ground_measurements/    Canonical raw ground-truth CSVs (tracked in git).
  bundles/                Fitted-model bundles (gitignored, regenerable).
  training_snapshots/     Parquet dataset snapshots (gitignored, regenerable).
scripts/                  Ad-hoc utilities and one-off data dumps.
docs/architecture.md      Library-internals reference for contributors.
```

## What's shipped vs planned

| Component | Status |
|---|---|
| Warehouse population (NASA POWER, CAMS, MERRA-2, MODIS, ground) | shipped |
| `FeatureService` / `FeatureSelection` / dataset snapshots | shipped |
| `FeatureSpec` cleaners + derived features (clear-sky index, day-of-year, altitude) | shipped |
| Random Forest / Linear / Mean Baseline models with the unified `Trainer` | shipped |
| `TrainedBundle` save/load + W&B artifact integration | shipped |
| Recomputation of Mukiibi & Mikelson (2026) | shipped — see `notebooks/papers/mukiibi_mikelson_2026/` |
| Tutorial 06 (evaluation: splitter abstraction + metrics class) | **planned** |
| Tutorial 07 (inference API: portal-facing `predict(lat, lon, date)`) | **planned** |
| `inspection/` notebooks (variable distributions, residual analysis, …) | **planned** — built on demand |

## Install

The recommended workflow uses a project-local virtualenv on Python 3.12.

```bash
git clone https://github.com/jansurfsdown/Solar_irradiation.git
cd Solar_irradiation
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For warehouse access you also need Application Default Credentials for
Google Cloud:

```bash
gcloud auth application-default login
gcloud config set project solar-irradiation-estimation
```

External data sources require these environment variables in `.env` at
the repo root (used only by the corresponding ingest paths — read-only
notebook usage does not need any of them):

```
CAMS_EMAIL=<your-soda-pro-registered-email>
EARTHDATA_USERNAME=<NASA Earthdata login>
EARTHDATA_PASSWORD=<NASA Earthdata password>
```

## Where to start reading

1. `notebooks/tutorial/01_warehouse_population.ipynb` — what's in the
   warehouse, how the three ingest patterns work.
2. `notebooks/tutorial/02_warehouse_access_and_dataset.ipynb` →
   `03_preprocessing.ipynb` → `04_models.ipynb` → `05_training.ipynb`
   — the assembly → training → bundle path, in order.
3. `notebooks/papers/mukiibi_mikelson_2026/01_recomputation.ipynb` — a
   complete worked example: warehouse → preprocessor → RF → Katongole
   validation, with deliberate paper-faithful choices documented inline.
4. `docs/architecture.md` — library internals, for when you need to
   extend it.

## Developer workflow

| Task | Command |
|---|---|
| Run tests with coverage | `tox -e py312` |
| Auto-format the source tree | `tox -e format` |
| Check formatting + lint + docstring coverage | `tox -e check-format` |
| Static type checking with mypy | `tox -e static-analysis` |

CI runs all three checks plus pytest on every PR. See
[.github/workflows/tox.yml](.github/workflows/tox.yml) for the
configuration.

**Branches.** Don't push to `main`. Branch as `<initials>/<short-slug>`,
e.g. `jm/add_model`. Open a pull request and request review.

**Commits.** One logical change per commit. Imperative, present-tense
subjects (`Add bias-correction RF model`, not `Added the model`).
