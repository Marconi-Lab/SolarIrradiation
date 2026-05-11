# Library architecture

Reference for the SuSSE warehouse-population code. The user-facing
walkthrough lives in [`notebooks/tutorial/01_warehouse_population.ipynb`](../notebooks/tutorial/01_warehouse_population.ipynb);
this document is for engineers who need to extend or debug the library
itself.

## Three layers, top-down

```
┌──────────────────────────────────────────────────────────────────────────┐
│ External sources                                                         │
│   NASA POWER (REST/JSON)    CAMS (pvlib)    MERRA-2 (OPeNDAP)            │
│   Ground CSVs (disk/GCS)                                                 │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │  raw fetch
┌─────────────────────────────────────▼────────────────────────────────────┐
│ susse.api_clients/   — raw I/O, source-specific protocols                │
│   NASAPowerFetchData    CAMSClient    MerraDailyFetcher                  │
│   GroundSourceAdapter (StandardCsvAdapter, …)                            │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │  long-format DataFrame
┌─────────────────────────────────────▼────────────────────────────────────┐
│ susse.warehouse_ops.population/   — typed plans, jobs, loaders           │
│                                                                          │
│   Plans      NamedLocationsPlan, GridPlan, GroundFilePlan                │
│   Jobs       NasaPowerSatelliteJob, CamsSatelliteJob, MerraRegionJob,    │
│              ModisJob (BaseSatelliteJob)       GroundIngestJob           │
│   Helpers    VariableCatalog, CoverageRepository, MergeLoader,           │
│              validators                                                  │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │  MERGE via staging table
┌─────────────────────────────────────▼────────────────────────────────────┐
│ susse.warehouse_ops.io/   — BigQuery surface                             │
│   BigQueryClient, WarehouseConfig, TableSchemas, TableRefs               │
│   feature_service, repositories                                          │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │  SQL
┌─────────────────────────────────────▼────────────────────────────────────┐
│ BigQuery — solar-irradiation-estimation.solar_warehouse                  │
│   nasa_daily_vars_long, cams_daily_vars_long, merra_daily_vars_long,     │
│   irradiance_daily, ground_measurements, ground_measurements_raw,        │
│   dim_variable                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## What each layer does

**`susse.api_clients/`** — knows about specific external services.
`NASAPowerFetchData` builds NASA's URL, batches requests under their
20-vars-per-call limit, parses their JSON. `CAMSClient` wraps
`pvlib.iotools.get_cams` and handles the SoDa email handshake.
`MerraDailyFetcher` authenticates with NASA Earthdata, fetches MERRA-2
hourly data via OPeNDAP, and aggregates to daily values via
cosine-zenith weighting. Each `GroundSourceAdapter` parses one
provider's CSV layout. Replace one of these classes when a vendor
changes their API; nothing higher up has to know.

**`susse.warehouse_ops.population/`** — the heart of ingest. A
`FetchPlan` (one of three subclasses) describes *what* to fetch. A
`Job` runs the plan: pre-checks coverage to skip already-loaded rows,
calls the api_client for the missing slice, validates and reshapes the
result, then hands off to a `MergeLoader` for an idempotent server-side
MERGE. `VariableCatalog` is the single source of truth for what
variables exist and where they live in the warehouse.

**`susse.warehouse_ops.io/`** — the BigQuery surface.
`BigQueryClient` is a thin wrapper around `google-cloud-bigquery`.
`WarehouseConfig` and `TableRefs` are the only place the project ID and
table names are spelled out — every other module imports symbols from
here. `feature_service` and `repositories` are read-side helpers used
by downstream notebooks (02 onwards) to assemble training datasets.

## Three architectural rules

These rules guide the layering and prevent drift:

1. **Orthogonality.** A `Job` doesn't import from a specific api_client;
   it accepts a plan and dispatches by `Source`. An api_client doesn't
   know about BigQuery; it returns a DataFrame. The library layer
   mediates between the two. Swapping NASA POWER's API or BigQuery's
   client library is a localised edit, not a refactor.

2. **Idempotency at every boundary.** Every MERGE upserts on the
   schema's declared keys, every coverage check answers "what's already
   here?" before any API call, every script in `warehouse/migrations/`
   is safe to re-run. Ctrl+C is always safe.

3. **Typed configs over string dicts.** `WarehouseConfig`,
   `VariableSpec`, `TableSchema`, the `FetchPlan` subclasses,
   `MergeSpec` are all frozen dataclasses. A misspelled key fails at
   construction time, not as a silent KeyError three minutes into an
   ingest run.

## Operational practices

### Migrations vs notebook cells

Two distinct workflows touch the warehouse:

* **Reference / debugging** lives in `notebooks/tutorial/01_warehouse_population.ipynb`.
  Read-only by default. Useful for "did this work?", "what does this
  plan look like?", "which station has zero pairs?".
* **One-shot operational changes** live in `warehouse/migrations/*.py`.
  Each script does one well-defined warehouse mutation: create a table,
  backfill rows, run a multi-station ingest. Idempotent — re-running is
  a no-op. Versioned in git, dated by filename.

When you want to run a multi-station ingest, **don't flip
`RUN_INGEST=True` and rerun the whole notebook**. Write or invoke a
migration. The [`warehouse/migrations/README.md`](../warehouse/migrations/README.md)
documents the convention and indexes every migration that's been
applied.

Rule of thumb: if it changes BigQuery state and you'd ever want to
re-run it the same way, it's a migration.

### `dim_variable` is the canonical catalog

The `VariableCatalog` class lists every satellite variable SuSSE
ingests, with metadata (units, valid range, spatial resolution,
description) that mirrors the `dim_variable` table's schema. To add a
new variable, add a `VariableSpec` to the catalog and run
`populate_dim_variable()` — typically wrapped in a migration alongside
the data ingest that needs the new variable. Ingest jobs and downstream
feature assembly pick it up without further changes.

### Idempotency contract

Every operation in the system has the same property: re-running
produces the same end state and only does work for what's missing. This
is the foundation that makes long-running ingests safe to interrupt.

* `MergeLoader` upserts on `TableSchema.merge_keys`. A second MERGE
  with the same staging data updates existing rows in place; no
  duplicates. For long date ranges that exceed BigQuery's
  4,000-partitions-per-DML limit, the loader chunks the MERGE
  automatically.
* `BaseSatelliteJob.run` calls `CoverageRepository.existing_long_keys`
  and `existing_irradiance_keys` before fetching, scoped to the plan's
  geohashes. Locations whose data is already cached skip the API call
  entirely.
* Migration scripts use `CREATE TABLE IF NOT EXISTS` and pre-condition
  checks; a no-op re-run is a normal outcome.

If you ever find a code path that *isn't* idempotent, treat it as a
bug.
