# Warehouse

The SuSSE warehouse is a BigQuery dataset (`solar_warehouse` in the
`solar-irradiation-estimation` project, EU region) that holds every
satellite/reanalysis variable, every ground measurement, and the
catalog metadata that ties them together. It is the single source of
truth that every notebook reads from.

## Tables at a glance

| Table | Shape | What it stores |
|---|---|---|
| `dim_variable` | catalog | One row per `(variable_id, source)` describing every variable SuSSE knows about: unit, valid range, spatial resolution, and which physical table it lives in |
| `irradiance_daily` | wide | Per-`(date, geohash5, source)` daily GHI / DHI / DNI for NASA POWER and CAMS |
| `nasa_daily_vars_long` | long | NASA POWER auxiliary variables (`temperature`, `aod_550`, `cloud_amount`, …) keyed by `(date, geohash5, variable_id)` |
| `cams_daily_vars_long` | long | CAMS auxiliary variables (`ghi_clear`, `bhi`, `dni_clear`, …) — same shape as the NASA long table |
| `merra_daily_vars_long` | long | MERRA-2 reanalysis variables (aerosol decomposition, precipitable water, …) — same shape |
| `modis_observations` | observation-keyed | MODIS imagery composites keyed by `(date, geohash5, product_id, band_id)` because MODIS doesn't fit the long-format schema |
| `ground_measurements` | curated long | Daily ground GHI per station, with QC level + curation provenance |
| `ground_measurements_raw` | uncurated long | The same data before curation; kept so the QC pipeline is re-runnable |

Wide vs long is chosen per source: GHI / DHI / DNI go in the wide
`irradiance_daily` because they're a tightly coupled trio always queried
together; everything else lives in a long-format table per source so
adding a new variable does not require a schema migration.

`dim_variable.physical_storage` tags every catalog row with which of the
three storage shapes it lives in (`long_format` / `irradiance_wide` /
`modis_observations`). `FeatureSelection` validates that requested
`variable_id`s match the storage their target field reads from — a typo
or a wide-vs-long mismatch fails at construction with a remediation
message, not as a silent NaN column at fetch time.

## Data lineage

```
                    ╔═══════════════════════════════════════════════════╗
                    ║         External sources (read-only)              ║
                    ║                                                   ║
                    ║   NASA POWER API     CAMS API    MERRA-2 OPeNDAP  ║
                    ║   MODIS (ORNL DAAC)  Partner CSVs under           ║
                    ║                      data/ground_measurements/    ║
                    ╚═══════════════════════════════════════════════════╝
                                            │
                                            │  ingest jobs (warehouse/migrations/)
                                            ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │                       Warehouse tables                             │
        │                                                                    │
        │  ┌────────────────────────────┐    ┌────────────────────────┐      │
        │  │ irradiance_daily (wide)    │◄───┤ NASA POWER             │      │
        │  │  ghi / dhi / dni per       │◄───┤ CAMS                   │      │
        │  │  (date, geohash5, source)  │    │  satellite ingest jobs │      │
        │  └────────────────────────────┘    │  (write both wide and  │      │
        │                                    │   long in parallel)    │      │
        │  ┌────────────────────────────┐    │                        │      │
        │  │ nasa_daily_vars_long       │◄───┤                        │      │
        │  └────────────────────────────┘    │                        │      │
        │  ┌────────────────────────────┐    │                        │      │
        │  │ cams_daily_vars_long       │◄───┘                        │      │
        │  └────────────────────────────┘                                    │
        │  ┌────────────────────────────┐                                    │
        │  │ merra_daily_vars_long      │◄── MerraRegionJob (OPeNDAP)        │
        │  └────────────────────────────┘                                    │
        │  ┌────────────────────────────┐                                    │
        │  │ modis_observations         │◄── ModisJob (ORNL DAAC)            │
        │  └────────────────────────────┘                                    │
        │  ┌────────────────────────────┐                                    │
        │  │ ground_measurements_raw    │◄── GroundIngestJob (CSV adapters)  │
        │  └────────────┬───────────────┘    Raw, never edited by hand.      │
        │               │                                                    │
        │               │  curate_ground (QC, unit conversion)               │
        │               ▼                                                    │
        │  ┌────────────────────────────┐                                    │
        │  │ ground_measurements        │   The table notebooks read for     │
        │  │  (curated, QC-tagged)      │   training targets.                │
        │  └────────────────────────────┘                                    │
        │                                                                    │
        │  ┌────────────────────────────┐                                    │
        │  │ dim_variable               │  Catalogue (units, valid range,    │
        │  │  (no upstream — schema)    │  spatial_resolution, physical_     │
        │  └────────────────────────────┘  storage). Hand-maintained via     │
        │                                  the VariableCatalog Python class. │
        │                                                                    │
        │  ┌────────────────────────────┐                                    │
        │  │ v_irr_weekly_by_point      │  Read-side gold views over         │
        │  │ v_irr_monthly_by_point     │  irradiance_daily.                 │
        │  └────────────────────────────┘                                    │
        └────────────────────────────────────────────────────────────────────┘
```

Two observations worth internalising before extending the warehouse:

1. **`irradiance_daily` is not derived from the `*_daily_vars_long` tables.**
   They are *siblings* — the NASA POWER and CAMS satellite ingest jobs
   write to both at the same time. GHI/DHI/DNI go to the wide
   `irradiance_daily` (where they're queried as a trio); every other
   variable goes to the matching `*_daily_vars_long` (where adding a
   new variable doesn't need a schema change).
2. **`ground_measurements` *is* derived** — it's the curated output of
   `curate_ground` over `ground_measurements_raw`. The raw table is
   write-once at ingest time and should never be hand-edited; the
   curated table is what training reads.

### If you want to add…

| Goal | Where to act | Notes |
|---|---|---|
| **A new ground station from an existing partner** | Drop the CSV under `data/ground_measurements/<provider>/<station>.csv` and run `GroundIngestJob` (Pattern 3 in NB 01) — its adapter parses the file, writes `ground_measurements_raw`, and the curation step writes `ground_measurements` | The two tables update together; you don't touch them by hand |
| **A new ground-measurement provider with a different CSV layout** | Subclass `GroundSourceAdapter` in `src/susse/warehouse_ops/population/adapters.py`; otherwise as above | Schema after the adapter must match `ground_measurements_raw`'s columns |
| **A new variable from an existing satellite source** | Add a `VariableSpec` to `VariableCatalog` (`src/susse/warehouse_ops/population/dim_variable.py`); write a migration to `populate_dim_variable`; re-run the relevant satellite ingest | The variable's `physical_storage` tag determines which table it lands in |
| **A whole new satellite source** | Add a fetcher under `src/susse/api_clients/<source>/`, a `Source` enum value, a `BaseSatelliteJob` subclass, the catalogue entries, a `TableSchema` + DDL + `CREATE TABLE` migration, and an ingest migration | The MERRA-2 wire-up (`b3` / `b7` / `b9` migrations + `MerraRegionJob`) is the worked template |
| **A read-side aggregation** | New SQL view under `warehouse/sql/10_gold/`; run via `warehouse/run_sql.py` | Views are cheap; prefer them over precomputed tables until a query is provably slow |

## Subfolders

| Path | Purpose |
|---|---|
| [`sql/00_schema/`](sql/00_schema/) | DDL for every table. New tables get a file here and a corresponding `CREATE TABLE IF NOT EXISTS` migration |
| [`sql/10_gold/`](sql/10_gold/) | Read-side gold views (e.g. weekly / monthly aggregations) that wrap common analytics queries |
| [`sql/20_functions/`](sql/20_functions/) | BigQuery user-defined functions (e.g. `fn_nearest_point`) used by the gold views and some notebooks |
| [`migrations/`](migrations/) | One-shot scripts that mutate the warehouse — create a table, backfill rows, run an ingest. Idempotent. See [`migrations/README.md`](migrations/README.md) |
| [`run_sql.py`](run_sql.py) | CLI for executing a file from `sql/` against the warehouse |

## Authentication

```bash
gcloud auth application-default login
gcloud config set project solar-irradiation-estimation
```

That single step gets you read access to every table for the notebooks.
Write paths (`warehouse/migrations/*.py`, `RUN_INGEST=True` in NB 01)
additionally need the relevant satellite-source credentials in `.env`
(`CAMS_EMAIL` / `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`).

## Reading from Python

```python
from susse.warehouse_ops.io import BigQueryClient, WarehouseConfig
from susse.warehouse_ops.io.config import TableRefs

bq = BigQueryClient(config=WarehouseConfig())
tables = TableRefs(config=bq.config)

# Any table's fully-qualified name:
print(tables.irradiance_daily)
# → solar-irradiation-estimation.solar_warehouse.irradiance_daily

df = bq.query(f"SELECT COUNT(*) FROM `{tables.irradiance_daily}`")
```

For higher-level assembly (joining ground truth with multi-source
satellite features), use `FeatureService` — see tutorial NB 02.
