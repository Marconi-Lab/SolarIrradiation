# Warehouse migrations

One-shot scripts that change the state of the BigQuery warehouse.

Use this directory for any operation that:

* creates, drops, or alters a table or view;
* backfills, deletes, or rewrites rows;
* runs an ingest job that loads data from an external source.

For diagnostics or reusable read-only scripts, use `scripts/` (e.g.
`scripts/audit_warehouse.py`). Anything in this directory is expected to
mutate warehouse state.

## Conventions

### File naming

```
YYYY-MM-DD_<phase><id>_<short_slug>.py
```

* **`YYYY-MM-DD`** — the date the migration was authored. A filesystem
  listing then sorts chronologically.
* **`<phase><id>`** — a short tag tying the migration to the plan it came
  from. Phases use single letters (`a`, `b`, `c`, ...). Within a phase, ids
  count up (`a1`, `a2`, ...). The plan that grouped these migrations lives
  in conversation/PR notes; the tag makes the connection traceable.
* **`<short_slug>`** — three or four lowercase tokens summarising the
  operation, separated by underscores.

Example: `2026-05-08_a1_populate_dim_variable.py`.

### Idempotency

Re-running a migration must be safe. Either:

* the operation is a MERGE / UPSERT keyed on a primary key, so a second
  run finds no new work; or
* the script first queries existing state and exits early when the
  expected post-condition already holds (`CREATE TABLE IF NOT EXISTS`,
  `existing_keys()` checks, count-the-rows-then-skip, etc.).

A migration is allowed to be a no-op when re-run; it must not be allowed
to crash, double-write, or leave the warehouse in a partial state.

### Structure

Each migration is a single Python file. Boilerplate goes in
[`_common.py`](./_common.py). The body of a migration is one
function with the signature:

```python
def migrate(bq: BigQueryClient, dry_run: bool) -> MigrationResult: ...
```

`dry_run=True` prints what would happen without touching BigQuery.
`dry_run=False` performs the operation and returns the row count.

The migration's `__main__` block invokes `run_migration(...)`, which
parses `--dry-run` / `--apply` flags from argv and handles logging.

### Top-of-file docstring

Every migration begins with a docstring covering:

* **What** it changes (table names, schema effects).
* **Why** — the user-facing reason or which plan phase it belongs to.
* **Expected effect** — row count, runtime, API calls (so future
  collaborators know if re-running is cheap or expensive).
* **Reversal** — how to undo this change if needed. "Not reversible" is
  an acceptable answer for some operations; say so explicitly.

### Running

```bash
# Plan only:
.venv/bin/python warehouse/migrations/2026-05-08_a1_populate_dim_variable.py --dry-run

# Apply:
.venv/bin/python warehouse/migrations/2026-05-08_a1_populate_dim_variable.py --apply

# Override target dataset (useful for sandbox testing):
.venv/bin/python warehouse/migrations/2026-05-08_a1_populate_dim_variable.py \
    --apply --project my-sandbox-project --dataset solar_warehouse_test
```

## Index

| ID | File | Summary | Status |
|---|---|---|---|
| a1 | [`2026-05-08_a1_populate_dim_variable.py`](./2026-05-08_a1_populate_dim_variable.py) | Replace the 4 sentinel-bug rows in `dim_variable` with the full 41-variable NASA POWER + CAMS catalog. | applied |
| a2 | [`2026-05-08_a2_fix_typos.py`](./2026-05-08_a2_fix_typos.py) | Rename `arimass` → `airmass` in `nasa_daily_vars_long` and `dim_variable`; refresh catalog (now 42 rows incl. `solar_zenith_angle`). | applied |
| a3 | [`2026-05-08_a3_create_cams_daily_vars_long.py`](./2026-05-08_a3_create_cams_daily_vars_long.py) | Create the empty `cams_daily_vars_long` table from its DDL. | applied |
| a4 | [`2026-05-08_a4_backfill_cams_aux_uganda_2024.py`](./2026-05-08_a4_backfill_cams_aux_uganda_2024.py) | MERGE the 6 CAMS auxiliary variables (Uganda 2024) from `cams_daily_ext` into `cams_daily_vars_long` (4.31M rows). | applied |
| a5 | _(no-op, not written)_ | Investigation showed the 48 "missing" points for `evaporation_land` / `evapotranspiration_energy` are Lake Victoria water-cells. NASA POWER's `EVLAND` / `EVPTRNS` are land-only variables; the gap is correct data. | skipped |
| a6 | [`2026-05-08_a6_ingest_28_ground_stations.py`](./2026-05-08_a6_ingest_28_ground_stations.py) | Ingest NASA POWER + CAMS for 28 ground-measurement stations over each station's full ground-data date range. Requires `CAMS_EMAIL` in `.env`. Idempotent via per-(geohash5, date) coverage check. | drafted |
| a7 | [`2026-05-08_a7_drop_external_relics.py`](./2026-05-08_a7_drop_external_relics.py) | Drop the legacy `cams_daily_ext` and `nasa_daily_ext` external CSV tables. | drafted |
| a8 | _(notebook edit)_ | Add audit and pair-assertion cells to `notebooks/01_warehouse_population.ipynb` between the warehouse-tour and Pattern-1 sections. | applied |
| a9 | [`2026-05-08_a9_fix_cams_units.py`](./2026-05-08_a9_fix_cams_units.py) | Multiply CAMS rows above magnitude threshold by 0.024 in `irradiance_daily` and `cams_daily_vars_long`. Fixes a units bug where `CamsSatelliteJob` wrote raw W/m² mean values into kWh/m²/day columns. 178,372 rows corrected. | applied |
| a10 | [`2026-05-08_a10_drop_solar_zenith_angle.py`](./2026-05-08_a10_drop_solar_zenith_angle.py) | Delete the `solar_zenith_angle` row from `dim_variable`. NASA POWER doesn't serve SZA at daily resolution; the entry caused spurious re-fetches because the coverage check saw it as always-missing. | applied |
| b3 | [`2026-05-08_b3_create_merra_daily_vars_long.py`](./2026-05-08_b3_create_merra_daily_vars_long.py) | Create the empty `merra_daily_vars_long` table from its DDL. | drafted |
| b7 | [`2026-05-08_b7_populate_dim_variable_merra.py`](./2026-05-08_b7_populate_dim_variable_merra.py) | Extend `dim_variable` with the 4 MERRA-2 catalog entries. Idempotent. | drafted |
| b8 | [`2026-05-08_b8_ingest_merra_stations_and_grid.py`](./2026-05-08_b8_ingest_merra_stations_and_grid.py) | Ingest MERRA-2 for the 28 ground stations and the Uganda 2024 grid. Requires `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` in `.env`. | drafted |

(Add a new row to this table for every migration. Status starts as
`drafted`, becomes `applied` once it has run against the production
warehouse, or `superseded` if a later migration replaces it.)

## Why no `_migration_log` BigQuery table

Some migration frameworks (Alembic, Django) record applied migrations in
a dedicated table so the system can answer "which migrations has this
database seen?". We deliberately don't do that yet because:

* one production warehouse, one developer — git history is enough audit
  trail;
* every migration is idempotent, so "did this run?" is answerable by
  re-running it (no-op vs work);
* the bookkeeping table itself becomes a thing to migrate.

If we ever grow to multiple environments (prod / staging / colab snapshots
that diverge), revisit. The `MigrationResult` summary line is already
structured so it could be inserted into a `_migration_log` table later
without changing the migration scripts themselves.

## Why not Weights & Biases for warehouse provenance

W&B is great for ML experiment + dataset-version tracking — tying a
training run to the exact data snapshot it consumed. That belongs in the
training notebooks (notebook 02 onwards), not here. Warehouse migration
provenance is an infra concern; using W&B for it would force every
collaborator to authenticate against a separate service just to read git
history.
