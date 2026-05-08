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
