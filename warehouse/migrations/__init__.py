"""One-shot warehouse migrations.

Each script in this package modifies BigQuery state (creates a table,
backfills rows, drops a relic, runs an ingest). Scripts are dated and
numbered so order is obvious from a filesystem listing, and idempotent
so re-running them is safe.

Conventions are documented in ``README.md`` next to this file.
"""
