"""Shared helpers for warehouse migration scripts.

A migration script is a self-contained Python file that mutates the
BigQuery warehouse in one well-defined way (create a table, backfill
rows, drop a relic, run an ingest). Migrations are idempotent: running
the same migration twice should be a no-op the second time.

This module collects the boilerplate that every migration needs:

* A standard logging setup, so each script's output is uniform.
* A factory for the :class:`susse.warehouse_ops.io.bq.BigQueryClient`,
  defaulting to the production warehouse but overridable for tests.
* A small CLI runner (:func:`run_migration`) that wraps a migration
  callable in dry-run / commit semantics and emits a structured summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Callable

from susse.warehouse_ops.io.bq import BigQueryClient
from susse.warehouse_ops.io.config import WarehouseConfig

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Standard logger setup for migration scripts.

    Idempotent: re-applying the basic config simply replaces handlers.
    """
    logging.basicConfig(level=level, format=_LOG_FORMAT, force=True)


def make_bq_client(
    *,
    project_id: str | None = None,
    dataset: str | None = None,
) -> BigQueryClient:
    """Construct a :class:`BigQueryClient` pointed at the production warehouse.

    The defaults match :class:`WarehouseConfig`. Override ``project_id`` /
    ``dataset`` to run a migration against a sandbox.
    """
    config = WarehouseConfig(
        project_id=project_id or WarehouseConfig.__dataclass_fields__["project_id"].default,
        dataset=dataset or WarehouseConfig.__dataclass_fields__["dataset"].default,
    )
    return BigQueryClient(config=config)


@dataclass(frozen=True)
class MigrationResult:
    """Structured summary returned by a migration callable.

    Attributes:
        migration_id: Filename stem (e.g. ``2026-05-08_a1_populate_dim_variable``).
        rows_affected: How many rows were written, updated, or deleted. Zero
            is a legitimate value when the migration was a no-op (already
            applied, idempotency win).
        notes: Short human-readable context for the log line — what the
            migration actually did this run.
    """

    migration_id: str
    rows_affected: int
    notes: str

    def summary_line(self) -> str:
        return (
            f"[{self.migration_id}] rows_affected={self.rows_affected} "
            f"— {self.notes}"
        )


MigrationFn = Callable[[BigQueryClient, bool], MigrationResult]
"""Signature each migration must implement: ``fn(bq, dry_run) -> MigrationResult``."""


def run_migration(
    migration_id: str,
    fn: MigrationFn,
    *,
    description: str,
) -> int:
    """CLI entrypoint shared by every migration script.

    Parses ``--dry-run`` / ``--apply`` / ``--project`` / ``--dataset``
    from argv, configures logging, builds the client, invokes ``fn``,
    and prints a structured summary. Returns a process exit code.

    Migrations should call this from their ``__main__`` block:

    .. code-block:: python

        if __name__ == "__main__":
            sys.exit(run_migration(
                migration_id="2026-05-08_a1_populate_dim_variable",
                fn=migrate,
                description="Populate dim_variable with the full 35-row catalog.",
            ))
    """
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the migration but do not write to BQ. Prints what would happen.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration to the warehouse.",
    )
    parser.add_argument("--project", default=None, help="GCP project id override.")
    parser.add_argument("--dataset", default=None, help="BQ dataset override.")
    args = parser.parse_args()

    configure_logging()
    log = logging.getLogger(migration_id)

    bq = make_bq_client(project_id=args.project, dataset=args.dataset)
    log.info(
        "Starting migration on %s.%s (mode=%s)",
        bq.config.project_id,
        bq.config.dataset,
        "dry-run" if args.dry_run else "apply",
    )

    try:
        result = fn(bq, args.dry_run)
    except Exception:
        log.exception("Migration failed.")
        return 1

    log.info(result.summary_line())
    return 0
