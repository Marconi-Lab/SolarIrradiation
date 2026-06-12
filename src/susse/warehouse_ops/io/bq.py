"""Thin BigQuery client wrapper used by repositories, loaders, and jobs.

Exposes the small surface the rest of SuSSE needs (query, load, DDL, key
existence checks for idempotency) and hides google-cloud-bigquery details
behind it. Callers construct via a :class:`WarehouseConfig` rather than
hand-rolled project strings, so swapping environments (prod vs. test
dataset) is one change in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from google.cloud import bigquery

from .config import WarehouseConfig

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BigQueryOptions:
    """Job-level overrides that don't belong on :class:`WarehouseConfig`."""

    location: str | None = None


class BigQueryClient:
    """Thin wrapper around ``google.cloud.bigquery.Client``.

    Provides exactly the operations SuSSE needs: query → DataFrame, append
    DataFrame → table, execute DDL, and a key-existence query used by
    ingest jobs to skip already-loaded rows (idempotency).
    """

    def __init__(
        self,
        config: WarehouseConfig | None = None,
        options: BigQueryOptions | None = None,
    ) -> None:
        self._config = config or WarehouseConfig()
        self._options = options or BigQueryOptions(location=self._config.location)
        self._client = bigquery.Client(
            project=self._config.project_id,
            location=self._options.location,
        )

    @property
    def config(self) -> WarehouseConfig:
        return self._config

    @property
    def native_client(self) -> bigquery.Client:
        """Escape hatch for callers that need the underlying client directly."""
        return self._client

    def query(
        self,
        sql: str,
        *,
        job_config: bigquery.QueryJobConfig | None = None,
    ) -> pd.DataFrame:
        """Run a SQL query and return the result as a DataFrame."""
        job = self._client.query(sql, job_config=job_config)
        return job.result().to_dataframe()

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_fqn: str,
        *,
        write_disposition: str = "WRITE_APPEND",
        schema: Sequence[bigquery.SchemaField] | None = None,
        autodetect: bool = True,
    ) -> None:
        """Append (or replace) a DataFrame to a BQ table via load job."""
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            schema=list(schema) if schema is not None else None,
            autodetect=autodetect,
        )
        self._client.load_table_from_dataframe(
            df, table_fqn, job_config=job_config
        ).result()

    def execute_ddl(self, ddl: str) -> None:
        """Run a DDL/DML statement that returns no rows."""
        self._client.query(ddl).result()

    def existing_keys(
        self,
        table_fqn: str,
        key_columns: Sequence[str],
        *,
        where_filters: Sequence[str] = (),
    ) -> set[tuple]:
        """Return the set of distinct key-column tuples currently in a table.

        Used by ingest jobs to determine which (key) rows are already loaded
        so they can skip the corresponding API calls. Apply ``where_filters``
        to scope the scan to the slice actually being ingested — scanning all
        ~20M rows of ``nasa_daily_vars_long`` for every job would be wasteful.

        Args:
            table_fqn: Fully-qualified table name.
            key_columns: Column names to project as the key tuple.
            where_filters: Sequence of SQL boolean expressions joined with
                ``AND``. Quote string literals yourself.

        Returns:
            ``set[tuple]`` whose tuples align with ``key_columns``. Empty if
            the table has no rows in the scoped slice.
        """
        if not key_columns:
            raise ValueError("existing_keys requires at least one key column.")

        cols = ", ".join(f"`{c}`" for c in key_columns)
        where = ""
        if where_filters:
            where = "WHERE " + " AND ".join(f"({f})" for f in where_filters)
        sql = f"SELECT DISTINCT {cols} FROM `{table_fqn}` {where}"
        df = self.query(sql)
        if df.empty:
            return set()
        return set(map(tuple, df[list(key_columns)].itertuples(index=False, name=None)))
