"""MERGE-loader for idempotent writes into BigQuery tables.

The loader writes a DataFrame to a staging table, then runs a server-side
MERGE that upserts into the target by the table's declared merge keys.
Re-running with overlapping data updates existing rows in place — never
creates duplicates.

For columns whose value is computed from the *staging* row rather than
carried by the client (e.g. ``geog`` = ``ST_GEOGPOINT(lon, lat)``), pass
:class:`DerivedColumn` instances in :class:`MergeSpec.derived_columns`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..io.bq import BigQueryClient
from ..io.config import TableSchema

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DerivedColumn:
    """A target column computed server-side from the staging row.

    Used for types that pandas can't carry across the wire (e.g. ``GEOGRAPHY``)
    or values better produced by the database (e.g. ``CURRENT_TIMESTAMP()``).
    The expression references staging columns by name.
    """

    name: str
    sql_expr: str


@dataclass(frozen=True)
class MergeSpec:
    """How to MERGE-load a DataFrame into a target table."""

    schema: TableSchema
    derived_columns: tuple[DerivedColumn, ...] = ()


class MergeLoader:
    """Loads a DataFrame into a BigQuery table via staging-table MERGE.

    The MERGE is idempotent: re-running with overlapping rows updates the
    existing target rows by the schema's ``merge_keys``; new rows are
    inserted; nothing is duplicated.
    """

    def __init__(
        self, bq: BigQueryClient, table_fqn: str, spec: MergeSpec
    ) -> None:
        self._bq = bq
        self._table_fqn = table_fqn
        self._spec = spec

    @property
    def schema(self) -> TableSchema:
        return self._spec.schema

    def load(self, df: pd.DataFrame) -> int:
        """Load ``df`` and return the number of rows staged.

        Returns 0 if ``df`` is empty (nothing staged, no MERGE issued).
        """
        if df.empty:
            return 0
        self._validate_columns(df)
        staging_fqn = f"{self._table_fqn}_staging"
        self._bq.load_dataframe(
            df, staging_fqn, write_disposition="WRITE_TRUNCATE"
        )
        self._merge_from_staging(staging_columns=tuple(df.columns))
        return len(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_keys = [
            k for k in self._spec.schema.merge_keys if k not in df.columns
        ]
        if missing_keys:
            raise ValueError(
                f"DataFrame is missing merge-key columns for "
                f"{self._spec.schema.table_id}: {missing_keys}. "
                f"Required keys: {list(self._spec.schema.merge_keys)}. "
                f"Add them to the DataFrame before calling load()."
            )
        derived_names = {d.name for d in self._spec.derived_columns}
        overlap = derived_names & set(df.columns)
        if overlap:
            raise ValueError(
                f"DataFrame already contains columns the loader would derive "
                f"server-side: {sorted(overlap)}. Drop them before loading or "
                f"remove the corresponding DerivedColumn entries."
            )

    def _merge_from_staging(self, staging_columns: tuple[str, ...]) -> None:
        merge_keys = self._spec.schema.merge_keys
        derived = self._spec.derived_columns

        all_target_columns = list(staging_columns) + [d.name for d in derived]
        non_key_columns = [c for c in all_target_columns if c not in merge_keys]

        # Staging projection: pass-through staging columns + derived expressions.
        select_parts = [f"`{c}`" for c in staging_columns] + [
            f"{d.sql_expr} AS `{d.name}`" for d in derived
        ]
        staging_select = ", ".join(select_parts)

        on_clause = " AND ".join(f"t.`{k}` = s.`{k}`" for k in merge_keys)
        update_clause = ", ".join(
            f"`{c}` = s.`{c}`" for c in non_key_columns
        )
        col_list = ", ".join(f"`{c}`" for c in all_target_columns)
        val_list = ", ".join(f"s.`{c}`" for c in all_target_columns)

        # If every column is a key column, there is nothing to UPDATE on match.
        when_matched_clause = (
            f"WHEN MATCHED THEN UPDATE SET {update_clause}\n        "
            if non_key_columns
            else ""
        )

        staging_fqn = f"{self._table_fqn}_staging"
        sql = f"""
        BEGIN TRANSACTION;
        CREATE TABLE IF NOT EXISTS `{self._table_fqn}` AS
            SELECT {staging_select} FROM `{staging_fqn}` WHERE 1=0;
        MERGE `{self._table_fqn}` t
        USING (SELECT {staging_select} FROM `{staging_fqn}`) s
        ON {on_clause}
        {when_matched_clause}WHEN NOT MATCHED THEN
            INSERT ({col_list}) VALUES ({val_list});
        DROP TABLE `{staging_fqn}`;
        COMMIT TRANSACTION;
        """
        _logger.debug("MERGE SQL for %s:\n%s", self._table_fqn, sql)
        self._bq.execute_ddl(sql)
