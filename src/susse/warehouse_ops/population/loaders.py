"""MERGE-loader for idempotent writes into BigQuery tables.

The loader writes a DataFrame to a staging table, then runs a server-side
MERGE that upserts into the target by the table's declared merge keys.
Re-running with overlapping data updates existing rows in place — never
creates duplicates.

For columns whose value is computed from the *staging* row rather than
carried by the client (e.g. ``geog`` = ``ST_GEOGPOINT(lon, lat)``), pass
:class:`DerivedColumn` instances in :class:`MergeSpec.derived_columns`.

When a staging frame contains a ``date`` column and spans many distinct
dates against a date-partitioned target, the loader chunks the MERGE by
date range to stay under BigQuery's 4,000-partitions-per-DML-statement
limit (see :data:`_BQ_PARTITION_LIMIT_PER_STATEMENT`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from ..io.bq import BigQueryClient
from ..io.config import TableSchema

_logger = logging.getLogger(__name__)

# BigQuery enforces a hard cap of 4,000 partitions modified per DML
# statement. We chunk well under it (3,500 days per MERGE) so a single
# multi-year ingest won't trip the limit even with leap-year edge cases
# or accidental duplicate-date rows in the staging table.
_BQ_PARTITION_LIMIT_PER_STATEMENT = 3500


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

    def __init__(self, bq: BigQueryClient, table_fqn: str, spec: MergeSpec) -> None:
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
        self._bq.load_dataframe(df, staging_fqn, write_disposition="WRITE_TRUNCATE")
        self._merge_from_staging(staging_columns=tuple(df.columns))
        return len(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_keys = [k for k in self._spec.schema.merge_keys if k not in df.columns]
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
        update_clause = ", ".join(f"`{c}` = s.`{c}`" for c in non_key_columns)
        col_list = ", ".join(f"`{c}`" for c in all_target_columns)
        val_list = ", ".join(f"s.`{c}`" for c in all_target_columns)

        # If every column is a key column, there is nothing to UPDATE on match.
        when_matched_clause = (
            f"WHEN MATCHED THEN UPDATE SET {update_clause}\n        "
            if non_key_columns
            else ""
        )

        staging_fqn = f"{self._table_fqn}_staging"

        # BigQuery rejects DDL inside multi-statement transactions, so we
        # issue create / merge / drop as three separate statements. MERGE
        # is atomic on its own; staging persists between failures but is
        # WRITE_TRUNCATE'd on the next load anyway.
        create_sql = (
            f"CREATE TABLE IF NOT EXISTS `{self._table_fqn}` AS "
            f"SELECT {staging_select} FROM `{staging_fqn}` WHERE 1=0;"
        )

        def _merge_sql(date_filter: str = "1=1") -> str:
            return f"""
            MERGE `{self._table_fqn}` t
            USING (
                SELECT {staging_select}
                FROM `{staging_fqn}`
                WHERE {date_filter}
            ) s
            ON {on_clause}
            {when_matched_clause}WHEN NOT MATCHED THEN
                INSERT ({col_list}) VALUES ({val_list});
            """

        drop_sql = f"DROP TABLE `{staging_fqn}`;"

        self._bq.execute_ddl(create_sql)
        self._merge_in_partition_chunks(
            staging_columns=staging_columns,
            staging_fqn=staging_fqn,
            merge_sql_factory=_merge_sql,
        )
        self._bq.execute_ddl(drop_sql)

    def _merge_in_partition_chunks(
        self,
        *,
        staging_columns: tuple[str, ...],
        staging_fqn: str,
        merge_sql_factory,
    ) -> None:
        """Run the MERGE, chunking by date when the staging spans many partitions.

        BigQuery rejects a single DML statement that touches more than
        ~4,000 partitions. For most loads the staging spans far fewer
        dates and a single MERGE is enough; for long-history named-location
        ingests (e.g. one Uganda station with 12 years of daily data) we
        split the MERGE into date-range chunks of at most
        :data:`_BQ_PARTITION_LIMIT_PER_STATEMENT` calendar days each.

        Tables without a ``date`` column (e.g. ``dim_variable``) cannot
        have this issue and are MERGEd in one statement.
        """
        if "date" not in staging_columns:
            self._bq.execute_ddl(merge_sql_factory())
            return

        span_df = self._bq.query(
            f"SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM `{staging_fqn}`"
        )
        if span_df.empty or pd.isna(span_df.iloc[0]["min_d"]):
            return  # Staging is empty — nothing to MERGE.

        min_d = span_df.iloc[0]["min_d"]
        max_d = span_df.iloc[0]["max_d"]
        if hasattr(min_d, "date"):
            min_d = min_d.date()
        if hasattr(max_d, "date"):
            max_d = max_d.date()

        chunks = _date_chunks(min_d, max_d, _BQ_PARTITION_LIMIT_PER_STATEMENT)
        if len(chunks) > 1:
            _logger.info(
                "MERGE for %s spans %d days from %s to %s — splitting into "
                "%d chunks of <=%d days to stay under BigQuery's "
                "partition-per-DML cap.",
                self._table_fqn,
                (max_d - min_d).days + 1,
                min_d,
                max_d,
                len(chunks),
                _BQ_PARTITION_LIMIT_PER_STATEMENT,
            )
        for start, end in chunks:
            filter_expr = (
                f"date BETWEEN DATE('{start.isoformat()}') "
                f"AND DATE('{end.isoformat()}')"
            )
            self._bq.execute_ddl(merge_sql_factory(filter_expr))


def _date_chunks(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into contiguous spans of at most ``max_days``.

    The returned spans are inclusive on both ends and cover every day in
    ``[start, end]`` exactly once.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end}).")
    if max_days < 1:
        raise ValueError(f"max_days={max_days} must be >= 1.")
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=max_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks
