"""Coverage queries: ask the warehouse what's already loaded.

Used by ingest jobs to skip API calls for rows that are already in the
target table — the rate-limit-aware, idempotent half of the contract.
Every coverage query is scoped by the slice the job is currently
processing (date range, source, etc.) so we never scan the full
20M-row long-format table when only a month's worth of data matters.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Sequence

from .types import DateRange, Source
from ..io.bq import BigQueryClient
from ..io.config import TableSchemas

_logger = logging.getLogger(__name__)


class CoverageRepository:
    """Read-only queries that report what is already in the warehouse."""

    def __init__(self, bq: BigQueryClient) -> None:
        self._bq = bq

    def existing_long_keys(
        self,
        table_fqn: str,
        *,
        date_range: DateRange,
        source: Source,
        variable_ids: Sequence[str],
    ) -> set[tuple]:
        """Return existing ``(date, geohash5, variable_id)`` tuples.

        Scoped by date range, source, and variable list — runs over a small
        slice of the long-format table, not the whole thing. The ``source``
        filter is implicit (the caller should pass keys without it; tuples
        we return omit the source column for clarity).
        """
        var_list = ", ".join(f"'{v}'" for v in variable_ids)
        filters = [
            f"date BETWEEN DATE('{date_range.start}') AND DATE('{date_range.end}')",
            f"source = '{source.value}'",
            f"variable_id IN ({var_list})",
        ]
        # We project geohash5 + date + variable_id; source is fixed by the filter.
        return self._bq.existing_keys(
            table_fqn,
            key_columns=("date", "geohash5", "variable_id"),
            where_filters=filters,
        )

    def existing_irradiance_keys(
        self,
        table_fqn: str,
        *,
        date_range: DateRange,
        source: Source,
    ) -> set[tuple]:
        """Return existing ``(date, geohash5)`` tuples in ``irradiance_daily``."""
        filters = [
            f"date BETWEEN DATE('{date_range.start}') AND DATE('{date_range.end}')",
            f"source = '{source.value}'",
        ]
        return self._bq.existing_keys(
            table_fqn,
            key_columns=("date", "geohash5"),
            where_filters=filters,
        )

    def existing_ground_raw_keys(
        self,
        table_fqn: str,
        *,
        location: str | None = None,
    ) -> set[tuple]:
        """Return existing ``(datetime, location)`` tuples in raw ground.

        Scoped by ``location`` if provided (one ingest file usually targets
        a single named site).
        """
        filters: list[str] = []
        if location is not None:
            filters.append(f"location = '{location}'")
        return self._bq.existing_keys(
            table_fqn,
            key_columns=("datetime", "location"),
            where_filters=tuple(filters),
        )
