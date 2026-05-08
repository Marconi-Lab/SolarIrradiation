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
        geohash5s: Sequence[str] | None = None,
    ) -> set[tuple]:
        """Return existing ``(date, geohash5, variable_id)`` tuples.

        Scoped by date range, source, and variable list. When ``geohash5s``
        is provided the scan is further constrained — essential when only
        a small set of locations is being processed but the date range
        overlaps the existing warehouse footprint (otherwise the query
        pulls back millions of irrelevant rows).
        """
        var_list = ", ".join(f"'{v}'" for v in variable_ids)
        filters = [
            f"date BETWEEN DATE('{date_range.start}') AND DATE('{date_range.end}')",
            f"source = '{source.value}'",
            f"variable_id IN ({var_list})",
        ]
        if geohash5s is not None:
            if not geohash5s:
                # Empty location list → no rows can match. Skip the BQ trip.
                return set()
            gh_list = ", ".join(f"'{g}'" for g in geohash5s)
            filters.append(f"geohash5 IN ({gh_list})")
        _logger.info(
            "coverage: scanning %s for date %s..%s, source=%s, %d vars%s",
            table_fqn, date_range.start, date_range.end, source.value,
            len(variable_ids),
            f", {len(geohash5s)} geohash(es)" if geohash5s is not None else "",
        )
        result = self._bq.existing_keys(
            table_fqn,
            key_columns=("date", "geohash5", "variable_id"),
            where_filters=filters,
        )
        _logger.info("coverage: %s → %d existing keys.", table_fqn, len(result))
        return result

    def existing_irradiance_keys(
        self,
        table_fqn: str,
        *,
        date_range: DateRange,
        source: Source,
        geohash5s: Sequence[str] | None = None,
    ) -> set[tuple]:
        """Return existing ``(date, geohash5)`` tuples in ``irradiance_daily``.

        Same ``geohash5s`` scope semantics as :meth:`existing_long_keys`.
        """
        filters = [
            f"date BETWEEN DATE('{date_range.start}') AND DATE('{date_range.end}')",
            f"source = '{source.value}'",
        ]
        if geohash5s is not None:
            if not geohash5s:
                return set()
            gh_list = ", ".join(f"'{g}'" for g in geohash5s)
            filters.append(f"geohash5 IN ({gh_list})")
        _logger.info(
            "coverage: scanning %s for date %s..%s, source=%s%s",
            table_fqn, date_range.start, date_range.end, source.value,
            f", {len(geohash5s)} geohash(es)" if geohash5s is not None else "",
        )
        result = self._bq.existing_keys(
            table_fqn,
            key_columns=("date", "geohash5"),
            where_filters=filters,
        )
        _logger.info("coverage: %s → %d existing keys.", table_fqn, len(result))
        return result

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
