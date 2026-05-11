"""Tests for the MergeLoader column-validation contract.

These tests cover the loader's input-validation logic; the actual MERGE
SQL execution requires a live BigQuery and is exercised by the notebook
+ jobs end-to-end against the real warehouse.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from susse.warehouse_ops.io.config import TableSchemas
from susse.warehouse_ops.population.loaders import (
    DerivedColumn,
    MergeLoader,
    MergeSpec,
    _date_chunks,
)


class _FakeBQ:
    """Stand-in for BigQueryClient that records calls without executing.

    The chunked MERGE path issues a ``SELECT MIN(date), MAX(date)`` against
    the staging table to decide how many chunks are needed. In tests we
    answer that query from the most recently loaded DataFrame so the
    derived span is consistent with what the test set up.
    """

    def __init__(self) -> None:
        self.loaded: list[tuple[pd.DataFrame, str]] = []
        self.ddl_statements: list[str] = []
        self.queries: list[str] = []

    def load_dataframe(self, df: pd.DataFrame, table_fqn: str, **_kwargs) -> None:
        self.loaded.append((df.copy(), table_fqn))

    def execute_ddl(self, ddl: str) -> None:
        self.ddl_statements.append(ddl)

    def query(self, sql: str, **_kwargs) -> pd.DataFrame:
        self.queries.append(sql)
        # Only the date-span query is expected here; answer it from the
        # most recently staged DataFrame.
        if "MIN(date)" in sql and "MAX(date)" in sql and self.loaded:
            df = self.loaded[-1][0]
            if "date" in df.columns and not df.empty:
                return pd.DataFrame(
                    [{"min_d": df["date"].min(), "max_d": df["date"].max()}]
                )
        return pd.DataFrame([{"min_d": pd.NaT, "max_d": pd.NaT}])


class TestMergeLoaderValidation:
    def test_empty_frame_is_no_op(self) -> None:
        bq = _FakeBQ()
        loader = MergeLoader(
            bq=bq,  # type: ignore[arg-type]
            table_fqn="x.y.ground_measurements",
            spec=MergeSpec(schema=TableSchemas.GROUND_MEASUREMENTS),
        )
        n = loader.load(pd.DataFrame())
        assert n == 0
        assert bq.loaded == []
        assert bq.ddl_statements == []

    def test_rejects_missing_merge_keys(self) -> None:
        # GROUND_MEASUREMENTS keys on (date, location). A frame missing
        # 'location' must fail validation before any BQ call.
        bq = _FakeBQ()
        loader = MergeLoader(
            bq=bq,  # type: ignore[arg-type]
            table_fqn="x.y.ground_measurements",
            spec=MergeSpec(schema=TableSchemas.GROUND_MEASUREMENTS),
        )
        df = pd.DataFrame([{"date": date(2024, 1, 1), "ghi_kwh_m2_day": 5.0}])
        with pytest.raises(ValueError, match="merge-key columns"):
            loader.load(df)
        assert bq.loaded == []

    def test_rejects_derived_column_already_present(self) -> None:
        # Derived columns are computed server-side; the client must not
        # ALSO carry a column with the same name.
        bq = _FakeBQ()
        loader = MergeLoader(
            bq=bq,  # type: ignore[arg-type]
            table_fqn="x.y.t",
            spec=MergeSpec(
                schema=TableSchemas.GROUND_MEASUREMENTS,
                derived_columns=(
                    DerivedColumn(name="geog", sql_expr="ST_GEOGPOINT(lon, lat)"),
                ),
            ),
        )
        df = pd.DataFrame(
            [
                {
                    "date": date(2024, 1, 1),
                    "location": "x",
                    "geog": "POINT(0 0)",
                }
            ]
        )
        with pytest.raises(ValueError, match="would derive server-side"):
            loader.load(df)


class TestMergeLoaderSql:
    def test_merge_sql_includes_derived_column(self) -> None:
        bq = _FakeBQ()
        loader = MergeLoader(
            bq=bq,  # type: ignore[arg-type]
            table_fqn="x.y.t",
            spec=MergeSpec(
                schema=TableSchemas.GROUND_MEASUREMENTS,
                derived_columns=(
                    DerivedColumn(name="geog", sql_expr="ST_GEOGPOINT(lon, lat)"),
                ),
            ),
        )
        df = pd.DataFrame(
            [
                {
                    "date": date(2024, 1, 1),
                    "location": "kampala",
                    "lat": 0.333542,
                    "lon": 32.56863,
                }
            ]
        )
        loader.load(df)

        # Staging load happened, then create / merge / drop as three
        # separate DDL statements (BQ rejects DDL inside multi-statement
        # transactions, so the loader splits them).
        assert len(bq.loaded) == 1
        assert len(bq.ddl_statements) == 3
        create_sql, merge_sql, drop_sql = bq.ddl_statements
        assert create_sql.startswith("CREATE TABLE IF NOT EXISTS `x.y.t`")
        assert "MERGE `x.y.t` t" in merge_sql
        assert "ST_GEOGPOINT(lon, lat) AS `geog`" in merge_sql
        # MERGE keys for ground_measurements are (date, location).
        assert "t.`date` = s.`date`" in merge_sql
        assert "t.`location` = s.`location`" in merge_sql
        assert drop_sql == "DROP TABLE `x.y.t_staging`;"


class TestDateChunks:
    """Helper that splits a date span into BQ-partition-safe chunks."""

    def test_single_day_one_chunk(self) -> None:
        d = date(2024, 1, 1)
        assert _date_chunks(d, d, max_days=3500) == [(d, d)]

    def test_span_below_limit_one_chunk(self) -> None:
        chunks = _date_chunks(date(2024, 1, 1), date(2024, 6, 30), max_days=3500)
        assert chunks == [(date(2024, 1, 1), date(2024, 6, 30))]

    def test_span_above_limit_splits_with_full_coverage(self) -> None:
        # 4310 days is the kampala span that triggered BQ's 4000-partition cap.
        start, end = date(2011, 4, 6), date(2023, 1, 22)
        chunks = _date_chunks(start, end, max_days=3500)
        # Each chunk no larger than the limit.
        for cs, ce in chunks:
            assert (ce - cs).days + 1 <= 3500
        # Chunks are contiguous and cover the full span.
        assert chunks[0][0] == start
        assert chunks[-1][1] == end
        for prev, curr in zip(chunks, chunks[1:]):
            from datetime import timedelta

            assert curr[0] == prev[1] + timedelta(days=1)
        # 4310 days / 3500 max → 2 chunks.
        assert len(chunks) == 2

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            _date_chunks(date(2024, 1, 31), date(2024, 1, 1), max_days=10)

    def test_rejects_zero_max_days(self) -> None:
        with pytest.raises(ValueError, match="max_days"):
            _date_chunks(date(2024, 1, 1), date(2024, 1, 31), max_days=0)


class TestMergeChunkingByPartition:
    """The MERGE must split into multiple statements when the staging
    spans more partitions than BigQuery accepts in a single DML.

    This is the regression guard for the kampala (4310-day) failure:
    one MERGE statement against a date-partitioned table can only touch
    4,000 partitions. The loader chunks under that cap automatically.
    """

    def _make_long_span_df(self, *, start: date, end: date) -> pd.DataFrame:
        """One row at start and one at end — same shape as a long-history
        load, just minimal to keep the fixture small.
        """
        return pd.DataFrame(
            [
                {
                    "date": d,
                    "geohash5": "s8p1v",
                    "variable_id": "ghi",
                    "value": 5.0,
                    "source": "NASA",
                    "latitude": 0.33,
                    "longitude": 32.57,
                }
                for d in (start, end)
            ]
        )

    def test_short_span_runs_one_merge(self) -> None:
        bq = _FakeBQ()
        loader = MergeLoader(
            bq=bq,  # type: ignore[arg-type]
            table_fqn="x.y.nasa_long",
            spec=MergeSpec(schema=TableSchemas.NASA_DAILY_VARS_LONG),
        )
        loader.load(
            self._make_long_span_df(
                start=date(2024, 1, 1),
                end=date(2024, 6, 30),
            )
        )
        merge_statements = [s for s in bq.ddl_statements if "MERGE" in s]
        assert (
            len(merge_statements) == 1
        ), "spans below the partition cap should issue a single MERGE"

    def test_kampala_span_splits_into_multiple_merges(self) -> None:
        bq = _FakeBQ()
        loader = MergeLoader(
            bq=bq,  # type: ignore[arg-type]
            table_fqn="x.y.nasa_long",
            spec=MergeSpec(schema=TableSchemas.NASA_DAILY_VARS_LONG),
        )
        # 4310 days — the exact span that originally tripped BQ's
        # 4000-partition-per-DML limit on irradiance_daily.
        loader.load(
            self._make_long_span_df(
                start=date(2011, 4, 6),
                end=date(2023, 1, 22),
            )
        )
        merge_statements = [s for s in bq.ddl_statements if "MERGE" in s]
        assert len(merge_statements) >= 2, (
            "spans above the partition cap must be split into multiple "
            "MERGE statements"
        )
        # Every chunked MERGE must filter by date BETWEEN ... so each
        # statement only touches the partitions in its own chunk.
        for sql in merge_statements:
            assert (
                "date BETWEEN DATE(" in sql
            ), f"chunked MERGE missing date filter: {sql!r}"
