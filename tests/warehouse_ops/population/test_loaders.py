"""Tests for the MergeLoader column-validation contract.

These tests cover the loader's input-validation logic; the actual MERGE
SQL execution requires a live BigQuery and is exercised by the notebook
+ jobs end-to-end against the real warehouse.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from susse.warehouse_ops.population.loaders import (
    DerivedColumn,
    MergeLoader,
    MergeSpec,
)
from susse.warehouse_ops.io.config import TableSchemas


class _FakeBQ:
    """Stand-in for BigQueryClient that records calls without executing."""

    def __init__(self) -> None:
        self.loaded: list[tuple[pd.DataFrame, str]] = []
        self.ddl_statements: list[str] = []

    def load_dataframe(self, df: pd.DataFrame, table_fqn: str, **_kwargs) -> None:
        self.loaded.append((df.copy(), table_fqn))

    def execute_ddl(self, ddl: str) -> None:
        self.ddl_statements.append(ddl)


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
        df = pd.DataFrame([{
            "date": date(2024, 1, 1), "location": "x", "geog": "POINT(0 0)",
        }])
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
        df = pd.DataFrame([{
            "date": date(2024, 1, 1),
            "location": "kampala",
            "lat": 0.333542,
            "lon": 32.56863,
        }])
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
