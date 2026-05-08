"""Tests for ground-source adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from susse.warehouse_ops.population.adapters import StandardCsvAdapter


class TestStandardCsvAdapter:
    def test_adapter_id_includes_source(self) -> None:
        adapter = StandardCsvAdapter(source_id="CBE")
        assert adapter.adapter_id == "standard-csv:CBE"

    def test_rejects_empty_source_id(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            StandardCsvAdapter(source_id="")

    def test_parses_real_somalia_slice(self, somalia_csv_path: Path) -> None:
        adapter = StandardCsvAdapter(source_id="CBE")
        df = adapter.parse(somalia_csv_path)

        assert list(df.columns) == [
            "datetime", "ghi", "location", "latitude", "longitude"
        ]
        assert len(df) == 5  # the fixture is a 5-row slice
        assert (df["location"] == "somalia_location1").all()
        assert df["latitude"].between(3.0, 4.0).all()
        assert df["longitude"].between(43.0, 44.0).all()

    def test_helpful_error_on_missing_columns(self, tmp_path: Path) -> None:
        bad = tmp_path / "wrong_schema.csv"
        bad.write_text("date,solar,site\n2024-01-01,5000,kampala\n")
        adapter = StandardCsvAdapter(source_id="bad")
        with pytest.raises(ValueError, match="write a new GroundSourceAdapter"):
            adapter.parse(bad)
