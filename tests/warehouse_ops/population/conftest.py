"""Shared fixtures for warehouse_ops.population tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture(scope="session")
def somalia_csv_path() -> Path:
    """Tiny pinned slice of real CBE somalia ground measurements."""
    path = _FIXTURE_DIR / "ground_measurements" / "somalia_slice.csv"
    if not path.exists():
        pytest.fail(
            f"Fixture missing: {path}. Was the fixture file deleted? It must be "
            f"checked in alongside these tests."
        )
    return path


@pytest.fixture()
def somalia_raw_df(somalia_csv_path: Path) -> pd.DataFrame:
    """The somalia fixture parsed as a raw DataFrame (without going through
    the adapter).

    Used by tests that exercise downstream functions independently of the
    adapter parsing logic.
    """
    df = pd.read_csv(somalia_csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
    df["ghi"] = df["ghi"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    return df
