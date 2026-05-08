from __future__ import annotations
import pandas as pd
REQUIRED_COLS_LONG = ["date","geohash5","variable_id","value","source"]
def validate_long_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS_LONG if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.isna().all(axis=None):
        raise ValueError("All values are NA.")
