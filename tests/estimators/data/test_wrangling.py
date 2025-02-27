from pathlib import Path

import pandas as pd
import pytest

from susse.estimators.data.wrangling import (
    wrangle_CAMSRAD_data,
    wrangle_NREL_Data,
    wrangle_Solcast_data,
)

# Define base data path
DATA_PATH = Path("jupyter_notebooks/Data/irradiation_estimates/CBE_Data/Egypt/raw")


def test_wrangle_solcast_data():
    solcast_file = (
        DATA_PATH / "Solcast_Location1_29.887561_32.460447_fixed_23_180_PT60M.csv"
    )
    df = wrangle_Solcast_data(solcast_file)

    # Test structure and columns
    expected_columns = {
        "Temperature (C)",
        "Albedo",
        "Clearsky DHI (W/m2)",
        "Clearsky DNI (W/m2)",
        "Clearsky GHI (W/m2)",
        "Clearsky GTI (W/m2)",
        "DHI (W/m2)",
        "DNI (W/m2)",
        "GHI (W/m2)",
        "GTI (W/m2)",
        "Date",
        "Period",
    }

    assert all(col in df.columns for col in expected_columns)
    assert isinstance(df["Date"].iloc[0], pd.Timestamp)
    assert df.shape[1] == 12  # Verify we have exactly 12 columns
    assert not df.empty  # Verify the dataframe is not empty


def test_wrangle_camsrad_data():
    camsrad_file = DATA_PATH / "CAMSRAD_location1.csv"
    df = wrangle_CAMSRAD_data(camsrad_file)

    # Test structure and columns
    expected_columns = {
        "TAO (Wh/m2/day)",
        "GHI (Wh/m2/day)",
        "DHI (Wh/m2/day)",
        "BHI (Wh/m2/day)",
        "DNI (Wh/m2/day)",
        "Clear sky GHI (Wh/m2/day)",
        "Clear sky DHI (Wh/m2/day)",
        "Clear sky BHI (Wh/m2/day)",
        "Clear sky DNI (Wh/m2/day)",
        "Time",
    }

    assert all(col in df.columns for col in expected_columns)
    assert isinstance(df["Time"].iloc[0], pd.Timestamp)
    assert "# Observation period" not in df.columns
    assert "End Day" not in df.columns
    assert not df.empty  # Verify the dataframe is not empty


def test_data_files_exist():
    """Test to ensure required data files are present"""
    solcast_file = (
        DATA_PATH / "Solcast_Location1_29.887561_32.460447_fixed_23_180_PT60M.csv"
    )
    camsrad_file = DATA_PATH / "CAMSRAD_location1.csv"

    assert solcast_file.exists(), f"Solcast data file not found at {solcast_file}"
    assert camsrad_file.exists(), f"CAMSRAD data file not found at {camsrad_file}"
