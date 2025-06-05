import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pvlib
import pytest
from dotenv import set_key as dotenv_set_key

from susse.api_clients import EMAIL_ENV_KEY, ENV_PATH, CAMSClient


def test_get_email_from_env(monkeypatch):
    client = CAMSClient()
    test_email = "user@example.com"
    monkeypatch.setenv(EMAIL_ENV_KEY, test_email)
    monkeypatch.setattr(
        "builtins.input", lambda prompt: pytest.skip("Should not prompt")
    )
    monkeypatch.setattr(
        dotenv_set_key,
        "__call__",
        lambda *args, **kwargs: pytest.skip("Should not set key"),
    )

    email = client._get_email()
    assert email == test_email


def test_process_dataframe():
    # Create sample DataFrame
    idx = pd.date_range("2025-01-01", periods=2, freq="H")
    df = pd.DataFrame({"value": [1, 2]}, index=idx)

    processed = CAMSClient._process_dataframe(df)
    assert "timestamp" in processed.columns
    assert processed.loc[0, "timestamp"].endswith("Z")
    assert processed.loc[1, "value"] == 2


def test_fetch_data_success(monkeypatch):
    client = CAMSClient()
    # Prepare dummy raw_df and metadata
    idx = pd.date_range("2025-01-01", periods=2, freq="H")
    raw_df = pd.DataFrame({"a": [10, 20]}, index=idx)
    metadata = {"meta": "data"}

    # Patch _get_email and pvlib.iotools.get_cams
    monkeypatch.setattr(client, "_get_email", lambda: "u@e.com")
    monkeypatch.setattr(pvlib.iotools, "get_cams", lambda **kwargs: (raw_df, metadata))

    result = client.fetch_data(
        0, 0, datetime(2025, 1, 1), datetime(2025, 1, 1, 1), "PT1H"
    )
    assert result["error"] is None
    assert result["metadata"] == metadata
    assert isinstance(result["data"], list)
    assert result["columns"] == ["timestamp", "a"]
    assert "timestamp" in result["data"][0]
    assert result["data"][1]["a"] == 20


def test_fetch_data_exception(monkeypatch):
    client = CAMSClient()
    def fail(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(client, "_get_email", lambda: "u@e.com")
    monkeypatch.setattr(pvlib.iotools, "get_cams", fail)

    result = client.fetch_data(
        0, 0, datetime(2025, 1, 1), datetime(2025, 1, 1, 1), "PT1H"
    )
    assert result["data"] is None
    assert "fail" in result["error"]
    assert result["columns"] is None
    assert result["metadata"] is None
