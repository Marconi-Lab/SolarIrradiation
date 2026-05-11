"""Tests for the CAMS Radiation Service client wrapper.

End-to-end fetches require a registered SoDa email and live network;
these tests cover only the parts that don't.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pvlib
import pytest

from susse.api_clients.cams import CAMSClient
from susse.api_clients.cams.cams_client import CamsApiError


def test_get_email_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CAMSClient.DEFAULT_EMAIL_ENV_KEY, "user@example.com")
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: pytest.fail("input() should not be called when env var is set"),
    )
    assert CAMSClient()._get_email() == "user@example.com"


def test_process_dataframe_iso_formats_timestamp_column() -> None:
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    raw = pd.DataFrame({"value": [1.0, 2.0]}, index=idx)
    processed = CAMSClient._process_dataframe(raw)
    assert "timestamp" in processed.columns
    assert processed.loc[0, "timestamp"].endswith("Z")
    assert processed.loc[1, "value"] == 2.0


def test_fetch_data_returns_processed_df_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CAMSClient.fetch_data returns a (processed_df, metadata) tuple, not
    # a dict. Earlier tests asserted the dict shape and broke after the
    # refactor — this is the up-to-date contract.
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    raw_df = pd.DataFrame({"ghi": [10.0, 20.0]}, index=idx)
    metadata = {"latitude": 0.0, "longitude": 0.0}

    monkeypatch.setattr(CAMSClient, "_get_email", lambda self: "u@e.com")
    monkeypatch.setattr(pvlib.iotools, "get_cams", lambda **kw: (raw_df, metadata))

    df, meta = CAMSClient().fetch_data(
        latitude=0.0,
        longitude=0.0,
        start=datetime(2025, 1, 1),
        end=datetime(2025, 1, 1, 1),
        time_step="1h",
    )
    assert "timestamp" in df.columns
    assert list(df["ghi"]) == [10.0, 20.0]
    assert meta == metadata


def test_fetch_data_raises_cams_api_error_on_pvlib_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Earlier behaviour was to return a {"error": ...} dict; the CAMS
    # refactor removed the silent-failure path so callers can't propagate
    # ``None`` DataFrames downstream.
    def fail(**kwargs):
        raise RuntimeError("network fell over")

    monkeypatch.setattr(CAMSClient, "_get_email", lambda self: "u@e.com")
    monkeypatch.setattr(pvlib.iotools, "get_cams", fail)

    with pytest.raises(CamsApiError, match="network fell over"):
        CAMSClient().fetch_data(
            latitude=0.0,
            longitude=0.0,
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 1, 1),
            time_step="1h",
        )
