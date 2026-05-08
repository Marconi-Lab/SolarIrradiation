"""Thin client around pvlib's ``get_cams`` for CAMS radiation data."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Tuple

import pandas as pd
import pvlib
from dotenv import load_dotenv, set_key


class CamsApiError(RuntimeError):
    """Raised when a CAMS request fails or returns no usable data.

    Wraps the underlying pvlib / network exception so callers can catch
    a single error type at the boundary instead of unrelated ones.
    """


class CAMSClient:
    """Client for fetching CAMS radiation data via pvlib.

    Reads the registered CAMS email from a ``.env`` file (or prompts and
    persists it). Failures from the underlying pvlib call surface as
    :class:`CamsApiError` so callers don't silently propagate ``None``
    DataFrames downstream.
    """

    DEFAULT_ENV_PATH: ClassVar[Path] = Path(".") / ".env"
    DEFAULT_EMAIL_ENV_KEY: ClassVar[str] = "CAMS_EMAIL"
    DEFAULT_IDENTIFIER: ClassVar[str] = "cams_radiation"
    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 180

    def __init__(
        self,
        env_path: Path | None = None,
        email_env_key: str | None = None,
        identifier: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._env_path = env_path or self.DEFAULT_ENV_PATH
        self._email_env_key = email_env_key or self.DEFAULT_EMAIL_ENV_KEY
        self._identifier = identifier or self.DEFAULT_IDENTIFIER
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT_SECONDS
        load_dotenv(dotenv_path=self._env_path)

    @property
    def identifier(self) -> str:
        return self._identifier

    def _get_email(self) -> str:
        """Retrieve the CAMS email from environment or prompt the user."""
        email = os.getenv(self._email_env_key)
        if not email:
            email = input("Enter your registered CAMS email: ").strip()
            set_key(str(self._env_path), self._email_env_key, email)
        return email

    def fetch_data(
        self,
        latitude: float,
        longitude: float,
        start: datetime,
        end: datetime,
        time_step: str,
    ) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch CAMS radiation data for one point.

        Args:
            latitude: degrees, [-90, 90].
            longitude: degrees, [-180, 180].
            start: UTC datetime.
            end: UTC datetime.
            time_step: pvlib time-step string. One of ``'1min'``, ``'15min'``,
                ``'1h'``, ``'1d'``, ``'1M'``.

        Returns:
            ``(processed_df, metadata)`` — DataFrame with a ``timestamp``
            column (ISO string) followed by per-variable columns, plus the
            metadata dict pvlib returns.

        Raises:
            CamsApiError: pvlib raised any exception during the request.
        """
        email = self._get_email()
        try:
            raw_df, metadata = pvlib.iotools.get_cams(
                latitude=latitude,
                longitude=longitude,
                start=start,
                end=end,
                email=email,
                time_step=time_step,
                timeout=self._timeout,
                identifier=self._identifier,
            )
        except Exception as exc:
            raise CamsApiError(
                f"CAMS fetch failed for ({latitude}, {longitude}) "
                f"between {start.date()} and {end.date()}: {exc}"
            ) from exc

        return self._process_dataframe(raw_df), metadata

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Reset the datetime index and ISO-format the timestamp column."""
        processed = df.reset_index().rename(columns={"index": "timestamp"})
        processed["timestamp"] = processed["timestamp"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return processed


# Legacy module-level constants exported for backwards compatibility with
# code that imported them directly. New code should reference the ClassVar
# attributes on :class:`CAMSClient`.
ENV_PATH = CAMSClient.DEFAULT_ENV_PATH
EMAIL_ENV_KEY = CAMSClient.DEFAULT_EMAIL_ENV_KEY
DEFAULT_IDENTIFIER = CAMSClient.DEFAULT_IDENTIFIER
TIMEOUT_SECONDS = CAMSClient.DEFAULT_TIMEOUT_SECONDS
