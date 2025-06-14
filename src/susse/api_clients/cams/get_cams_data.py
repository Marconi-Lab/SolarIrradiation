import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pvlib
from dotenv import load_dotenv, set_key

# Constants for configuration and magic strings
ENV_PATH = Path(".") / ".env"
EMAIL_ENV_KEY = "CAMS_EMAIL"
DEFAULT_IDENTIFIER = "cams_radiation"
TIMEOUT_SECONDS = 180


class CAMSClient:
    """
    Client for fetching CAMS radiation data via pvlib.

    Reads and stores user email in environment variables.
    """

    def __init__(
        self,
        env_path: Path = ENV_PATH,
        email_env_key: str = EMAIL_ENV_KEY,
        identifier: str = DEFAULT_IDENTIFIER,
        timeout: int = TIMEOUT_SECONDS,
    ) -> None:
        self._env_path = env_path
        self._email_env_key = email_env_key
        self._identifier = identifier
        self._timeout = timeout

        # Load environment variables from .env file
        load_dotenv(dotenv_path=self._env_path)

    def _get_email(self) -> str:
        """
        Retrieve the CAMS email from environment or prompt the user.

        Returns:
            str: Registered CAMS email.
        """
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
    ) -> Dict[str, Any]:
        """
        Fetch and process CAMS radiation data.

        Args:
            latitude (float): Latitude of location.
            longitude (float): Longitude of location.
            start (datetime): Start datetime (UTC).
            end (datetime): End datetime (UTC).
            time_step (str): ISO 8601 duration string (e.g., 'PT1H').

        Returns:
            Dict[str, Any]: Dictionary containing records, column names,
            metadata, and any error message.
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

            processed_df = self._process_dataframe(raw_df)

            return {
                "data": processed_df.to_dict(orient="records"),
                "columns": list(processed_df.columns),
                "metadata": metadata,
                "error": None,
            }

        except Exception as exc:
            return {
                "data": None,
                "columns": None,
                "metadata": None,
                "error": str(exc),
            }

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reset index and format timestamp column.

        Args:
            df (pd.DataFrame): Raw DataFrame with DateTimeIndex.

        Returns:
            pd.DataFrame: Processed DataFrame with ISO-formatted timestamps.
        """
        processed = df.reset_index().rename(columns={"index": "timestamp"})
        processed["timestamp"] = processed["timestamp"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return processed
