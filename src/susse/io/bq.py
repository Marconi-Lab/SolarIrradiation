from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
from datetime import date

import pandas as pd
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# BigQuery client wrapper
# ---------------------------------------------------------------------------


class BQ:
    """Lightweight BigQuery client wrapper with DataFrame helpers."""

    def __init__(self, project: Optional[str] = None) -> None:
        self.client = bigquery.Client(project=project)

    def df(self, sql: str, job_config: Optional[bigquery.QueryJobConfig] = None) -> pd.DataFrame:
        job = self.client.query(sql, job_config=job_config)
        return job.result().to_dataframe()
