from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
from datetime import date

import pandas as pd
from google.cloud import bigquery

from .config import PROJECT_ID, DATASET

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BigQuery client wrapper
# ---------------------------------------------------------------------------

@dataclass
class BigQueryOptions:
    project_id: str
    location: str | None = None

class BigQueryClient:
    def __init__(self, options: Optional[BigQueryOptions] = None):
        options = options or BigQueryOptions(PROJECT_ID)
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery not available or not configured.")
        self._client = bigquery.Client(project=options.project_id, location=options.location)

    def query(self, sql: str, *, job_config: Optional[bigquery.QueryJobConfig] = None) -> pd.DataFrame:
        job = self._client.query(sql, job_config=job_config)
        return job.result().to_dataframe()

    def load_dataframe(self, df: pd.DataFrame, table: str, write_disposition: str = "WRITE_APPEND",
                       schema: Optional[Sequence['bigquery.SchemaField']] = None, autodetect: bool = True) -> None:
        job_config = bigquery.LoadJobConfig(write_disposition=write_disposition, schema=schema, autodetect=autodetect)
        self._client.load_table_from_dataframe(df, table, job_config=job_config).result()

    def execute_ddl(self, ddl: str) -> None:
        self._client.query(ddl).result()
