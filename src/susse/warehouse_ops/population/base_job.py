from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from ..io.bq import BigQueryClient


@dataclass
class BaseJob:
    bq: BigQueryClient

    def run(self) -> None:
        raise NotImplementedError

    def _validate_nonempty(self, df: pd.DataFrame, context: str) -> None:
        if df.empty:
            raise ValueError(f"No data returned for {context}.")
