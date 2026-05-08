"""Pluggable parsers for heterogeneous ground-measurement files.

Each new ground-truth source (a publication's CSV, a ministry export, a
station logger dump) gets its own :class:`GroundSourceAdapter` subclass.
The adapter knows the source's column conventions and produces a
DataFrame in the standard raw schema; everything downstream (validation,
curation, MERGE-load) sees the same shape regardless of origin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd

from .validators import validate_ground_raw


class GroundSourceAdapter(ABC):
    """Parses a ground-measurement file into the standard raw schema.

    The standard raw schema (matching ``ground_measurements_raw``):

    * ``datetime`` — pandas-parseable date / timestamp
    * ``ghi`` — daily total/mean GHI in Wh/m²/day
    * ``location`` — site name (stable string)
    * ``latitude`` — degrees, [-90, 90]
    * ``longitude`` — degrees, [-180, 180]

    Implementations should call :func:`validate_ground_raw` on their output
    so contract violations surface immediately.
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Stable identifier for this adapter (logged with ingest runs)."""

    @abstractmethod
    def parse(self, file_path: Path) -> pd.DataFrame:
        """Parse ``file_path`` and return a DataFrame in the standard schema."""


class StandardCsvAdapter(GroundSourceAdapter):
    """Adapter for sources already shaped in the standard CSV schema.

    The CrossBoundary, Makerere Physics, and Uganda Ministry of Energy
    sources all emit CSVs with columns ``datetime, ghi, location, latitude,
    longitude`` — they share this adapter rather than each having its own.

    Args:
        source_id: human-readable label for the source (e.g. ``"CBE"``,
            ``"MAK_physics_dept"``). Embedded in :attr:`adapter_id` for log
            traceability and helpful when one warehouse run touches several
            files of the same shape.
    """

    EXPECTED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "datetime",
        "ghi",
        "location",
        "latitude",
        "longitude",
    )

    def __init__(self, source_id: str) -> None:
        if not source_id:
            raise ValueError("StandardCsvAdapter.source_id must be non-empty.")
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def adapter_id(self) -> str:
        return f"standard-csv:{self._source_id}"

    def parse(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        missing = [c for c in self.EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{file_path.name}: missing required columns {missing}. "
                f"StandardCsvAdapter expects {list(self.EXPECTED_COLUMNS)}. "
                f"If this source has different columns, write a new "
                f"GroundSourceAdapter subclass rather than reshaping the file."
            )
        out = df[list(self.EXPECTED_COLUMNS)].copy()
        out["datetime"] = pd.to_datetime(out["datetime"]).dt.date
        out["ghi"] = out["ghi"].astype(float)
        out["location"] = out["location"].astype(str)
        out["latitude"] = out["latitude"].astype(float)
        out["longitude"] = out["longitude"].astype(float)
        validate_ground_raw(out, context=f"StandardCsvAdapter[{file_path.name}]")
        return out
