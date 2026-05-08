"""Dataset manifest — the regeneration recipe + provenance for a snapshot.

A :class:`DatasetManifest` is the metadata record that travels alongside
every materialised parquet snapshot. It captures everything you'd need
to either (a) prove what data went into a model run or (b) re-derive the
exact same snapshot from scratch:

* The :class:`FeatureSelection` recipe (what columns).
* Date range and location filter (what rows).
* Warehouse identity and per-source-table modification timestamps at
  query time (what state of the warehouse was sampled).
* The git SHA of the code that built it.
* The SHA-256 of the parquet payload (so corruption / tampering is
  detectable on load).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

import pandas as pd

from .feature_selection import FeatureSelection


@dataclass(frozen=True)
class DatasetManifest:
    """Provenance + regeneration recipe for one materialised dataset.

    Attributes:
        name: Logical dataset name (e.g. ``"susse_training"``). Stable
            across versions; the version field distinguishes them.
        version: Version label (e.g. ``"v1"``, ``"v2-pre-merra"``).
            User-controlled; W&B assigns its own ``v0``/``v1``/... when
            the dataset is logged as an artifact, but the manifest's
            version is the human-meaningful one.
        created_at_utc: When the snapshot was built (UTC, ISO 8601).
        susse_version: ``susse`` package version at build time.
        git_sha: ``git rev-parse HEAD`` at build time, or ``None`` if
            the build host had no git context.
        feature_selection: The recipe used to build the snapshot.
        date_start, date_end: Inclusive date range used in the query.
        location_filter: Tuple of location names if the query was scoped
            to a specific list, or ``None`` for "all locations".
        warehouse_project: GCP project hosting the warehouse.
        warehouse_dataset: BigQuery dataset name.
        warehouse_table_mods: Per-source ``last_modified_time`` (ISO 8601
            UTC string) of every warehouse table that contributed rows
            to the snapshot. Re-running with a manifest whose tabular
            mods don't match the live warehouse signals upstream drift.
        n_rows, n_cols: Shape of the materialised dataframe.
        column_names: Ordered tuple of column names. Lets a manifest
            consumer validate the parquet schema without loading it.
        content_hash: Hex SHA-256 of the parquet file's bytes. Validated
            on load; mismatch raises.
    """

    name: str
    version: str
    created_at_utc: str
    susse_version: str
    git_sha: str | None
    feature_selection: FeatureSelection
    date_start: date
    date_end: date
    location_filter: tuple[str, ...] | None
    warehouse_project: str
    warehouse_dataset: str
    warehouse_table_mods: dict[str, str]
    n_rows: int
    n_cols: int
    column_names: tuple[str, ...]
    content_hash: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DatasetManifest.name must be non-empty.")
        if not self.version:
            raise ValueError("DatasetManifest.version must be non-empty.")
        if self.date_start > self.date_end:
            raise ValueError(
                f"date_start ({self.date_start}) is after date_end "
                f"({self.date_end}). Swap the arguments."
            )
        if self.n_rows < 0 or self.n_cols < 0:
            raise ValueError(
                f"n_rows={self.n_rows}, n_cols={self.n_cols} must be >= 0."
            )

    def with_content_hash(self, content_hash: str) -> "DatasetManifest":
        """Return a copy with ``content_hash`` set.

        Used by snapshot writers: the manifest is constructed before the
        parquet exists, then re-stamped with the real hash after the file
        is written.
        """
        return replace(self, content_hash=content_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "created_at_utc": self.created_at_utc,
            "susse_version": self.susse_version,
            "git_sha": self.git_sha,
            "feature_selection": self.feature_selection.to_dict(),
            "date_start": self.date_start.isoformat(),
            "date_end": self.date_end.isoformat(),
            "location_filter": (
                list(self.location_filter)
                if self.location_filter is not None
                else None
            ),
            "warehouse_project": self.warehouse_project,
            "warehouse_dataset": self.warehouse_dataset,
            "warehouse_table_mods": dict(self.warehouse_table_mods),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "column_names": list(self.column_names),
            "content_hash": self.content_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetManifest":
        return cls(
            name=d["name"],
            version=d["version"],
            created_at_utc=d["created_at_utc"],
            susse_version=d["susse_version"],
            git_sha=d.get("git_sha"),
            feature_selection=FeatureSelection.from_dict(d["feature_selection"]),
            date_start=date.fromisoformat(d["date_start"]),
            date_end=date.fromisoformat(d["date_end"]),
            location_filter=(
                tuple(d["location_filter"])
                if d.get("location_filter") is not None
                else None
            ),
            warehouse_project=d["warehouse_project"],
            warehouse_dataset=d["warehouse_dataset"],
            warehouse_table_mods=dict(d.get("warehouse_table_mods", {})),
            n_rows=int(d["n_rows"]),
            n_cols=int(d["n_cols"]),
            column_names=tuple(d["column_names"]),
            content_hash=d["content_hash"],
        )

    @classmethod
    def from_json(cls, s: str) -> "DatasetManifest":
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class TrainingDataset:
    """A loaded training-dataset snapshot.

    Bundles the materialised dataframe with its manifest. Construct via
    :func:`susse.datasets.snapshot_io.load_snapshot` (which validates the
    content hash) rather than instantiating directly.
    """

    df: pd.DataFrame
    manifest: DatasetManifest

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version
