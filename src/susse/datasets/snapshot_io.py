"""Local snapshot writer / reader.

Materialises a built training DataFrame + its manifest into a directory
holding ``dataset.parquet`` and ``manifest.json``. The reader validates
the parquet's SHA-256 against the manifest on load — corruption or
post-hoc tampering is detected before training consumes the data.

Storage layout::

    <snapshot_dir>/
        dataset.parquet
        manifest.json

Snapshot dirs are W&B-artifact-friendly: ``add_dir(snapshot_dir)`` on a
``wandb.Artifact`` uploads both files together; the artifact's
``.metadata`` mirrors a subset of the manifest.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..provenance import git_sha as _git_sha
from ..provenance import susse_version as _susse_version
from .feature_selection import FeatureSelection
from .feature_service import FeatureService
from .manifest import DatasetManifest, TrainingDataset

_logger = logging.getLogger(__name__)

_PARQUET_FILENAME = "dataset.parquet"
_MANIFEST_FILENAME = "manifest.json"


def build_and_write_snapshot(
    *,
    feature_service: FeatureService,
    selection: FeatureSelection,
    date_start: date,
    date_end: date,
    locations: Optional[Sequence[str]],
    name: str,
    version: str,
    dest: Path,
) -> TrainingDataset:
    """Build a training-pairs DataFrame and write it as a snapshot.

    One-shot helper: queries the warehouse, builds the manifest with full
    provenance (warehouse table mods, git SHA, susse version, content
    hash), writes both files into ``dest``, and returns the loaded
    :class:`TrainingDataset` so the caller can inspect it without a
    second read.

    Args:
        feature_service: The configured :class:`FeatureService` to query
            with.
        selection: Recipe — what variables to materialise.
        date_start, date_end: Inclusive date range.
        locations: Optional station-name filter; ``None`` = all.
        name: Logical dataset name (e.g. ``"susse_training"``).
        version: Human-meaningful version label (e.g. ``"v1"``).
        dest: Directory to write the snapshot into. Created if missing.

    Returns:
        A :class:`TrainingDataset` bundling the loaded df + manifest.
    """
    df = feature_service.build_training_pairs(
        selection=selection,
        date_start=date_start,
        date_end=date_end,
        locations=locations,
    )
    table_mods = feature_service.warehouse_table_mods(selection)
    config = feature_service.tables.config
    manifest = _build_manifest(
        df=df,
        name=name,
        version=version,
        selection=selection,
        date_start=date_start,
        date_end=date_end,
        locations=locations,
        warehouse_project=config.project_id,
        warehouse_dataset=config.dataset,
        warehouse_table_mods=table_mods,
    )
    return write_snapshot(df=df, manifest=manifest, dest=dest)


def write_snapshot(
    *,
    df: pd.DataFrame,
    manifest: DatasetManifest,
    dest: Path,
) -> TrainingDataset:
    """Write ``df`` + ``manifest`` into ``dest`` and return them as a dataset.

    The parquet file is hashed *after* it lands on disk; the manifest is
    re-stamped with the real hash before being written. This guarantees
    the manifest's ``content_hash`` matches the bytes that future readers
    will load.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    parquet_path = dest / _PARQUET_FILENAME
    manifest_path = dest / _MANIFEST_FILENAME

    df.to_parquet(parquet_path, index=False)
    content_hash = _sha256_file(parquet_path)
    manifest_final = manifest.with_content_hash(content_hash)
    manifest_path.write_text(manifest_final.to_json())

    _logger.info(
        "Wrote snapshot %s v%s to %s — %d rows × %d cols, hash=%s.",
        manifest_final.name,
        manifest_final.version,
        dest,
        manifest_final.n_rows,
        manifest_final.n_cols,
        content_hash[:12],
    )
    return TrainingDataset(df=df, manifest=manifest_final)


def load_snapshot(src: Path) -> TrainingDataset:
    """Load a snapshot dir, validating its content hash on the way in.

    Raises:
        FileNotFoundError: If the parquet or manifest is missing.
        ValueError: If the parquet's SHA-256 doesn't match the manifest.
    """
    src = Path(src)
    parquet_path = src / _PARQUET_FILENAME
    manifest_path = src / _MANIFEST_FILENAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Snapshot is missing {_PARQUET_FILENAME}: {parquet_path}. "
            f"Was the directory truncated?"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Snapshot is missing {_MANIFEST_FILENAME}: {manifest_path}. "
            f"A snapshot dir must contain both {_PARQUET_FILENAME} and "
            f"{_MANIFEST_FILENAME}."
        )

    manifest = DatasetManifest.from_json(manifest_path.read_text())
    actual_hash = _sha256_file(parquet_path)
    if actual_hash != manifest.content_hash:
        raise ValueError(
            f"Content-hash mismatch on {parquet_path}: manifest declares "
            f"{manifest.content_hash} but the file hashes to {actual_hash}. "
            f"The snapshot was modified or is corrupted; re-derive it from "
            f"the recipe in manifest.json (see feature_selection field)."
        )
    df = pd.read_parquet(parquet_path)
    return TrainingDataset(df=df, manifest=manifest)


def _build_manifest(
    *,
    df: pd.DataFrame,
    name: str,
    version: str,
    selection: FeatureSelection,
    date_start: date,
    date_end: date,
    locations: Optional[Sequence[str]],
    warehouse_project: str,
    warehouse_dataset: str,
    warehouse_table_mods: dict[str, str],
) -> DatasetManifest:
    return DatasetManifest(
        name=name,
        version=version,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        susse_version=_susse_version(),
        git_sha=_git_sha(),
        feature_selection=selection,
        date_start=date_start,
        date_end=date_end,
        location_filter=tuple(locations) if locations is not None else None,
        warehouse_project=warehouse_project,
        warehouse_dataset=warehouse_dataset,
        warehouse_table_mods=warehouse_table_mods,
        n_rows=len(df),
        n_cols=len(df.columns),
        column_names=tuple(df.columns),
        content_hash="",  # placeholder, stamped by write_snapshot
    )


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes (handles arbitrarily large parquets)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
