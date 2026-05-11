"""Weights & Biases artifact I/O for training-dataset snapshots.

A snapshot directory (parquet + manifest) maps onto a W&B artifact of
type ``"training_dataset"``. Logging produces an artifact whose
contents are the snapshot's two files; downloading reverses it. The
manifest is mirrored into the artifact's ``metadata`` so the W&B UI
shows row count / version / date range without unpacking the parquet.

Two entry points:

* :func:`log_dataset_artifact` — create a fresh ``"dataset_build"`` run
  and upload the snapshot. Use after :func:`build_and_write_snapshot`.
* :func:`use_dataset_artifact` — download an artifact. Pass an active
  W&B run to record the consumption in the run's lineage; omit it for
  one-off downloads (a temporary ``"dataset_consume"`` run is created
  and finished).

W&B is imported lazily so the rest of the ``data`` package works even
when ``wandb`` is not installed (e.g. in unit tests).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .manifest import DatasetManifest
from .snapshot_io import load_snapshot

if TYPE_CHECKING:  # pragma: no cover - imported only for type hints
    import wandb

_logger = logging.getLogger(__name__)

_ARTIFACT_TYPE = "training_dataset"


def log_dataset_artifact(
    snapshot_dir: Path,
    *,
    artifact_name: str,
    project: str,
    entity: Optional[str] = None,
    aliases: tuple[str, ...] = (),
    notes: Optional[str] = None,
) -> str:
    """Upload a local snapshot dir as a W&B artifact.

    Args:
        snapshot_dir: Directory holding ``dataset.parquet`` +
            ``manifest.json`` (typically the output of
            :func:`build_and_write_snapshot`).
        artifact_name: Logical artifact name (e.g. ``"susse_training"``).
            W&B versions it as ``v0``, ``v1``, ... on each new upload.
        project: W&B project. The dataset_build run is logged here too.
        entity: W&B team / user. ``None`` uses the default from
            ``wandb login``.
        aliases: Extra aliases to apply (e.g. ``("v1", "candidate")``).
            ``"latest"`` is added automatically by W&B.
        notes: Optional human-readable description (free text).

    Returns:
        Qualified artifact reference, e.g.
        ``"<entity>/<project>/<artifact_name>:v0"``.
    """
    import wandb

    snapshot_dir = Path(snapshot_dir)
    manifest = _load_manifest_for_metadata(snapshot_dir)

    description = notes or (
        f"{manifest.name} {manifest.version} — "
        f"{manifest.n_rows} rows × {manifest.n_cols} cols, "
        f"date range {manifest.date_start}..{manifest.date_end}."
    )

    with wandb.init(
        project=project,
        entity=entity,
        job_type="dataset_build",
        config={
            "feature_selection": manifest.feature_selection.to_dict(),
            "date_start": manifest.date_start.isoformat(),
            "date_end": manifest.date_end.isoformat(),
            "location_filter": (
                list(manifest.location_filter)
                if manifest.location_filter is not None
                else None
            ),
            "warehouse_project": manifest.warehouse_project,
            "warehouse_dataset": manifest.warehouse_dataset,
            "git_sha": manifest.git_sha,
            "susse_version": manifest.susse_version,
        },
    ) as run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type=_ARTIFACT_TYPE,
            description=description,
            metadata=_artifact_metadata(manifest),
        )
        artifact.add_dir(str(snapshot_dir))
        for alias in aliases:
            artifact.aliases.append(alias)
        run.log_artifact(artifact)
        artifact.wait()  # block until the upload + version assignment lands
        ref = f"{run.entity}/{project}/{artifact_name}:v{artifact.version}"
        _logger.info("Logged dataset artifact %s.", ref)
    # Return outside the `with` block — mypy can't trace context-manager
    # suppression of exceptions otherwise, and the return-inside-with
    # pattern triggers a [return] error.
    return ref


def use_dataset_artifact(
    artifact_ref: str,
    dest: Path,
    *,
    run: Optional["wandb.Run"] = None,
):
    """Download an artifact reference into ``dest`` and return the loaded dataset.

    Args:
        artifact_ref: Qualified W&B reference, e.g.
            ``"<entity>/<project>/<name>:v0"`` or ``"...:latest"``.
        dest: Local directory to download the artifact contents into.
        run: An active W&B run. When provided, the consumption is
            recorded in the run's lineage (preferred during training).
            If ``None``, a temporary ``"dataset_consume"`` run is created
            and finished after download.

    Returns:
        :class:`TrainingDataset` from :func:`load_snapshot` — the parquet
        is hash-validated against the manifest on load.
    """
    import wandb

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    own_run = run is None
    active_run = run or wandb.init(
        project=_project_from_ref(artifact_ref),
        entity=_entity_from_ref(artifact_ref),
        job_type="dataset_consume",
    )
    try:
        artifact = active_run.use_artifact(artifact_ref, type=_ARTIFACT_TYPE)
        artifact_dir = Path(artifact.download(root=str(dest)))
        return load_snapshot(artifact_dir)
    finally:
        if own_run:
            active_run.finish()


def _load_manifest_for_metadata(snapshot_dir: Path) -> DatasetManifest:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cannot log {snapshot_dir} as a W&B artifact: "
            f"manifest.json is missing. Build the snapshot via "
            f"susse.datasets.build_and_write_snapshot first."
        )
    return DatasetManifest.from_json(manifest_path.read_text())


def _artifact_metadata(manifest: DatasetManifest) -> dict[str, Any]:
    """Subset of the manifest mirrored into the artifact metadata.

    The full manifest still travels inside the artifact as
    ``manifest.json``; this is just the bits W&B's UI surfaces.
    """
    return {
        "version": manifest.version,
        "n_rows": manifest.n_rows,
        "n_cols": manifest.n_cols,
        "date_start": manifest.date_start.isoformat(),
        "date_end": manifest.date_end.isoformat(),
        "git_sha": manifest.git_sha,
        "susse_version": manifest.susse_version,
        "content_hash": manifest.content_hash,
        "feature_selection": manifest.feature_selection.to_dict(),
        "warehouse_project": manifest.warehouse_project,
        "warehouse_dataset": manifest.warehouse_dataset,
        "warehouse_table_mods": manifest.warehouse_table_mods,
    }


def _entity_from_ref(artifact_ref: str) -> Optional[str]:
    parts = artifact_ref.split("/")
    return parts[0] if len(parts) >= 3 else None


def _project_from_ref(artifact_ref: str) -> str:
    parts = artifact_ref.split("/")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    raise ValueError(
        f"Cannot parse project from artifact_ref={artifact_ref!r}. "
        f"Expected '<entity>/<project>/<name>:<alias>' or "
        f"'<project>/<name>:<alias>'."
    )
