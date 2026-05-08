"""Training-dataset assembly + versioning.

Public entry points:

* :class:`FeatureSelection` — typed recipe of which variables to
  materialise.
* :class:`FeatureService` — assembles training pairs / inference frames
  from the warehouse using a :class:`FeatureSelection`.
* :class:`DatasetManifest`, :class:`TrainingDataset` — value objects
  that travel with every snapshot.
* :func:`build_and_write_snapshot`, :func:`write_snapshot`,
  :func:`load_snapshot` — local parquet + manifest IO.
* :func:`log_dataset_artifact`, :func:`use_dataset_artifact` — W&B
  artifact integration for cross-machine dataset versioning.

The module imports are lightweight; ``wandb`` is only loaded when
:mod:`wandb_io` is actually used.
"""

from .feature_selection import FeatureSelection
from .feature_service import FeatureService
from .manifest import DatasetManifest, TrainingDataset
from .snapshot_io import (
    build_and_write_snapshot,
    load_snapshot,
    write_snapshot,
)
from .wandb_io import log_dataset_artifact, use_dataset_artifact

__all__ = [
    "DatasetManifest",
    "FeatureSelection",
    "FeatureService",
    "TrainingDataset",
    "build_and_write_snapshot",
    "load_snapshot",
    "log_dataset_artifact",
    "use_dataset_artifact",
    "write_snapshot",
]
