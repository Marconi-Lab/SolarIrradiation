"""Training orchestration — Trainer, TrainedBundle, splitter helpers.

Public entry points:

* :class:`Trainer`, :class:`TrainerConfig` — the orchestrator that
  takes a :class:`PreprocessedDataset` + :class:`BaseModelParams` +
  :data:`Splitter` and returns a :class:`TrainedBundle`.
* :class:`TrainedBundle`, :func:`load_bundle` — fully self-describing
  trained-model artifacts (model + feature spec + metadata + source
  manifest).
* :class:`TrainingMetadata`, :class:`ScoreSet` — typed records embedded
  in every bundle's ``metadata.json``.
* :data:`Splitter` (type alias), :func:`make_station_loso_splitter`,
  :func:`auto_pick_largest_station` — minimum splitter surface for
  NB 05; NB 06 will define the full splitter abstraction.
* :func:`score_predictions` — MAE / RMSE / R² helper used by
  :class:`Trainer` and re-exported for ad-hoc scoring.

W&B is opt-in: ``Trainer()`` constructed without a config performs
no W&B calls. See :class:`TrainerConfig` for the toggle.
"""

from .bundle import TrainedBundle, load_bundle
from .metadata import ScoreSet, TrainingMetadata
from .splitter import (
    Splitter,
    auto_pick_largest_station,
    make_station_loso_splitter,
)
from .trainer import Trainer, TrainerConfig, score_predictions

__all__ = [
    "ScoreSet",
    "Splitter",
    "TrainedBundle",
    "Trainer",
    "TrainerConfig",
    "TrainingMetadata",
    "auto_pick_largest_station",
    "load_bundle",
    "make_station_loso_splitter",
    "score_predictions",
]
