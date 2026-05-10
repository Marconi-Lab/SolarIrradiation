"""Typed metadata + score sets recorded inside every trained-model bundle.

A :class:`TrainedBundle` (defined in :mod:`susse.training.bundle`)
embeds one :class:`TrainingMetadata` instance describing how the model
was trained: when, against what data, with which splitter, and how it
scored on both the train fold and the held-out val fold.

Everything here is JSON-serialisable; nothing carries fitted state.
The fitted model lives elsewhere in the bundle, dispatched via
:class:`susse.models.ModelKind`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreSet:
    """A single fold's regression scores.

    Attributes:
        n_rows: Number of rows scored. Excludes any (true, pred) pair
            where either side was NaN — see :func:`score_predictions`.
        mae: Mean absolute error.
        rmse: Root mean squared error.
        r2: Coefficient of determination. ``NaN`` when the fold has
            zero target variance (a degenerate case worth surfacing
            rather than silencing).
    """

    n_rows: int
    mae: float
    rmse: float
    r2: float

    def __post_init__(self) -> None:
        if self.n_rows < 0:
            raise ValueError(f"ScoreSet.n_rows={self.n_rows} must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreSet":
        return cls(
            n_rows=int(d["n_rows"]),
            mae=float(d["mae"]),
            rmse=float(d["rmse"]),
            r2=float(d["r2"]),
        )


@dataclass(frozen=True)
class TrainingMetadata:
    """Provenance + scoring record for one trained model.

    Attributes:
        created_at_utc: When the training run produced the bundle
            (UTC, ISO 8601).
        susse_version: ``susse`` package version at training time.
        git_sha: ``git rev-parse HEAD`` at training time, or ``None``
            if the host had no git context.
        holdout_label: Free-form description of the held-out fold
            (e.g. ``"station-LOSO:kenya_location13"``). Survives
            persistence so cross-bundle comparisons are honest about
            *which* fold each bundle was scored against.
        splitter_name: ``__name__`` of the splitter callable that
            produced the train / val indices. Lets the bundle remember
            *how* the holdout was constructed in addition to *what*
            it was.
        n_train_rows: Number of rows the model was fit on.
        n_val_rows: Number of rows in the held-out fold.
        train_metrics: Scores of the trained model on the train fold.
        val_metrics: Scores of the trained model on the val fold.
        baseline_metrics: Scores of any raw-satellite baselines on the
            *same* val fold, keyed by the satellite-prediction column
            (e.g. ``"sat_ghi_nasa_kwh_m2_day"``). Empty when the
            preprocessed dataset doesn't expose a satellite column.
    """

    created_at_utc: str
    susse_version: str
    git_sha: str | None
    holdout_label: str
    splitter_name: str
    n_train_rows: int
    n_val_rows: int
    train_metrics: ScoreSet
    val_metrics: ScoreSet
    baseline_metrics: dict[str, ScoreSet]

    def __post_init__(self) -> None:
        if self.n_train_rows < 0 or self.n_val_rows < 0:
            raise ValueError(
                f"n_train_rows={self.n_train_rows}, "
                f"n_val_rows={self.n_val_rows} must be >= 0."
            )
        if not self.holdout_label:
            raise ValueError("holdout_label must be non-empty.")
        if not self.splitter_name:
            raise ValueError("splitter_name must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at_utc": self.created_at_utc,
            "susse_version": self.susse_version,
            "git_sha": self.git_sha,
            "holdout_label": self.holdout_label,
            "splitter_name": self.splitter_name,
            "n_train_rows": self.n_train_rows,
            "n_val_rows": self.n_val_rows,
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
            "baseline_metrics": {
                k: v.to_dict() for k, v in self.baseline_metrics.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingMetadata":
        return cls(
            created_at_utc=d["created_at_utc"],
            susse_version=d["susse_version"],
            git_sha=d.get("git_sha"),
            holdout_label=d["holdout_label"],
            splitter_name=d["splitter_name"],
            n_train_rows=int(d["n_train_rows"]),
            n_val_rows=int(d["n_val_rows"]),
            train_metrics=ScoreSet.from_dict(d["train_metrics"]),
            val_metrics=ScoreSet.from_dict(d["val_metrics"]),
            baseline_metrics={
                k: ScoreSet.from_dict(v)
                for k, v in d.get("baseline_metrics", {}).items()
            },
        )

    @classmethod
    def from_json(cls, s: str) -> "TrainingMetadata":
        return cls.from_dict(json.loads(s))
