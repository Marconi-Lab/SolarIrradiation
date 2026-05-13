"""Trainer — orchestrate fit / predict / score / bundle for one model.

Single-responsibility: take a preprocessed dataset, a typed params
object, and a splitter; return a :class:`TrainedBundle` with full
provenance. Optional W&B integration is **strict opt-in**: a default
``Trainer()`` performs no W&B calls. Pass a non-``None``
:attr:`TrainerConfig.wandb_project` to enable run + artifact logging.

What the trainer does *not* do (deliberate exclusions):

* No hyperparameter search — out of scope for the v1 pipeline.
* No richer metrics than the :class:`ScoreSet` MAE/RMSE/R² triplet —
  paper-style metrics (IOA, MBE, normalised RMSE/MAE) live in
  :mod:`susse.evaluation.metrics` and are applied after training via
  :class:`susse.evaluation.Evaluator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ..models import BaseModelParams, ModelFactory
from ..preprocessing import PreprocessedDataset
from ..provenance import git_sha, susse_version
from ..warehouse_ops.population.types import (
    IrradianceBand,
    Source,
    satellite_irradiance_column,
)
from .bundle import TrainedBundle
from .metadata import ScoreSet, TrainingMetadata
from .splitter import Splitter

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

_DEFAULT_BASELINE_COLUMNS: tuple[str, ...] = (
    satellite_irradiance_column(Source.NASA_POWER, IrradianceBand.GHI),
    satellite_irradiance_column(Source.CAMS, IrradianceBand.GHI),
)


@dataclass(frozen=True)
class TrainerConfig:
    """Trainer configuration — W&B opt-in plus baseline-scoring choices.

    Defaults are **fully offline**: ``wandb_project=None`` means no
    run is created and no artifact is logged. Setting it to a string
    activates the full ``wandb.init → log_metrics → log_artifact``
    flow in :meth:`Trainer.train`.

    Attributes:
        wandb_project: W&B project name. ``None`` (default) disables
            all W&B activity. Mirrors the explicit-opt-in convention
            from NB 02's ``LOG_TO_WANDB``.
        wandb_entity: W&B team / user. ``None`` uses your default
            from ``wandb login``.
        wandb_tags: Extra tags to apply to the W&B run.
        baseline_columns: Column names on ``processed.df`` whose
            values are scored against the val target as baselines.
            Missing columns are silently skipped (so a spec without
            both satellite columns still works).
    """

    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: tuple[str, ...] = ()
    baseline_columns: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_BASELINE_COLUMNS,
    )


class Trainer:
    """Orchestrator. Construct once per training-time configuration."""

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self._config = config or TrainerConfig()

    @property
    def config(self) -> TrainerConfig:
        return self._config

    def train(
        self,
        *,
        processed: PreprocessedDataset,
        params: BaseModelParams,
        splitter: Splitter,
        holdout_label: str,
        bundle_dest: Optional["Path"] = None,
        dataset_artifact_ref: Optional[str] = None,
    ) -> TrainedBundle:
        """Fit, score, bundle, and (optionally) log to W&B.

        Args:
            processed: The model-ready dataset. Carries its source
                :class:`DatasetManifest` for full lineage.
            params: Typed hyperparameters dispatching the concrete
                regressor via :class:`susse.models.ModelFactory`.
            splitter: :class:`Splitter` instance producing
                ``(train_idx, val_idx)``. The two index sets must be
                disjoint and both must be subsets of
                ``processed.df.index``.
            holdout_label: Free-form description of the val fold,
                recorded verbatim in :class:`TrainingMetadata`.
            bundle_dest: If given, the bundle is saved to this
                directory after training. If W&B is enabled and this
                is ``None``, a temporary directory is created so the
                W&B artifact has files to upload.
            dataset_artifact_ref: Qualified W&B reference of the
                source dataset artifact (e.g.
                ``"<entity>/<project>/<name>:v0"``). When provided
                alongside an active W&B run, recorded as a consumed
                artifact for lineage.

        Returns:
            A fully populated :class:`TrainedBundle`.

        Raises:
            ValueError: If the splitter returns overlapping indices,
                or if either fold ends up empty.
        """
        train_idx, val_idx = splitter(processed)
        self._validate_split(train_idx, val_idx, processed)

        X_full = processed.X()
        y_full = processed.y()
        X_train, y_train = X_full.loc[train_idx], y_full.loc[train_idx]
        X_val, y_val = X_full.loc[val_idx], y_full.loc[val_idx]

        regressor = ModelFactory.create(params).fit(X_train, y_train)

        train_metrics = score_predictions(y_train, regressor.predict(X_train))
        val_metrics = score_predictions(y_val, regressor.predict(X_val))
        baseline_metrics = self._score_baselines(
            processed=processed,
            val_idx=val_idx,
            y_val=y_val,
        )

        metadata = TrainingMetadata(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            susse_version=susse_version(),
            git_sha=git_sha(),
            holdout_label=holdout_label,
            splitter_name=splitter.name,
            n_train_rows=len(train_idx),
            n_val_rows=len(val_idx),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            baseline_metrics=baseline_metrics,
        )

        bundle = TrainedBundle(
            regressor=regressor,
            feature_spec=processed.feature_spec,
            metadata=metadata,
            source_manifest=processed.source_manifest,
        )

        if self._config.wandb_project is not None:
            bundle = self._log_to_wandb(
                bundle=bundle,
                params=params,
                bundle_dest=bundle_dest,
                dataset_artifact_ref=dataset_artifact_ref,
            )
        elif bundle_dest is not None:
            bundle.save(bundle_dest)

        return bundle

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _validate_split(
        train_idx: pd.Index,
        val_idx: pd.Index,
        processed: PreprocessedDataset,
    ) -> None:
        if len(train_idx) == 0:
            raise ValueError(
                "Splitter returned an empty train fold — nothing to fit. "
                "Check that the held-out group is not the entire dataset."
            )
        if len(val_idx) == 0:
            raise ValueError(
                "Splitter returned an empty val fold — nothing to score "
                "against. Check the splitter's match criterion."
            )
        overlap = train_idx.intersection(val_idx)
        if len(overlap) > 0:
            raise ValueError(
                f"Splitter returned overlapping train/val indices "
                f"({len(overlap)} rows in both). Trainer requires a "
                f"disjoint split to avoid leakage."
            )
        full_index = processed.df.index
        if not (train_idx.isin(full_index).all() and val_idx.isin(full_index).all()):
            raise ValueError(
                "Splitter returned indices not present in "
                "processed.df.index. The splitter must select rows "
                "from the supplied dataset only."
            )

    def _score_baselines(
        self,
        *,
        processed: PreprocessedDataset,
        val_idx: pd.Index,
        y_val: pd.Series,
    ) -> dict[str, ScoreSet]:
        out: dict[str, ScoreSet] = {}
        for col in self._config.baseline_columns:
            if col not in processed.df.columns:
                continue
            preds = processed.df.loc[val_idx, col]
            out[col] = score_predictions(y_val, preds)
        return out

    def _log_to_wandb(
        self,
        *,
        bundle: TrainedBundle,
        params: BaseModelParams,
        bundle_dest: Optional["Path"],
        dataset_artifact_ref: Optional[str],
    ) -> TrainedBundle:
        # Local import — only paid by users who opt into W&B.
        import tempfile
        from pathlib import Path

        import wandb

        cfg = self._config
        wandb_config = {
            "params": params.to_dict(),
            "splitter_name": bundle.metadata.splitter_name,
            "holdout_label": bundle.metadata.holdout_label,
            "source_dataset_name": bundle.source_manifest.name,
            "source_dataset_version": bundle.source_manifest.version,
            "source_content_hash": bundle.source_manifest.content_hash,
            "git_sha": bundle.metadata.git_sha,
            "susse_version": bundle.metadata.susse_version,
        }

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            job_type="training",
            tags=list(cfg.wandb_tags),
            config=wandb_config,
        )
        try:
            if dataset_artifact_ref is not None:
                run.use_artifact(dataset_artifact_ref)

            run.log(
                {
                    "train/n_rows": bundle.metadata.n_train_rows,
                    "val/n_rows": bundle.metadata.n_val_rows,
                    "train/mae": bundle.metadata.train_metrics.mae,
                    "train/rmse": bundle.metadata.train_metrics.rmse,
                    "train/r2": bundle.metadata.train_metrics.r2,
                    "val/mae": bundle.metadata.val_metrics.mae,
                    "val/rmse": bundle.metadata.val_metrics.rmse,
                    "val/r2": bundle.metadata.val_metrics.r2,
                    **{
                        f"baseline/{col}/mae": s.mae
                        for col, s in bundle.metadata.baseline_metrics.items()
                    },
                    **{
                        f"baseline/{col}/rmse": s.rmse
                        for col, s in bundle.metadata.baseline_metrics.items()
                    },
                    **{
                        f"baseline/{col}/r2": s.r2
                        for col, s in bundle.metadata.baseline_metrics.items()
                    },
                }
            )

            if bundle_dest is not None:
                bundle.save(bundle_dest)
                artifact_dir = Path(bundle_dest)
                artifact = wandb.Artifact(
                    name=f"{bundle.source_manifest.name}-" f"{params.kind.value}",
                    type="trained_model",
                    metadata={
                        "model_kind": params.kind.value,
                        "val_mae": bundle.metadata.val_metrics.mae,
                        "val_rmse": bundle.metadata.val_metrics.rmse,
                        "val_r2": bundle.metadata.val_metrics.r2,
                        "holdout_label": bundle.metadata.holdout_label,
                    },
                )
                artifact.add_dir(str(artifact_dir))
                run.log_artifact(artifact)
            else:
                # No persistent dest given — use a temp dir so the
                # artifact still has files to upload, but warn the
                # caller that the local copy is ephemeral.
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_dir = Path(tmp)
                    bundle.save(tmp_dir)
                    artifact = wandb.Artifact(
                        name=f"{bundle.source_manifest.name}-" f"{params.kind.value}",
                        type="trained_model",
                        metadata={
                            "model_kind": params.kind.value,
                            "val_mae": bundle.metadata.val_metrics.mae,
                            "val_rmse": bundle.metadata.val_metrics.rmse,
                            "val_r2": bundle.metadata.val_metrics.r2,
                            "holdout_label": bundle.metadata.holdout_label,
                        },
                    )
                    artifact.add_dir(str(tmp_dir))
                    run.log_artifact(artifact)
                    artifact.wait()
        finally:
            run.finish()

        return bundle


def score_predictions(y_true: pd.Series, y_pred: pd.Series) -> ScoreSet:
    """MAE / RMSE / R² over rows where neither side is NaN.

    Treating NaN-bearing rows as scoreable (with whatever pandas
    default) would silently inflate / deflate metrics; instead we
    drop them and record the surviving row count in
    :attr:`ScoreSet.n_rows` so the caller can spot data sparsity.

    Args:
        y_true: Target series.
        y_pred: Predicted series (or any column of predictions).
            Index alignment is required — use ``.loc[idx]`` upstream.

    Returns:
        A :class:`ScoreSet` over the (true, pred) rows that survived
        the NaN filter. ``r2`` is ``NaN`` when target variance is
        zero; ``n_rows`` is zero in that pathological case but still
        reported.
    """
    valid = y_true.notna() & y_pred.notna()
    n = int(valid.sum())
    if n == 0:
        return ScoreSet(n_rows=0, mae=float("nan"), rmse=float("nan"), r2=float("nan"))
    yt = y_true[valid].to_numpy(dtype=float)
    yp = y_pred[valid].to_numpy(dtype=float)
    err = yp - yt
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    ss_res = float((err**2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return ScoreSet(n_rows=n, mae=mae, rmse=rmse, r2=r2)
