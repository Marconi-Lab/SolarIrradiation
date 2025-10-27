from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json

from ..models import BaseRegressor, RFRegressor
from ..preprocessing import Preprocessor, FeatureSpec
from ..configs import FeatureConfig, ModelConfig, TrainingConfig
from ..tracking import MLflowLogger

# ---------------------------------------------------------------------------
# Trained bundle (artifacts + configs)
# ---------------------------------------------------------------------------


@dataclass
class TrainedBundle:
    model: BaseRegressor
    preprocessor: Preprocessor
    feature_spec: FeatureSpec
    feature_cfg: FeatureConfig
    model_cfg: ModelConfig
    training_cfg: TrainingConfig
    versions: Dict[str, Any]

    def save(self, directory: Path, mlflow_logger: Optional[MLflowLogger] = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # Save model + preprocessor + configs
        model_dir = directory / "model"
        prep_dir = directory / "preprocessor"
        model_dir.mkdir(exist_ok=True)
        prep_dir.mkdir(exist_ok=True)

        self.model.save(model_dir)
        self.preprocessor.save(prep_dir)
        (directory / "feature_spec.json").write_text(self.feature_spec.to_json(), encoding="utf-8")
        (directory / "feature_cfg.json").write_text(json.dumps(asdict(self.feature_cfg), indent=2), encoding="utf-8")
        (directory / "model_cfg.json").write_text(json.dumps(asdict(self.model_cfg), indent=2), encoding="utf-8")
        (directory / "training_cfg.json").write_text(json.dumps(asdict(self.training_cfg), indent=2), encoding="utf-8")
        (directory / "versions.json").write_text(json.dumps(self.versions, indent=2), encoding="utf-8")

        # MLflow artifacts
        if mlflow_logger is not None and mlflow_logger.enabled:
            for p in [model_dir, prep_dir, directory / "feature_spec.json", directory / "feature_cfg.json", directory / "model_cfg.json", directory / "training_cfg.json", directory / "versions.json"]:
                mlflow_logger.log_artifact(p if isinstance(p, Path) else Path(p))

    @staticmethod
    def load(directory: Path) -> TrainedBundle:
        # Read configs/specs
        feature_spec = FeatureSpec.from_json((directory / "feature_spec.json").read_text(encoding="utf-8"))
        feature_cfg = FeatureConfig(**json.loads((directory / "feature_cfg.json").read_text(encoding="utf-8")))
        model_cfg = ModelConfig(**json.loads((directory / "model_cfg.json").read_text(encoding="utf-8")))
        training_cfg = TrainingConfig(**json.loads((directory / "training_cfg.json").read_text(encoding="utf-8")))
        versions = json.loads((directory / "versions.json").read_text(encoding="utf-8"))

        # Load preprocessor
        preprocessor, _ = Preprocessor.load(directory / "preprocessor")

        # Load model based on recorded type
        model_type = (directory / "model" / "model_type.txt").read_text(encoding="utf-8").strip()
        if model_type == "random_forest":
            model = RFRegressor.load(directory / "model")
        else:
            raise ValueError(f"Unsupported model_type in bundle: {model_type}")

        return TrainedBundle(
            model=model,
            preprocessor=preprocessor,
            feature_spec=feature_spec,
            feature_cfg=feature_cfg,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            versions=versions,
        )



