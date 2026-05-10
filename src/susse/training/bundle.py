"""TrainedBundle — a fully self-describing trained-model artifact.

A bundle is the *only* thing inference needs: it carries the fitted
model, the :class:`FeatureSpec` that produced its inputs, the training
metadata, and the source dataset's :class:`DatasetManifest` verbatim.
Loading a bundle reconstructs every transform identically; no separate
dataset-resolution or registry lookup is required.

On-disk layout::

    <bundle_dir>/
        model/                 # output of BaseRegressor.save()
            model_kind.txt
            params.json
            state.joblib
        feature_spec.json
        metadata.json
        source_manifest.json

The layout is W&B-artifact-friendly: ``add_dir(bundle_dir)`` uploads
all four pieces in one artifact of type ``"trained_model"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..datasets import DatasetManifest
from ..models import BaseRegressor, load_regressor
from ..preprocessing import FeatureSpec
from .metadata import TrainingMetadata

_MODEL_SUBDIR = "model"
_FEATURE_SPEC_FILENAME = "feature_spec.json"
_METADATA_FILENAME = "metadata.json"
_SOURCE_MANIFEST_FILENAME = "source_manifest.json"


@dataclass(frozen=True)
class TrainedBundle:
    """Fitted regressor + everything inference needs to reproduce its inputs.

    Attributes:
        regressor: The fitted :class:`BaseRegressor`. Must be fitted
            before :meth:`save`.
        feature_spec: The :class:`FeatureSpec` whose
            ``output_feature_names`` ordering matches what the
            regressor was fit against.
        metadata: Training-time provenance + score sets.
        source_manifest: Manifest of the source training dataset,
            verbatim. Embedding it makes the bundle self-describing
            even if the original W&B dataset artifact is later
            deleted.
    """

    regressor: BaseRegressor
    feature_spec: FeatureSpec
    metadata: TrainingMetadata
    source_manifest: DatasetManifest

    def save(self, dest: Path) -> None:
        """Persist the bundle into ``dest`` (created if absent).

        Raises:
            RuntimeError: If the regressor has not been fitted.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self.regressor.save(dest / _MODEL_SUBDIR)
        (dest / _FEATURE_SPEC_FILENAME).write_text(self.feature_spec.to_json())
        (dest / _METADATA_FILENAME).write_text(self.metadata.to_json())
        (dest / _SOURCE_MANIFEST_FILENAME).write_text(
            self.source_manifest.to_json()
        )


def load_bundle(src: Path) -> TrainedBundle:
    """Inverse of :meth:`TrainedBundle.save`.

    Reads the four bundle files, dispatches the regressor via its
    persisted ``model_kind.txt`` (no caller-side knowledge of which
    flavour was trained), and returns the fully reconstructed bundle.

    Raises:
        FileNotFoundError: If any of the four pieces is missing. The
            message names the missing file so the caller can spot
            partial / corrupted bundles before they reach inference.
    """
    src = Path(src)
    for required in (
        _MODEL_SUBDIR,
        _FEATURE_SPEC_FILENAME,
        _METADATA_FILENAME,
        _SOURCE_MANIFEST_FILENAME,
    ):
        path = src / required
        if not path.exists():
            raise FileNotFoundError(
                f"Bundle at {src} is incomplete: missing {required!r}. "
                f"A valid bundle has model/, feature_spec.json, "
                f"metadata.json, and source_manifest.json."
            )
    regressor = load_regressor(src / _MODEL_SUBDIR)
    feature_spec = FeatureSpec.from_json(
        (src / _FEATURE_SPEC_FILENAME).read_text()
    )
    metadata = TrainingMetadata.from_json(
        (src / _METADATA_FILENAME).read_text()
    )
    source_manifest = DatasetManifest.from_json(
        (src / _SOURCE_MANIFEST_FILENAME).read_text()
    )
    return TrainedBundle(
        regressor=regressor,
        feature_spec=feature_spec,
        metadata=metadata,
        source_manifest=source_manifest,
    )
