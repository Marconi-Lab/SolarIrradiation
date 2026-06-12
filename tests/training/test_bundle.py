"""TrainedBundle save/load roundtrip + missing-file diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from susse.datasets import DatasetManifest
from susse.models import (
    LinearParams,
    MeanBaselineParams,
    ModelFactory,
    RandomForestParams,
)
from susse.preprocessing import PreprocessedDataset
from susse.training import ScoreSet, TrainedBundle, TrainingMetadata, load_bundle


def _toy_metadata() -> TrainingMetadata:
    s = ScoreSet(n_rows=10, mae=0.5, rmse=0.6, r2=0.8)
    return TrainingMetadata(
        created_at_utc="2026-05-10T00:00:00+00:00",
        susse_version="test",
        git_sha=None,
        holdout_label="station-LOSO:sta_b",
        splitter_name="station_loso[sta_b]",
        n_train_rows=60,
        n_val_rows=60,
        train_metrics=s,
        val_metrics=s,
        baseline_metrics={"sat_ghi_nasa_kwh_m2_day": s},
    )


@pytest.mark.parametrize(
    "params",
    [
        MeanBaselineParams(),
        RandomForestParams(n_estimators=10, max_depth=4, random_state=0),
        LinearParams(with_scaling=True),
    ],
    ids=["mean_baseline", "random_forest", "linear"],
)
def test_save_load_roundtrip_preserves_predictions(
    params,
    tmp_path: Path,
    toy_processed: PreprocessedDataset,
) -> None:
    """Same predictions before and after save/load, across all model kinds."""
    X, y = toy_processed.X(), toy_processed.y()
    regressor = ModelFactory.create(params).fit(X, y)
    bundle = TrainedBundle(
        regressor=regressor,
        feature_spec=toy_processed.feature_spec,
        metadata=_toy_metadata(),
        source_manifest=toy_processed.source_manifest,
    )
    bundle.save(tmp_path)
    restored = load_bundle(tmp_path)

    np.testing.assert_allclose(
        restored.regressor.predict(X).values,
        regressor.predict(X).values,
        rtol=1e-10,
        atol=1e-10,
    )
    assert restored.feature_spec == bundle.feature_spec
    assert restored.metadata == bundle.metadata
    assert restored.source_manifest == bundle.source_manifest


class TestMissingFiles:
    def _bundled_dir(
        self,
        tmp_path: Path,
        toy_processed: PreprocessedDataset,
    ) -> Path:
        regressor = ModelFactory.create(MeanBaselineParams()).fit(
            toy_processed.X(),
            toy_processed.y(),
        )
        TrainedBundle(
            regressor=regressor,
            feature_spec=toy_processed.feature_spec,
            metadata=_toy_metadata(),
            source_manifest=toy_processed.source_manifest,
        ).save(tmp_path)
        return tmp_path

    def test_missing_metadata_surfaces_clear_error(
        self,
        tmp_path: Path,
        toy_processed: PreprocessedDataset,
    ) -> None:
        d = self._bundled_dir(tmp_path, toy_processed)
        (d / "metadata.json").unlink()
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            load_bundle(d)

    def test_missing_source_manifest_surfaces_clear_error(
        self,
        tmp_path: Path,
        toy_processed: PreprocessedDataset,
    ) -> None:
        d = self._bundled_dir(tmp_path, toy_processed)
        (d / "source_manifest.json").unlink()
        with pytest.raises(FileNotFoundError, match="source_manifest.json"):
            load_bundle(d)

    def test_missing_model_dir_surfaces_clear_error(
        self,
        tmp_path: Path,
        toy_processed: PreprocessedDataset,
    ) -> None:
        d = self._bundled_dir(tmp_path, toy_processed)
        # Wipe the model subdir.
        for f in (d / "model").iterdir():
            f.unlink()
        (d / "model").rmdir()
        with pytest.raises(FileNotFoundError, match="'model'"):
            load_bundle(d)


class TestSourceManifestEmbedding:
    def test_bundle_carries_full_source_manifest_verbatim(
        self,
        tmp_path: Path,
        toy_processed: PreprocessedDataset,
    ) -> None:
        """Embedded manifest survives byte-for-byte across save/load."""
        regressor = ModelFactory.create(MeanBaselineParams()).fit(
            toy_processed.X(),
            toy_processed.y(),
        )
        bundle = TrainedBundle(
            regressor=regressor,
            feature_spec=toy_processed.feature_spec,
            metadata=_toy_metadata(),
            source_manifest=toy_processed.source_manifest,
        )
        bundle.save(tmp_path)
        # Direct file inspection: manifest JSON equals original.
        on_disk = DatasetManifest.from_json(
            (tmp_path / "source_manifest.json").read_text()
        )
        assert on_disk == toy_processed.source_manifest
