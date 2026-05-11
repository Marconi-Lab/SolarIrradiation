"""Trainer end-to-end (offline only — W&B path is not exercised here).

Pins:
* fit / score / bundle round-trips work without W&B
* metadata records the splitter name, holdout label, and baseline scores
* invalid splits surface clear errors
* `bundle_dest=` writes the bundle to disk; loading it reproduces predictions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from susse.models import LinearParams, MeanBaselineParams, RandomForestParams
from susse.preprocessing import PreprocessedDataset
from susse.training import (
    Trainer,
    TrainerConfig,
    load_bundle,
    make_station_loso_splitter,
)


def _train_one(
    processed: PreprocessedDataset,
    params: Any,
    *,
    held_out: str = "sta_b",
    bundle_dest: Path | None = None,
):
    splitter = make_station_loso_splitter(held_out)
    return Trainer().train(
        processed=processed,
        params=params,
        splitter=splitter,
        holdout_label=f"station-LOSO:{held_out}",
        bundle_dest=bundle_dest,
    )


class TestEndToEnd:
    def test_returns_bundle_with_full_metadata(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        bundle = _train_one(
            toy_processed,
            RandomForestParams(n_estimators=10, max_depth=3, random_state=0),
        )
        # Splitter name was recorded.
        assert bundle.metadata.splitter_name == "station_loso[sta_b]"
        # Baselines were scored on the val fold (both NASA + CAMS columns
        # exist in the toy fixture, so both should be present).
        assert set(bundle.metadata.baseline_metrics) == {
            "sat_ghi_nasa_kwh_m2_day",
            "sat_ghi_cams_kwh_m2_day",
        }
        # Source manifest is carried verbatim.
        assert bundle.source_manifest == toy_processed.source_manifest

    def test_bundle_dest_persists_and_reloads(
        self,
        tmp_path: Path,
        toy_processed: PreprocessedDataset,
    ) -> None:
        bundle = _train_one(
            toy_processed,
            LinearParams(with_scaling=True),
            bundle_dest=tmp_path,
        )
        # The dir holds all four pieces.
        assert (tmp_path / "model").is_dir()
        assert (tmp_path / "feature_spec.json").is_file()
        assert (tmp_path / "metadata.json").is_file()
        assert (tmp_path / "source_manifest.json").is_file()

        # Reloaded bundle reproduces predictions.
        restored = load_bundle(tmp_path)
        X = toy_processed.X()
        np.testing.assert_allclose(
            restored.regressor.predict(X).values,
            bundle.regressor.predict(X).values,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_n_train_n_val_match_split(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        bundle = _train_one(toy_processed, MeanBaselineParams())
        # Toy fixture: 60 rows per station, 2 stations → 60 / 60.
        assert bundle.metadata.n_train_rows == 60
        assert bundle.metadata.n_val_rows == 60


class TestInvalidSplits:
    def test_empty_train_fold_raises(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        # A splitter that puts every row into val.
        def all_into_val(p: PreprocessedDataset):
            return p.df.index[:0], p.df.index

        all_into_val.__name__ = "all_into_val"

        with pytest.raises(ValueError, match="empty train fold"):
            Trainer().train(
                processed=toy_processed,
                params=MeanBaselineParams(),
                splitter=all_into_val,
                holdout_label="degenerate",
            )

    def test_overlapping_indices_raises(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        def overlap(p: PreprocessedDataset):
            return p.df.index, p.df.index  # 100% overlap

        overlap.__name__ = "overlap"

        with pytest.raises(ValueError, match="overlapping"):
            Trainer().train(
                processed=toy_processed,
                params=MeanBaselineParams(),
                splitter=overlap,
                holdout_label="degenerate",
            )


class TestBaselineScoringWithMissingColumn:
    def test_missing_baseline_column_is_silently_skipped(
        self,
        toy_processed: PreprocessedDataset,
    ) -> None:
        """A baseline column not in processed.df is just absent from metadata."""
        bundle = Trainer(
            TrainerConfig(
                baseline_columns=("does_not_exist", "sat_ghi_nasa_kwh_m2_day")
            ),
        ).train(
            processed=toy_processed,
            params=MeanBaselineParams(),
            splitter=make_station_loso_splitter("sta_b"),
            holdout_label="station-LOSO:sta_b",
        )
        # Only the existing column got scored.
        assert set(bundle.metadata.baseline_metrics) == {"sat_ghi_nasa_kwh_m2_day"}
