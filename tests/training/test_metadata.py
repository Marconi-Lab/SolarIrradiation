"""TrainingMetadata + ScoreSet validation and JSON-roundtrip tests."""

from __future__ import annotations

import pytest

from susse.training import ScoreSet, TrainingMetadata


def _toy_score(n: int = 10, mae: float = 0.5) -> ScoreSet:
    return ScoreSet(n_rows=n, mae=mae, rmse=mae * 1.2, r2=0.8)


class TestScoreSet:
    def test_negative_n_rows_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"n_rows=-1"):
            ScoreSet(n_rows=-1, mae=0.0, rmse=0.0, r2=0.0)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = _toy_score(n=42, mae=0.37)
        recovered = ScoreSet.from_dict(original.to_dict())
        assert recovered == original


class TestTrainingMetadata:
    def _full(self) -> TrainingMetadata:
        return TrainingMetadata(
            created_at_utc="2026-05-10T00:00:00+00:00",
            susse_version="test-0.0.0",
            git_sha=None,
            holdout_label="station-LOSO:sta_b",
            splitter_name="station_loso[sta_b]",
            n_train_rows=60,
            n_val_rows=60,
            train_metrics=_toy_score(60, 0.4),
            val_metrics=_toy_score(60, 0.5),
            baseline_metrics={
                "sat_ghi_nasa_kwh_m2_day": _toy_score(60, 0.7),
                "sat_ghi_cams_kwh_m2_day": _toy_score(60, 0.6),
            },
        )

    def test_empty_holdout_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="holdout_label must be non-empty"):
            TrainingMetadata(
                created_at_utc="t", susse_version="v",
                git_sha=None, holdout_label="",
                splitter_name="x",
                n_train_rows=1, n_val_rows=1,
                train_metrics=_toy_score(),
                val_metrics=_toy_score(),
                baseline_metrics={},
            )

    def test_empty_splitter_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="splitter_name must be non-empty"):
            TrainingMetadata(
                created_at_utc="t", susse_version="v",
                git_sha=None, holdout_label="x",
                splitter_name="",
                n_train_rows=1, n_val_rows=1,
                train_metrics=_toy_score(),
                val_metrics=_toy_score(),
                baseline_metrics={},
            )

    def test_negative_n_train_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"n_train_rows=-1"):
            TrainingMetadata(
                created_at_utc="t", susse_version="v",
                git_sha=None, holdout_label="x",
                splitter_name="y",
                n_train_rows=-1, n_val_rows=1,
                train_metrics=_toy_score(),
                val_metrics=_toy_score(),
                baseline_metrics={},
            )

    def test_json_roundtrip_preserves_baselines(self) -> None:
        original = self._full()
        recovered = TrainingMetadata.from_json(original.to_json())
        assert recovered == original
        # Baseline keys survive verbatim (dict ordering is irrelevant for equality).
        assert set(recovered.baseline_metrics.keys()) == {
            "sat_ghi_nasa_kwh_m2_day", "sat_ghi_cams_kwh_m2_day",
        }
