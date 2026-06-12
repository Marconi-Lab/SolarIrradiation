"""Round-trip tests for the local snapshot writer + reader.

Specifically pins:
* parquet + manifest land in the dest dir,
* manifest's content_hash is set to the actual SHA-256 of the parquet,
* load_snapshot validates the hash and raises on mismatch,
* missing files surface clear FileNotFoundError messages.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from susse.datasets import (
    DatasetManifest,
    FeatureSelection,
    load_snapshot,
    write_snapshot,
)


def _toy_manifest() -> DatasetManifest:
    return DatasetManifest(
        name="snapshot_io_test",
        version="v0",
        created_at_utc=datetime(2026, 5, 8, tzinfo=timezone.utc).isoformat(),
        susse_version="test-0.0.0",
        git_sha=None,
        feature_selection=FeatureSelection(),
        date_start=date(2024, 1, 1),
        date_end=date(2024, 1, 2),
        location_filter=None,
        warehouse_project="test-project",
        warehouse_dataset="test_dataset",
        warehouse_table_mods={},
        n_rows=2,
        n_cols=2,
        column_names=("date", "value"),
        content_hash="",  # filled in by write_snapshot
    )


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": [date(2024, 1, 1), date(2024, 1, 2)], "value": [1.5, 2.5]}
    )


class TestRoundtrip:
    def test_write_then_load_preserves_dataframe(self, tmp_path: Path) -> None:
        df = _toy_df()
        manifest = _toy_manifest()
        written = write_snapshot(df=df, manifest=manifest, dest=tmp_path)

        # write_snapshot stamps content_hash; original manifest had ""
        assert written.manifest.content_hash != ""

        # Re-read via load_snapshot and compare.
        loaded = load_snapshot(tmp_path)
        pd.testing.assert_frame_equal(loaded.df, df)
        assert loaded.manifest == written.manifest

    def test_files_are_written_in_dest(self, tmp_path: Path) -> None:
        write_snapshot(df=_toy_df(), manifest=_toy_manifest(), dest=tmp_path)
        assert (tmp_path / "dataset.parquet").exists()
        assert (tmp_path / "manifest.json").exists()

    def test_load_validates_content_hash(self, tmp_path: Path) -> None:
        # Tamper with the parquet after writing — load must catch it.
        write_snapshot(df=_toy_df(), manifest=_toy_manifest(), dest=tmp_path)
        with (tmp_path / "dataset.parquet").open("ab") as f:
            f.write(b"trailing-garbage")
        with pytest.raises(ValueError, match="Content-hash mismatch"):
            load_snapshot(tmp_path)


class TestMissingFiles:
    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")  # parquet absent
        with pytest.raises(FileNotFoundError, match="dataset.parquet"):
            load_snapshot(tmp_path)

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        # Write only the parquet — no manifest. load must surface the
        # missing-file before attempting to deserialize.
        df = _toy_df()
        df.to_parquet(tmp_path / "dataset.parquet", index=False)
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            load_snapshot(tmp_path)


class TestManifestRoundtrip:
    def test_manifest_to_json_from_json_roundtrip(self) -> None:
        original = _toy_manifest()
        recovered = DatasetManifest.from_json(original.to_json())
        assert recovered == original

    def test_manifest_with_content_hash_is_immutable_copy(self) -> None:
        original = _toy_manifest()
        stamped = original.with_content_hash("abc123")
        assert original.content_hash == ""
        assert stamped.content_hash == "abc123"
        # All other fields preserved.
        assert stamped.name == original.name
        assert stamped.feature_selection == original.feature_selection
