"""Stateless preprocessor: training datasets or inference frames → features.

The :class:`Preprocessor` applies a :class:`FeatureSpec` to either a
:class:`susse.datasets.TrainingDataset` (yielding a
:class:`PreprocessedDataset` with the source manifest carried through)
or a bare :class:`pandas.DataFrame` (yielding a transformed
:class:`pandas.DataFrame` for inference). All transformations are
deterministic given the inputs — no fit/transform asymmetry, no
internal state to worry about leaking across folds.

The transform pipeline is **single-loop over
``spec.derived_features``**: each :class:`DerivedFeature` declares
its own output columns and computes them. The preprocessor itself
knows nothing about kt vs cyclical-doy vs altitude — adding a new
feature type is a new ``DerivedFeature`` subclass, with no edits here.

Scaling deliberately lives in the model wrapper, not here: tree-based
models (RF / XGBoost) don't need it; neural-network models do, and only
they know which columns to scale + on which fold to fit. Keeping the
preprocessor stateless means the same instance can transform train,
val, and test data with no risk of cross-fold leakage from a fitted
scaler.

Training vs inference shape (the only difference between the two entry
points):

* :meth:`apply` runs cleaners and applies the spec's NaN-drop policies.
  It expects a :class:`TrainingDataset` so the source manifest can
  travel into the resulting :class:`PreprocessedDataset` for downstream
  artifact provenance.
* :meth:`apply_to_dataframe` skips cleaners and NaN-drop. Inference is
  "compute the features as defined; touch nothing else" — silently
  dropping inference rows because some aux column is NaN would surface
  as missing predictions and would be far worse than NaN predictions
  the caller can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from ..datasets import DatasetManifest, TrainingDataset
from .feature_spec import FeatureSpec


@dataclass(frozen=True)
class PreprocessedDataset:
    """Result of applying a :class:`FeatureSpec` to a :class:`TrainingDataset`.

    Attributes:
        df: Materialised DataFrame with id columns + derived features +
            target. Column order: ``id_columns`` first, then
            ``output_feature_names``, then ``target_column``.
        feature_columns: Ordered tuple of model-input column names. The
            consuming model uses ``df[list(feature_columns)]`` to slice
            X; this avoids string-keyed indexing or positional drift.
        target_column: Name of the target column in ``df``.
        feature_spec: The :class:`FeatureSpec` that produced ``df``.
            Persisted so inference can recreate identical
            transformations.
        source_manifest: The full :class:`DatasetManifest` of the source
            :class:`TrainingDataset`. Carrying it verbatim (rather than
            just name + version + hash) lets downstream artifacts
            (e.g. trained-model bundles in NB 05) be fully self-
            describing without a separate dataset-resolution step.
        created_at_utc: ISO 8601 timestamp of preprocessing.
    """

    df: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    feature_spec: FeatureSpec
    source_manifest: DatasetManifest
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def source_dataset_name(self) -> str:
        return self.source_manifest.name

    @property
    def source_dataset_version(self) -> str:
        return self.source_manifest.version

    @property
    def source_content_hash(self) -> str:
        return self.source_manifest.content_hash

    def X(self) -> pd.DataFrame:
        """Model-input slice: ``df[feature_columns]`` in the spec's order."""
        return self.df[list(self.feature_columns)]

    def y(self) -> pd.Series:
        """Target slice."""
        return self.df[self.target_column]


class Preprocessor:
    """Apply a :class:`FeatureSpec` to training datasets or inference frames.

    Construct once with a spec; call :meth:`apply` per training dataset
    or :meth:`apply_to_dataframe` per inference frame. Stateless between
    calls — the same instance can transform train, val, and test folds
    without fit-time data leakage (because there is no fit time).

    Provider-style runtime dependencies (e.g. an elevation lookup
    service) are captured by individual :class:`DerivedFeature`
    instances at *their* construction. The preprocessor knows nothing
    about provider wiring; that's a concern of the spec's authors.
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> FeatureSpec:
        return self._spec

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def apply(self, dataset: TrainingDataset) -> PreprocessedDataset:
        """Transform a :class:`TrainingDataset` into a :class:`PreprocessedDataset`.

        Runs cleaners and applies the spec's NaN-drop policies — the
        training-shape transform. The source manifest is carried through
        for downstream artifact provenance.
        """
        out = self._apply_core(dataset.df, training_shape=True)
        return PreprocessedDataset(
            df=out,
            feature_columns=self._spec.output_feature_names,
            target_column=self._spec.target_column,
            feature_spec=self._spec,
            source_manifest=dataset.manifest,
        )

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a bare DataFrame — the inference-shape transform.

        Skips cleaners and NaN-drop policies. The target column is only
        copied through if present; inference frames typically have no
        ground truth, in which case the output simply has no target
        column.

        Returns:
            DataFrame with id columns + features + (optionally) target.
            Column order matches :meth:`apply`: ``id_columns`` first,
            then ``feature_columns`` in spec order, then the target
            column if present.
        """
        return self._apply_core(df, training_shape=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_core(self, df: pd.DataFrame, *, training_shape: bool) -> pd.DataFrame:
        """Single transform implementation behind both public entry points.

        Args:
            df: Input frame.
            training_shape: If True, run cleaners and NaN-drop policies.
                If False (inference), skip them — the rationale is in
                the module docstring.
        """
        spec = self._spec
        self._validate_columns(df, training_shape=training_shape)

        # Cleaners — training-shape only. Each cleaner gets the *current*
        # frame, so a cleaner downstream of another one sees the previous
        # one's filtered output.
        if training_shape:
            for cleaner in spec.cleaners:
                df = cleaner.apply(df)

        out = pd.DataFrame(index=df.index)

        # Pass-through id columns (kept for traceability, never given to model).
        for col in spec.id_columns:
            if col in df.columns:
                out[col] = df[col]
            # Missing id columns are non-fatal — not all datasets carry
            # every id (e.g. inference frames have no `location`). Skip.

        # Pass-through model features.
        for col in spec.feature_columns:
            out[col] = df[col]

        # Derived features — each computes its own output columns.
        for derived in spec.derived_features:
            result = derived.compute(df)
            for col in result.columns:
                out[col] = result[col]

        # Target — copied through if present; inference frames typically
        # have no ground truth, in which case this is a no-op.
        if spec.target_column in df.columns:
            out[spec.target_column] = df[spec.target_column]

        # NaN-drop policy — training-shape only.
        if training_shape:
            if spec.dropna_target and spec.target_column in out.columns:
                out = out.dropna(subset=[spec.target_column])
            if spec.dropna_features:
                feature_cols_present = [
                    c for c in spec.output_feature_names if c in out.columns
                ]
                if feature_cols_present:
                    out = out.dropna(subset=feature_cols_present)

        return out.reset_index(drop=True)

    def _validate_columns(self, df: pd.DataFrame, *, training_shape: bool) -> None:
        """Raise a clear error if the dataset is missing required columns.

        Validation is up-front so a misconfigured spec fails before any
        transforms run, with a message naming the missing column and
        which part of the :class:`FeatureSpec` referenced it.

        In inference shape, cleaner input columns and the
        ``dropna_target`` target requirement are skipped — those are
        training-shape concerns.
        """
        spec = self._spec
        missing: list[tuple[str, str]] = []  # (column, source_field)
        for col in spec.feature_columns:
            if col not in df.columns:
                missing.append((col, "feature_columns"))
        if training_shape:
            for cleaner in spec.cleaners:
                for col in cleaner.required_input_columns:
                    if col not in df.columns:
                        missing.append(
                            (
                                col,
                                f"cleaners[{cleaner.kind.value}]"
                                f".required_input_columns",
                            ),
                        )
        for derived in spec.derived_features:
            for col in derived.required_input_columns:
                if col not in df.columns:
                    missing.append(
                        (
                            col,
                            f"derived_features[{derived.kind.value}]"
                            f".required_input_columns",
                        ),
                    )
        # The target may legitimately be missing for inference frames; a
        # missing target is only an error in training shape if
        # dropna_target is True (which forces the column to exist).
        if (
            training_shape
            and spec.dropna_target
            and spec.target_column not in df.columns
        ):
            missing.append(
                (spec.target_column, "target_column (with dropna_target=True)")
            )
        if missing:
            details = "; ".join(
                f"{col!r} (referenced by {field})" for col, field in missing
            )
            raise ValueError(
                f"Preprocessor: input DataFrame is missing required "
                f"column(s): {details}. Either rebuild the upstream "
                f"input frame to include them, or amend the FeatureSpec."
            )
