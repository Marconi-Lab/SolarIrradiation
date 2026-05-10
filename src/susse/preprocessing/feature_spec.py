"""Typed config describing the preprocessing pipeline.

A :class:`FeatureSpec` is to the :class:`Preprocessor` what
:class:`susse.datasets.FeatureSelection` is to :class:`FeatureService`:
the recipe — which transforms to apply, which columns to project, what
the target is. Persisted as JSON alongside any model so inference
applies identical transformations.

Two ways to add a feature:

* **Pass-through** — name the column in :attr:`feature_columns`. The
  column must already exist on the input DataFrame.
* **Derived** — append a :class:`~susse.preprocessing.DerivedFeature`
  instance to :attr:`derived_features`. The feature knows what columns
  it needs as input, what columns it produces, and how to compute
  itself. Adding a new derived-feature type is one new ``DerivedFeature``
  subclass; this dataclass and the :class:`Preprocessor` do not change.

Row-level data cleaning (filter, impute) is configured through
:attr:`cleaners`, applied **before** any derived-feature computation.
See :class:`~susse.preprocessing.DataCleaner`.

Validation against a concrete :class:`~susse.datasets.TrainingDataset`
happens at apply time (in :class:`Preprocessor`), not at construction.
That way the same spec can be reused across multiple compatible
datasets without enumerating their column shapes here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from .cleaning import DataCleaner, data_cleaner_from_dict
from .derived import DerivedFeature, derived_feature_from_dict


@dataclass(frozen=True)
class FeatureSpec:
    """Recipe for the preprocessing pipeline.

    Attributes:
        target_column: Name of the target column in the input
            DataFrame. Default ``"y_ghi_kwh_m2_day"`` matches the output
            of :meth:`FeatureService.build_training_pairs`.
        feature_columns: Pass-through columns from the input dataset to
            keep verbatim as model inputs. Validated to exist in the
            input at apply time.
        cleaners: Tuple of :class:`~susse.preprocessing.DataCleaner`
            instances applied to the input frame *before* feature
            computation. Each may filter rows or impute values; column
            schema is preserved. Default ``()`` is the historical
            behaviour (no row-level cleaning).
        derived_features: Tuple of
            :class:`~susse.preprocessing.DerivedFeature` instances.
            Each declares its own output columns, required input
            columns, and how to compute itself; the preprocessor
            iterates this tuple in order and emits the columns
            unchanged. Concrete subclasses today:
            :class:`~susse.preprocessing.ClearSkyIndexFeature`,
            :class:`~susse.preprocessing.CyclicalDayOfYearFeature`,
            :class:`~susse.preprocessing.AltitudeFeature`,
            :class:`~susse.preprocessing.LongitudeFeature`.
        id_columns: Non-feature, non-target columns to keep in the
            output DataFrame for traceability (e.g. ``date``,
            ``location``, ``geohash5``). The model never sees these;
            they stay so downstream tools can join on them.
        dropna_target: If True (default), rows with NaN target are
            dropped. Set False only for inference (no target).
        dropna_features: If True (default), rows with any NaN feature
            (after derivation) are dropped. The simple-and-safe default
            for v1 — imputation can be added later if the data argues
            for it.
    """

    target_column: str = "y_ghi_kwh_m2_day"
    feature_columns: tuple[str, ...] = ()
    cleaners: tuple[DataCleaner, ...] = ()
    derived_features: tuple[DerivedFeature, ...] = ()
    id_columns: tuple[str, ...] = ("date", "location", "geohash5")
    dropna_target: bool = True
    dropna_features: bool = True

    def __post_init__(self) -> None:
        if not self.target_column:
            raise ValueError("target_column must be non-empty.")
        derived_outputs: list[str] = []
        for f in self.derived_features:
            derived_outputs.extend(f.output_columns)
        # No duplicate output names across all derived features.
        if len(set(derived_outputs)) != len(derived_outputs):
            seen: set[str] = set()
            dups = [c for c in derived_outputs if c in seen or seen.add(c)]  # type: ignore[func-returns-value]
            raise ValueError(
                f"derived_features produce duplicate output column "
                f"names: {sorted(set(dups))}. Each derived feature must "
                f"emit unique columns."
            )
        # No collision between derived outputs and pass-through features.
        feature_set = set(self.feature_columns)
        collisions = feature_set & set(derived_outputs)
        if collisions:
            raise ValueError(
                f"derived_features outputs collide with feature_columns: "
                f"{sorted(collisions)}. Either rename the derived output "
                f"or remove the duplicate from feature_columns."
            )

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        """Ordered list of model-input columns the preprocessor will produce.

        Order: pass-through ``feature_columns`` first, then the
        ``output_columns`` of each :class:`DerivedFeature` in the
        order they appear. This is the column order downstream models
        should expect.
        """
        names: list[str] = list(self.feature_columns)
        for f in self.derived_features:
            names.extend(f.output_columns)
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "cleaners": [c.to_dict() for c in self.cleaners],
            "derived_features": [f.to_dict() for f in self.derived_features],
            "id_columns": list(self.id_columns),
            "dropna_target": self.dropna_target,
            "dropna_features": self.dropna_features,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        providers: Optional[dict[str, Any]] = None,
    ) -> "FeatureSpec":
        """Reconstruct from a :meth:`to_dict` payload.

        Args:
            d: Output of :meth:`to_dict`.
            providers: Map of provider-key → provider for any
                :class:`DerivedFeature` that needs runtime injection
                at deserialisation time (e.g.
                ``{"altitude": PvlibElevationProvider()}``). Pass-through
                for features that don't need providers.
        """
        return cls(
            target_column=d.get("target_column", "y_ghi_kwh_m2_day"),
            feature_columns=tuple(d.get("feature_columns", ())),
            cleaners=tuple(
                data_cleaner_from_dict(c, providers=providers)
                for c in d.get("cleaners", ())
            ),
            derived_features=tuple(
                derived_feature_from_dict(f, providers=providers)
                for f in d.get("derived_features", ())
            ),
            id_columns=tuple(d.get("id_columns", ("date", "location", "geohash5"))),
            dropna_target=d.get("dropna_target", True),
            dropna_features=d.get("dropna_features", True),
        )

    @classmethod
    def from_json(
        cls,
        s: str,
        *,
        providers: Optional[dict[str, Any]] = None,
    ) -> "FeatureSpec":
        return cls.from_dict(json.loads(s), providers=providers)
