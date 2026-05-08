"""Typed config describing the preprocessing pipeline.

A :class:`FeatureSpec` is to the :class:`Preprocessor` what
:class:`susse.datasets.FeatureSelection` is to :class:`FeatureService`:
the recipe — which transforms to apply, which columns to project, what
the target is. Persisted as JSON alongside any model so inference
applies identical transformations.

Validation against a concrete :class:`~susse.datasets.TrainingDataset`
happens at apply time (in :class:`Preprocessor`), not at construction.
That way the same spec can be reused across multiple compatible
datasets without enumerating their column shapes here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClearSkyIndexSpec:
    """One ``kt = ghi / ghi_clear`` feature derivation.

    Attributes:
        ghi_column: Column name carrying the satellite-estimated GHI
            (e.g. ``"sat_ghi_nasa_kwh_m2_day"``).
        ghi_clear_column: Column name carrying the clear-sky reference
            (e.g. ``"nasa_ghi_clear"``).
        output_column: Name of the resulting kt column (e.g.
            ``"kt_nasa"``). Must be unique across the spec.
    """

    ghi_column: str
    ghi_clear_column: str
    output_column: str

    def __post_init__(self) -> None:
        for name, value in (
            ("ghi_column", self.ghi_column),
            ("ghi_clear_column", self.ghi_clear_column),
            ("output_column", self.output_column),
        ):
            if not value:
                raise ValueError(
                    f"ClearSkyIndexSpec.{name} must be non-empty."
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "ghi_column": self.ghi_column,
            "ghi_clear_column": self.ghi_clear_column,
            "output_column": self.output_column,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "ClearSkyIndexSpec":
        return cls(
            ghi_column=d["ghi_column"],
            ghi_clear_column=d["ghi_clear_column"],
            output_column=d["output_column"],
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Recipe for the preprocessing pipeline.

    Attributes:
        target_column: Name of the target column in the input
            DataFrame. Default ``"y_ghi_kwh_m2_day"`` matches the output
            of :meth:`FeatureService.build_training_pairs`.
        feature_columns: Pass-through columns from the input dataset to
            keep as model inputs. Validated to exist in the input at
            apply time.
        clear_sky_index_specs: Tuple of :class:`ClearSkyIndexSpec`s
            describing kt features to derive. Empty tuple skips kt
            entirely.
        include_cyclical_doy: If True, derive ``doy_sin`` / ``doy_cos``
            from the ``date`` column.
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
    clear_sky_index_specs: tuple[ClearSkyIndexSpec, ...] = ()
    include_cyclical_doy: bool = True
    id_columns: tuple[str, ...] = ("date", "location", "geohash5")
    dropna_target: bool = True
    dropna_features: bool = True

    def __post_init__(self) -> None:
        if not self.target_column:
            raise ValueError("target_column must be non-empty.")
        # Output names must be unique across kt specs.
        kt_outs = [s.output_column for s in self.clear_sky_index_specs]
        if len(set(kt_outs)) != len(kt_outs):
            raise ValueError(
                f"clear_sky_index_specs have duplicate output_column names: "
                f"{sorted(kt_outs)}. Each derived feature must have a "
                f"unique name."
            )
        # Output names must not collide with the pass-through features.
        feature_set = set(self.feature_columns)
        collisions = feature_set & set(kt_outs)
        if collisions:
            raise ValueError(
                f"clear_sky_index_specs.output_column collides with "
                f"feature_columns: {sorted(collisions)}. Either rename the "
                f"derived output or remove it from feature_columns."
            )
        if self.include_cyclical_doy:
            cyclical = {"doy_sin", "doy_cos"}
            cy_collisions = feature_set & cyclical
            if cy_collisions:
                raise ValueError(
                    f"include_cyclical_doy=True would produce {sorted(cyclical)}, "
                    f"but feature_columns already contains {sorted(cy_collisions)}. "
                    f"Remove the duplicate from feature_columns."
                )

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        """Ordered list of model-input columns the preprocessor will produce.

        Order: pass-through features → kt features → cyclical doy. This
        is the column order downstream models should expect.
        """
        names: list[str] = list(self.feature_columns)
        names.extend(s.output_column for s in self.clear_sky_index_specs)
        if self.include_cyclical_doy:
            names.extend(("doy_sin", "doy_cos"))
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "clear_sky_index_specs": [
                s.to_dict() for s in self.clear_sky_index_specs
            ],
            "include_cyclical_doy": self.include_cyclical_doy,
            "id_columns": list(self.id_columns),
            "dropna_target": self.dropna_target,
            "dropna_features": self.dropna_features,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureSpec":
        return cls(
            target_column=d.get("target_column", "y_ghi_kwh_m2_day"),
            feature_columns=tuple(d.get("feature_columns", ())),
            clear_sky_index_specs=tuple(
                ClearSkyIndexSpec.from_dict(s)
                for s in d.get("clear_sky_index_specs", ())
            ),
            include_cyclical_doy=d.get("include_cyclical_doy", True),
            id_columns=tuple(d.get("id_columns", ("date", "location", "geohash5"))),
            dropna_target=d.get("dropna_target", True),
            dropna_features=d.get("dropna_features", True),
        )

    @classmethod
    def from_json(cls, s: str) -> "FeatureSpec":
        return cls.from_dict(json.loads(s))
