"""Feature catalog — human-facing metadata for a bundle's model inputs.

:meth:`Predictor.predict` returns every model-input column alongside the
prediction, but the column names alone (``nasa_temperature``,
``sat_ghi_cams_kwh_m2_day``, ``kt_cams``) are opaque to an end user. A
:class:`FeatureCatalog` pairs each of those columns with a label, unit,
description, and presentation group, so a UI — the portal's variable
inspector — can show what every input *is* and where it comes from.

The catalog unifies two metadata origins behind one :class:`FeatureMetadata`
shape:

* **Warehouse variables** — NASA POWER / CAMS aux variables and satellite
  irradiance bands. Metadata comes from the warehouse ``VariableCatalog``.
* **Derived features** — clear-sky index, cyclical day-of-year, altitude,
  longitude. Metadata comes from each :class:`DerivedFeature`'s
  ``output_metadata`` (see :class:`DerivedColumnMetadata`).

Building the catalog is pure: it needs only the bundle's
:class:`FeatureSelection` and :class:`FeatureSpec`, no warehouse access.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterator

from ..datasets import FeatureSelection
from ..preprocessing import DerivedColumnMetadata, FeatureSpec
from ..warehouse_ops.population.dim_variable import VariableCatalog
from ..warehouse_ops.population.types import (
    Source,
    VariableSpec,
    aux_feature_column,
    satellite_irradiance_column,
)


class FeatureGroup(StrEnum):
    """Presentation bucket a model-input feature belongs to.

    Distinct from :class:`Source`: a feature group is how the variable
    inspector groups its table, and :attr:`DERIVED` has no warehouse
    source at all. The enum *value* is the human-readable group label
    the UI renders.
    """

    NASA_POWER = "NASA POWER"
    CAMS = "CAMS"
    DERIVED = "Derived"

    @classmethod
    def from_source(cls, source: Source) -> "FeatureGroup":
        """Map a warehouse :class:`Source` to its presentation group.

        Args:
            source: The warehouse source of a catalog variable.

        Returns:
            The matching :class:`FeatureGroup`.

        Raises:
            ValueError: For sources the feature catalog does not cover
                (MERRA-2, MODIS). Those are not inputs of any published
                bundle; extend this mapping before serving one that
                uses them.
        """
        if source is Source.NASA_POWER:
            return cls.NASA_POWER
        if source is Source.CAMS:
            return cls.CAMS
        raise ValueError(
            f"Source {source.value} has no FeatureGroup. The feature "
            f"catalog covers NASA POWER and CAMS only — add a "
            f"FeatureGroup member and extend FeatureGroup.from_source "
            f"before serving a bundle with MERRA-2 / MODIS inputs."
        )


@dataclass(frozen=True)
class FeatureMetadata:
    """Human-facing description of one model-input column.

    Constructed via :meth:`from_variable_spec` (for warehouse variables)
    or :meth:`from_derived_column` (for derived features) so the two
    metadata origins converge on one shape. The variable inspector
    renders one table row per instance.

    Attributes:
        column: Name of the feature column in :meth:`Predictor.predict`
            output.
        label: Short human-readable name, e.g. ``"2m Air Temperature"``.
        unit: Physical unit, or ``"unitless"`` for dimensionless
            quantities.
        description: One- or two-sentence explanation of the variable.
        group: Presentation bucket (NASA POWER / CAMS / Derived).
    """

    column: str
    label: str
    unit: str
    description: str
    group: FeatureGroup

    @classmethod
    def from_variable_spec(
        cls, spec: VariableSpec, *, column: str
    ) -> "FeatureMetadata":
        """Build from a warehouse :class:`VariableCatalog` entry.

        The catalog's ``display_name`` becomes :attr:`label` — the two
        name the same thing; ``label`` is the portal-facing spelling
        shared with :class:`DerivedColumnMetadata`.

        Args:
            spec: The catalog entry for the variable.
            column: The feature-column name the variable lands in
                (the catalog spec does not itself know this — it
                depends on the source prefix / irradiance convention).
        """
        return cls(
            column=column,
            label=spec.display_name,
            unit=spec.unit,
            description=spec.description,
            group=FeatureGroup.from_source(spec.source),
        )

    @classmethod
    def from_derived_column(cls, meta: DerivedColumnMetadata) -> "FeatureMetadata":
        """Build from a derived feature's per-column metadata."""
        return cls(
            column=meta.column,
            label=meta.label,
            unit=meta.unit,
            description=meta.description,
            group=FeatureGroup.DERIVED,
        )


@dataclass(frozen=True)
class FeatureCatalog:
    """Ordered metadata for every model-input column a bundle consumes.

    Build via :meth:`build`. Ordering is NASA POWER aux variables, then
    CAMS aux variables, then satellite irradiance, then derived
    features — the inspector regroups by :attr:`FeatureMetadata.group`
    anyway, so this is for determinism, not display.

    Attributes:
        entries: One :class:`FeatureMetadata` per model-input column.
    """

    entries: tuple[FeatureMetadata, ...]

    def __iter__(self) -> Iterator[FeatureMetadata]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def columns(self) -> tuple[str, ...]:
        """The feature-column names, in catalog order."""
        return tuple(entry.column for entry in self.entries)

    def get(self, column: str) -> FeatureMetadata:
        """Return the metadata for one feature column.

        Args:
            column: A feature-column name.

        Raises:
            KeyError: If ``column`` is not a model input. The message
                lists the known columns.
        """
        for entry in self.entries:
            if entry.column == column:
                return entry
        raise KeyError(
            f"No feature metadata for column {column!r}. Known feature "
            f"columns: {sorted(self.columns)}."
        )

    @classmethod
    def build(
        cls,
        *,
        selection: FeatureSelection,
        feature_spec: FeatureSpec,
    ) -> "FeatureCatalog":
        """Compose the catalog from a bundle's selection and spec.

        Args:
            selection: The bundle's :class:`FeatureSelection` — names
                the NASA / CAMS warehouse variables and irradiance
                bands the model consumes.
            feature_spec: The bundle's :class:`FeatureSpec` — supplies
                the derived features and the authoritative model-input
                column list used for the consistency check.

        Returns:
            A :class:`FeatureCatalog` covering exactly
            ``feature_spec.output_feature_names``.

        Raises:
            NotImplementedError: If ``selection`` requests MERRA-2 or
                MODIS variables — not yet supported by the catalog.
            ValueError: If the assembled catalog does not match the
                spec's model inputs (an inconsistent bundle).
            KeyError: If a selected variable is absent from the
                warehouse ``VariableCatalog``.
        """
        if selection.merra_variable_ids or selection.modis_variable_ids:
            raise NotImplementedError(
                "FeatureCatalog.build covers NASA POWER and CAMS inputs "
                "only. This selection also requests MERRA-2 / MODIS "
                f"variables (merra={list(selection.merra_variable_ids)}, "
                f"modis={list(selection.modis_variable_ids)}). Extend "
                "FeatureCatalog.build and FeatureGroup.from_source before "
                "serving a bundle that uses them."
            )
        entries: list[FeatureMetadata] = []
        entries.extend(cls._aux_entries(selection))
        entries.extend(cls._irradiance_entries(selection))
        entries.extend(cls._derived_entries(feature_spec))
        catalog = cls(entries=tuple(entries))
        catalog._check_covers_model_inputs(feature_spec)
        return catalog

    @staticmethod
    def _aux_entries(selection: FeatureSelection) -> list[FeatureMetadata]:
        """Metadata for NASA POWER + CAMS auxiliary (non-irradiance) variables."""
        entries: list[FeatureMetadata] = []
        for source, variable_ids in (
            (Source.NASA_POWER, selection.nasa_variable_ids),
            (Source.CAMS, selection.cams_variable_ids),
        ):
            for variable_id in variable_ids:
                spec = VariableCatalog.get(variable_id=variable_id, source=source)
                entries.append(
                    FeatureMetadata.from_variable_spec(
                        spec, column=aux_feature_column(source, variable_id)
                    )
                )
        return entries

    @staticmethod
    def _irradiance_entries(selection: FeatureSelection) -> list[FeatureMetadata]:
        """Metadata for the satellite irradiance bands (sat_<band>_<source>)."""
        entries: list[FeatureMetadata] = []
        for source in selection.include_satellite_irradiance:
            for band in selection.include_satellite_bands:
                spec = VariableCatalog.get(variable_id=band.value, source=source)
                entries.append(
                    FeatureMetadata.from_variable_spec(
                        spec, column=satellite_irradiance_column(source, band)
                    )
                )
        return entries

    @staticmethod
    def _derived_entries(feature_spec: FeatureSpec) -> list[FeatureMetadata]:
        """Metadata for every column the spec's derived features emit."""
        entries: list[FeatureMetadata] = []
        for feature in feature_spec.derived_features:
            for column_metadata in feature.output_metadata:
                entries.append(FeatureMetadata.from_derived_column(column_metadata))
        return entries

    def _check_covers_model_inputs(self, feature_spec: FeatureSpec) -> None:
        """Fail loudly if the catalog and the spec disagree on model inputs.

        The catalog is assembled from the feature *selection*; the model
        consumes ``feature_spec.output_feature_names``. The two are
        independent records of the same fact and must agree exactly — a
        mismatch means the bundle's manifest and spec were written by
        divergent code, and an inspector built on a wrong catalog would
        silently omit or invent variables.
        """
        catalog_columns = set(self.columns)
        model_columns = set(feature_spec.output_feature_names)
        missing = sorted(model_columns - catalog_columns)
        extra = sorted(catalog_columns - model_columns)
        if missing or extra:
            raise ValueError(
                "FeatureCatalog does not match the bundle's model inputs. "
                f"Columns the model expects but the catalog lacks: {missing}. "
                f"Columns the catalog has but the model never consumes: "
                f"{extra}. The bundle's feature selection and feature spec "
                f"are inconsistent — rebuild the bundle."
            )
