"""Typed feature-selection config for training-dataset assembly.

A :class:`FeatureSelection` is the *recipe* half of a dataset version
(per the dataset-versioning architecture: a dataset = recipe applied to
the warehouse at a point in time). It enumerates which variables from
which sources to materialize, plus the QC filter on ground truth and
which satellite irradiance sources to include.

Recipes live in code (and therefore in git); they are JSON-serialised
into the dataset's manifest so a snapshot can be re-derived from
scratch even if the original Python object is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..warehouse_ops.population.dim_variable import VariableCatalog
from ..warehouse_ops.population.types import IrradianceBand, PhysicalStorage, Source


@dataclass(frozen=True)
class FeatureSelection:
    """Which variables to include in a training dataset.

    Variable IDs are validated against the catalog at construction time —
    a typo or a stale catalog entry surfaces as a ``ValueError`` here,
    not as a silent NaN column at training time.

    Attributes:
        nasa_variable_ids: NASA POWER auxiliary variables, by warehouse
            ``variable_id`` (e.g. ``"T2M"``, ``"CLRSKY_DNI"``). Pivoted
            from ``nasa_daily_vars_long``.
        cams_variable_ids: CAMS auxiliary variables (e.g. ``"sza"``,
            ``"reliability"``). Pivoted from ``cams_daily_vars_long``.
        merra_variable_ids: MERRA-2 reanalysis variables (e.g.
            ``"aod_550_extinction"``, ``"precipitable_water"``). Pivoted
            from ``merra_daily_vars_long``.
        modis_variable_ids: MODIS observation slots, encoded as
            ``"{product}_{band}"`` (e.g. ``"MOD13Q1_250m_16_days_NDVI"``).
            Pulled from ``modis_observations`` with as-of forward-fill
            semantics (carry the most recent composite value forward to
            each daily row).
        include_satellite_irradiance: Which satellite sources contribute
            irradiance features. Each (source, band) combination becomes
            a column ``sat_<band>_<source>_kwh_m2_day``.
        include_satellite_bands: Which irradiance bands to pull from
            ``irradiance_daily`` for the requested sources. Defaults to
            GHI only — the common case for bias-correcting global
            horizontal irradiance. Add :attr:`IrradianceBand.DHI` /
            :attr:`IrradianceBand.DNI` to expose diffuse / direct-normal
            as additional features.
        qc_levels: QC levels to keep on ground measurements. Default
            ``("pass",)`` excludes failed-range and other rejected rows.
    """

    nasa_variable_ids: tuple[str, ...] = ()
    cams_variable_ids: tuple[str, ...] = ()
    merra_variable_ids: tuple[str, ...] = ()
    modis_variable_ids: tuple[str, ...] = ()
    include_satellite_irradiance: tuple[Source, ...] = (Source.NASA_POWER, Source.CAMS)
    include_satellite_bands: tuple[IrradianceBand, ...] = (IrradianceBand.GHI,)
    qc_levels: tuple[str, ...] = ("pass",)

    def __post_init__(self) -> None:
        # Each <source>_variable_ids field is wired to a specific physical
        # storage backend in the warehouse: long-format pivots for the
        # aux tables, the modis_observations table for MODIS. Validation
        # enforces that mapping so a request for an irradiance_wide
        # variable through a long-table field is rejected at construction
        # rather than silently producing a 100%-NaN column at fetch time.
        self._validate_against_catalog(
            self.nasa_variable_ids, Source.NASA_POWER,
            field="nasa_variable_ids",
            expected_storage=PhysicalStorage.LONG_FORMAT,
        )
        self._validate_against_catalog(
            self.cams_variable_ids, Source.CAMS,
            field="cams_variable_ids",
            expected_storage=PhysicalStorage.LONG_FORMAT,
        )
        self._validate_against_catalog(
            self.merra_variable_ids, Source.MERRA_2,
            field="merra_variable_ids",
            expected_storage=PhysicalStorage.LONG_FORMAT,
        )
        self._validate_against_catalog(
            self.modis_variable_ids, Source.MODIS,
            field="modis_variable_ids",
            expected_storage=PhysicalStorage.MODIS_OBSERVATIONS,
        )
        for src in self.include_satellite_irradiance:
            if src not in (Source.NASA_POWER, Source.CAMS):
                raise ValueError(
                    f"include_satellite_irradiance contains {src.value}, but "
                    f"only NASA POWER and CAMS write GHI to irradiance_daily. "
                    f"MERRA-2 GHI and MODIS imagery are auxiliary features, "
                    f"not satellite-irradiance series — request them via the "
                    f"corresponding *_variable_ids field."
                )
        if self.include_satellite_irradiance and not self.include_satellite_bands:
            raise ValueError(
                "include_satellite_bands is empty but "
                "include_satellite_irradiance requested sources "
                f"{[s.value for s in self.include_satellite_irradiance]}. "
                "Pass at least one IrradianceBand (default: "
                "(IrradianceBand.GHI,)) or set include_satellite_irradiance=()."
            )
        if len(set(self.include_satellite_bands)) != len(self.include_satellite_bands):
            raise ValueError(
                "include_satellite_bands contains duplicates: "
                f"{[b.value for b in self.include_satellite_bands]}. "
                "Each band should appear at most once."
            )
        if not self.qc_levels:
            raise ValueError(
                "qc_levels must be non-empty. Pass ('pass',) to keep only "
                "QC-passed ground rows (default), or include other levels "
                "like 'fail_range' explicitly. An empty tuple would "
                "silently drop every ground row."
            )

    @staticmethod
    def _validate_against_catalog(
        ids: tuple[str, ...],
        source: Source,
        *,
        field: str,
        expected_storage: PhysicalStorage,
    ) -> None:
        if not ids:
            return
        catalog = {
            v.variable_id: v for v in VariableCatalog.for_source(source)
        }
        if not catalog:
            raise ValueError(
                f"FeatureSelection.{field} = {list(ids)} but the catalog has "
                f"no variables registered for source {source.value}. Either "
                f"the source isn't ingested yet, or the catalog has not been "
                f"populated."
            )
        unknown = [vid for vid in ids if vid not in catalog]
        if unknown:
            valid = sorted(
                vid for vid, v in catalog.items()
                if v.physical_storage is expected_storage
            )
            raise ValueError(
                f"FeatureSelection.{field} contains unknown variable_id(s) "
                f"for source {source.value}: {unknown}. "
                f"Known {source.value} variables with "
                f"physical_storage={expected_storage.value}: {valid}."
            )
        # Detect duplicates within this source's list.
        if len(set(ids)) != len(ids):
            seen: set[str] = set()
            dups = [vid for vid in ids if vid in seen or seen.add(vid)]  # type: ignore[func-returns-value]
            raise ValueError(
                f"FeatureSelection.{field} contains duplicate variable_id(s): "
                f"{sorted(set(dups))}. Each variable should be listed at most once."
            )
        # Reject catalog entries that exist but live in a different
        # physical storage than this field's pivot reads from.
        wrong_storage = [
            vid for vid in ids
            if catalog[vid].physical_storage is not expected_storage
        ]
        if wrong_storage:
            actual = {vid: catalog[vid].physical_storage.value for vid in wrong_storage}
            remediation = (
                "Use `include_satellite_irradiance` for irradiance series."
                if expected_storage is PhysicalStorage.LONG_FORMAT
                and any(
                    catalog[vid].physical_storage is PhysicalStorage.IRRADIANCE_WIDE
                    for vid in wrong_storage
                )
                else f"Move these IDs to the field whose physical_storage matches."
            )
            raise ValueError(
                f"FeatureSelection.{field} expects "
                f"physical_storage={expected_storage.value} but received "
                f"variable_id(s) with mismatching storage: {actual}. "
                f"{remediation}"
            )

    @property
    def is_empty(self) -> bool:
        """True iff no aux features and no irradiance sources are requested."""
        return (
            not self.nasa_variable_ids
            and not self.cams_variable_ids
            and not self.merra_variable_ids
            and not self.modis_variable_ids
            and not self.include_satellite_irradiance
        )

    @property
    def aux_columns(self) -> tuple[str, ...]:
        """All non-irradiance feature columns the selection will produce.

        Source-prefixed to avoid collisions: ``T2M`` from NASA POWER and
        a future ``T2M`` from MERRA-2 would otherwise clobber each other.
        """
        cols: list[str] = []
        cols.extend(f"nasa_{v}" for v in self.nasa_variable_ids)
        cols.extend(f"cams_{v}" for v in self.cams_variable_ids)
        cols.extend(f"merra_{v}" for v in self.merra_variable_ids)
        cols.extend(f"modis_{v}" for v in self.modis_variable_ids)
        return tuple(cols)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for inclusion in a manifest."""
        return {
            "nasa_variable_ids": list(self.nasa_variable_ids),
            "cams_variable_ids": list(self.cams_variable_ids),
            "merra_variable_ids": list(self.merra_variable_ids),
            "modis_variable_ids": list(self.modis_variable_ids),
            "include_satellite_irradiance": [
                s.value for s in self.include_satellite_irradiance
            ],
            "include_satellite_bands": [b.value for b in self.include_satellite_bands],
            "qc_levels": list(self.qc_levels),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureSelection":
        """Reconstruct from a :meth:`to_dict` payload.

        Older manifests written before the GHI/DHI/DNI split lack the
        ``include_satellite_bands`` key. Default to ``(GHI,)`` so they
        deserialise to a selection equivalent to the original behaviour.
        """
        return cls(
            nasa_variable_ids=tuple(d.get("nasa_variable_ids", ())),
            cams_variable_ids=tuple(d.get("cams_variable_ids", ())),
            merra_variable_ids=tuple(d.get("merra_variable_ids", ())),
            modis_variable_ids=tuple(d.get("modis_variable_ids", ())),
            include_satellite_irradiance=tuple(
                Source(v) for v in d.get(
                    "include_satellite_irradiance",
                    [Source.NASA_POWER.value, Source.CAMS.value],
                )
            ),
            include_satellite_bands=tuple(
                IrradianceBand(v) for v in d.get(
                    "include_satellite_bands", [IrradianceBand.GHI.value]
                )
            ),
            qc_levels=tuple(d.get("qc_levels", ("pass",))),
        )
