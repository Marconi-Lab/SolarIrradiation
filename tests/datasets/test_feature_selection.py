"""Tests for FeatureSelection — catalog validation + JSON roundtrip."""

from __future__ import annotations

import pytest

from susse.datasets import FeatureSelection
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.types import IrradianceBand, PhysicalStorage, Source


def _first_long_format(variables) -> str:
    """Pick a representative LONG_FORMAT variable_id from a catalog tuple.

    Helpers use this rather than ``[0]`` so they don't accidentally pick
    an irradiance_wide entry (which the long-table fields reject).
    """
    return next(
        v.variable_id
        for v in variables
        if v.physical_storage is PhysicalStorage.LONG_FORMAT
    )


def _first_nasa_id() -> str:
    return _first_long_format(VariableCatalog.NASA_POWER_VARIABLES)


def _first_cams_id() -> str:
    return _first_long_format(VariableCatalog.CAMS_VARIABLES)


def _first_merra_id() -> str:
    return _first_long_format(VariableCatalog.MERRA_2_VARIABLES)


class TestCatalogValidation:
    """Variable IDs must exist in the catalog for their declared source."""

    def test_known_nasa_variable_accepted(self) -> None:
        sel = FeatureSelection(nasa_variable_ids=(_first_nasa_id(),))
        assert sel.nasa_variable_ids == (_first_nasa_id(),)

    def test_unknown_nasa_variable_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown variable_id"):
            FeatureSelection(nasa_variable_ids=("definitely_not_a_variable",))

    def test_cams_only_id_under_nasa_field_is_rejected(self) -> None:
        # A CAMS-only variable_id under nasa_variable_ids should be flagged
        # as unknown for the NASA source. (NASA + CAMS share `ghi/dhi/dni`
        # in the catalog; we deliberately pick an id that is CAMS-only.)
        nasa_ids = {v.variable_id for v in VariableCatalog.NASA_POWER_VARIABLES}
        cams_only = next(
            v.variable_id
            for v in VariableCatalog.CAMS_VARIABLES
            if v.variable_id not in nasa_ids
        )
        with pytest.raises(ValueError, match="unknown variable_id"):
            FeatureSelection(nasa_variable_ids=(cams_only,))

    def test_duplicate_ids_rejected(self) -> None:
        nid = _first_nasa_id()
        with pytest.raises(ValueError, match="duplicate"):
            FeatureSelection(nasa_variable_ids=(nid, nid))

    def test_irradiance_wide_variable_rejected_in_long_field(self) -> None:
        # `ghi`/`dhi`/`dni` exist in the NASA + CAMS catalog but live in
        # `irradiance_daily` (physical_storage=IRRADIANCE_WIDE), not in
        # the long-format aux table that `nasa_variable_ids` pivots
        # from. Requesting them through `nasa_variable_ids` would produce
        # 100%-NaN columns at fetch time — surface that as a clear
        # construction-time error pointing to the right field.
        with pytest.raises(ValueError, match="include_satellite_irradiance"):
            FeatureSelection(nasa_variable_ids=("ghi",))
        with pytest.raises(ValueError, match="include_satellite_irradiance"):
            FeatureSelection(cams_variable_ids=("dhi",))

    def test_storage_mismatch_error_names_actual_and_expected_storage(self) -> None:
        # Remediation message must be specific enough that the user can
        # both diagnose the problem (their variable's actual storage) and
        # fix it (the field that does match).
        with pytest.raises(ValueError) as exc:
            FeatureSelection(nasa_variable_ids=("ghi",))
        msg = str(exc.value)
        assert "physical_storage=long_format" in msg
        assert "irradiance_wide" in msg
        assert "include_satellite_irradiance" in msg


class TestIrradianceSourceRestriction:
    """Only NASA POWER and CAMS write to irradiance_daily."""

    def test_merra2_in_irradiance_sources_rejected(self) -> None:
        with pytest.raises(ValueError, match="irradiance_daily"):
            FeatureSelection(
                include_satellite_irradiance=(Source.NASA_POWER, Source.MERRA_2),
            )

    def test_modis_in_irradiance_sources_rejected(self) -> None:
        with pytest.raises(ValueError, match="irradiance_daily"):
            FeatureSelection(include_satellite_irradiance=(Source.MODIS,))

    def test_default_is_nasa_plus_cams(self) -> None:
        sel = FeatureSelection()
        assert sel.include_satellite_irradiance == (Source.NASA_POWER, Source.CAMS)


class TestIrradianceBands:
    """include_satellite_bands picks which irradiance series ship from the wide table."""

    def test_default_is_ghi_only(self) -> None:
        sel = FeatureSelection()
        assert sel.include_satellite_bands == (IrradianceBand.GHI,)

    def test_can_request_dhi_and_dni(self) -> None:
        sel = FeatureSelection(
            include_satellite_bands=(
                IrradianceBand.GHI,
                IrradianceBand.DHI,
                IrradianceBand.DNI,
            ),
        )
        assert IrradianceBand.DHI in sel.include_satellite_bands
        assert IrradianceBand.DNI in sel.include_satellite_bands

    def test_empty_bands_with_sources_rejected(self) -> None:
        with pytest.raises(ValueError, match="include_satellite_bands is empty"):
            FeatureSelection(include_satellite_bands=())

    def test_empty_bands_allowed_when_no_sources(self) -> None:
        sel = FeatureSelection(
            include_satellite_irradiance=(),
            include_satellite_bands=(),
        )
        assert sel.is_empty is True

    def test_duplicate_bands_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            FeatureSelection(
                include_satellite_bands=(IrradianceBand.GHI, IrradianceBand.GHI),
            )


class TestQcLevels:
    def test_empty_qc_levels_rejected(self) -> None:
        # An empty tuple would silently drop every ground row; pin this
        # so future regressions raise instead.
        with pytest.raises(ValueError, match="qc_levels"):
            FeatureSelection(qc_levels=())


class TestAuxColumns:
    """Source-prefixed column names avoid collisions across sources."""

    def test_aux_columns_are_source_prefixed(self) -> None:
        sel = FeatureSelection(
            nasa_variable_ids=(_first_nasa_id(),),
            cams_variable_ids=(_first_cams_id(),),
            merra_variable_ids=(_first_merra_id(),),
        )
        cols = sel.aux_columns
        assert any(c.startswith("nasa_") for c in cols)
        assert any(c.startswith("cams_") for c in cols)
        assert any(c.startswith("merra_") for c in cols)

    def test_empty_selection_aux_columns_empty(self) -> None:
        assert (
            FeatureSelection(
                include_satellite_irradiance=(),
            ).aux_columns
            == ()
        )


class TestJsonRoundtrip:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = FeatureSelection(
            nasa_variable_ids=(_first_nasa_id(),),
            cams_variable_ids=(_first_cams_id(),),
            merra_variable_ids=(_first_merra_id(),),
            include_satellite_irradiance=(Source.NASA_POWER,),
            include_satellite_bands=(IrradianceBand.GHI, IrradianceBand.DNI),
            qc_levels=("pass", "fail_range"),
        )
        recovered = FeatureSelection.from_dict(original.to_dict())
        assert recovered == original

    def test_minimal_selection_roundtrip(self) -> None:
        # Defaults-only — make sure the absent / empty cases don't drop
        # any fields.
        original = FeatureSelection()
        recovered = FeatureSelection.from_dict(original.to_dict())
        assert recovered == original

    def test_legacy_dict_without_bands_defaults_to_ghi(self) -> None:
        # Manifests written before the GHI/DHI/DNI split won't carry the
        # ``include_satellite_bands`` key. They must deserialise as if the
        # field had its default value, so older snapshots stay loadable.
        legacy = {
            "include_satellite_irradiance": [Source.NASA_POWER.value],
            "qc_levels": ["pass"],
        }
        sel = FeatureSelection.from_dict(legacy)
        assert sel.include_satellite_bands == (IrradianceBand.GHI,)
