"""Tests for FeatureSelection — catalog validation + JSON roundtrip."""

from __future__ import annotations

import pytest

from susse.datasets import FeatureSelection
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.types import Source


def _first_nasa_id() -> str:
    return VariableCatalog.NASA_POWER_VARIABLES[0].variable_id


def _first_cams_id() -> str:
    return VariableCatalog.CAMS_VARIABLES[0].variable_id


def _first_merra_id() -> str:
    return VariableCatalog.MERRA_2_VARIABLES[0].variable_id


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
            v.variable_id for v in VariableCatalog.CAMS_VARIABLES
            if v.variable_id not in nasa_ids
        )
        with pytest.raises(ValueError, match="unknown variable_id"):
            FeatureSelection(nasa_variable_ids=(cams_only,))

    def test_duplicate_ids_rejected(self) -> None:
        nid = _first_nasa_id()
        with pytest.raises(ValueError, match="duplicate"):
            FeatureSelection(nasa_variable_ids=(nid, nid))


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
        assert FeatureSelection(
            include_satellite_irradiance=(),
        ).aux_columns == ()


class TestJsonRoundtrip:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = FeatureSelection(
            nasa_variable_ids=(_first_nasa_id(),),
            cams_variable_ids=(_first_cams_id(),),
            merra_variable_ids=(_first_merra_id(),),
            include_satellite_irradiance=(Source.NASA_POWER,),
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
