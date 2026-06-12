"""Tests for FeatureCatalog — the bundle's model-input metadata.

Per CLAUDE.md these test the public contract: that ``build`` composes
warehouse-variable and derived-feature metadata into one catalog
covering exactly the bundle's model inputs, groups each column by
origin, and fails loudly on unsupported or inconsistent bundles.
"""

from __future__ import annotations

import pytest

from susse.datasets import FeatureSelection
from susse.inference import FeatureCatalog, FeatureGroup
from susse.preprocessing import (
    ClearSkyIndexFeature,
    CyclicalDayOfYearFeature,
    FeatureSpec,
)
from susse.warehouse_ops.population.types import IrradianceBand, Source

# Pass-through columns the standard selection below produces, in build
# order: NASA aux, CAMS aux, then satellite irradiance.
_PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "nasa_temperature",
    "cams_ghi_clear",
    "sat_ghi_nasa_kwh_m2_day",
    "sat_ghi_cams_kwh_m2_day",
)
_DERIVED_COLUMN: str = "kt_cams"


@pytest.fixture
def standard_catalog() -> FeatureCatalog:
    """A catalog over one NASA aux + one CAMS aux + GHI + a derived kt."""
    selection = FeatureSelection(
        nasa_variable_ids=("temperature",),
        cams_variable_ids=("ghi_clear",),
        include_satellite_irradiance=(Source.NASA_POWER, Source.CAMS),
        include_satellite_bands=(IrradianceBand.GHI,),
    )
    spec = FeatureSpec(
        feature_columns=_PASSTHROUGH_COLUMNS,
        derived_features=(
            ClearSkyIndexFeature(
                ghi_column="sat_ghi_cams_kwh_m2_day",
                ghi_clear_column="cams_ghi_clear",
                output_column=_DERIVED_COLUMN,
            ),
        ),
    )
    return FeatureCatalog.build(selection=selection, feature_spec=spec)


class TestBuild:
    def test_catalog_covers_aux_irradiance_and_derived(
        self, standard_catalog: FeatureCatalog
    ) -> None:
        assert set(standard_catalog.columns) == set(_PASSTHROUGH_COLUMNS) | {
            _DERIVED_COLUMN
        }

    def test_catalog_orders_aux_then_irradiance_then_derived(
        self, standard_catalog: FeatureCatalog
    ) -> None:
        # build documents this order; columns promises "catalog order".
        assert standard_catalog.columns == (
            "nasa_temperature",
            "cams_ghi_clear",
            "sat_ghi_nasa_kwh_m2_day",
            "sat_ghi_cams_kwh_m2_day",
            _DERIVED_COLUMN,
        )

    def test_derived_only_catalog_handles_multi_column_feature(self) -> None:
        # A bundle with no warehouse inputs, only a derived feature that
        # itself emits two columns — the catalog must fan out one entry
        # per emitted column, not one per feature.
        selection = FeatureSelection(include_satellite_irradiance=())
        spec = FeatureSpec(
            feature_columns=(),
            derived_features=(CyclicalDayOfYearFeature(),),
        )
        catalog = FeatureCatalog.build(selection=selection, feature_spec=spec)
        assert catalog.columns == ("doy_sin", "doy_cos")
        assert all(entry.group is FeatureGroup.DERIVED for entry in catalog)

    def test_groups_track_each_column_origin(
        self, standard_catalog: FeatureCatalog
    ) -> None:
        assert standard_catalog.get("nasa_temperature").group is FeatureGroup.NASA_POWER
        assert standard_catalog.get("cams_ghi_clear").group is FeatureGroup.CAMS
        assert (
            standard_catalog.get("sat_ghi_nasa_kwh_m2_day").group
            is FeatureGroup.NASA_POWER
        )
        assert standard_catalog.get(_DERIVED_COLUMN).group is FeatureGroup.DERIVED

    def test_warehouse_metadata_carries_label_and_unit(
        self, standard_catalog: FeatureCatalog
    ) -> None:
        # The label is the warehouse catalog's display_name; the unit is
        # its physical unit — both must survive the FeatureMetadata hop.
        temperature = standard_catalog.get("nasa_temperature")
        assert temperature.label  # non-empty display name
        assert temperature.unit == "degC"

    def test_derived_metadata_flows_from_output_metadata(
        self, standard_catalog: FeatureCatalog
    ) -> None:
        kt = standard_catalog.get(_DERIVED_COLUMN)
        assert kt.unit == "unitless"
        assert kt.description  # non-empty

    def test_merra_selection_raises_not_implemented(self) -> None:
        selection = FeatureSelection(
            nasa_variable_ids=("temperature",),
            merra_variable_ids=("precipitable_water",),
            include_satellite_irradiance=(),
        )
        spec = FeatureSpec(feature_columns=("nasa_temperature",))
        with pytest.raises(NotImplementedError, match="MERRA"):
            FeatureCatalog.build(selection=selection, feature_spec=spec)

    def test_inconsistent_bundle_raises(self) -> None:
        # The spec claims a model input the selection never produces —
        # an internally inconsistent bundle must fail loudly, not serve
        # an inspector that silently omits the column.
        selection = FeatureSelection(
            nasa_variable_ids=("temperature",),
            include_satellite_irradiance=(),
        )
        spec = FeatureSpec(feature_columns=("nasa_temperature", "nasa_phantom"))
        with pytest.raises(ValueError, match="does not match"):
            FeatureCatalog.build(selection=selection, feature_spec=spec)


class TestGet:
    def test_unknown_column_raises_with_known_columns(
        self, standard_catalog: FeatureCatalog
    ) -> None:
        with pytest.raises(KeyError, match="not_a_feature"):
            standard_catalog.get("not_a_feature")


class TestFeatureGroup:
    def test_from_source_maps_supported_sources(self) -> None:
        assert FeatureGroup.from_source(Source.NASA_POWER) is FeatureGroup.NASA_POWER
        assert FeatureGroup.from_source(Source.CAMS) is FeatureGroup.CAMS

    def test_from_source_rejects_unsupported_source(self) -> None:
        with pytest.raises(ValueError, match="MERRA"):
            FeatureGroup.from_source(Source.MERRA_2)
