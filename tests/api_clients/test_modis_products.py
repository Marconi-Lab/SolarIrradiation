import pytest

from susse import ModisProductEnum
from susse.api_clients import ModisProductFactory


@pytest.mark.skip(
    reason="Modis service seems to be temporarily unavailable. This needs to be re-evaluated"
)
def test_modis_products():
    factory = ModisProductFactory()
    surface_reflectance = factory.get_product_by_enum(
        ModisProductEnum.SURFACE_REFLACTANCE
    )
    emissivity = factory.get_product_by_enum(ModisProductEnum.EMISSIVITY)
    leaf_area_index = factory.get_product_by_enum(ModisProductEnum.LEAF_AREA_INDEX)

    num_bands_reflectance = 13
    num_bands_emissivity = 11
    num_bands_leaf = 6

    assert num_bands_reflectance == len(surface_reflectance.get_band_names())
    assert num_bands_emissivity == len(emissivity.get_band_names())
    assert num_bands_leaf == len(leaf_area_index.get_band_names())

    for product_enum in ModisProductEnum:
        factory.get_product_by_enum(product_enum)
