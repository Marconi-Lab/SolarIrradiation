from susse import ModisProductEnum
from susse.api_clients import ModisProductFactory


def test_modis_products():

    factory = ModisProductFactory()
    surface_reflectance = factory.get_product_by_enum(ModisProductEnum.SURFACE_REFLACTANCE)
    land_surface_temperature = factory.get_product_by_enum(ModisProductEnum.LAND_SURFACE_TEMPERATURE)
    daymet = factory.get_product_by_enum(ModisProductEnum.DAYMET)

    num_bands_reflectance = 13
    num_bands_temperature = 11
    num_bands_daymet = 7

    assert num_bands_reflectance == len(surface_reflectance.get_band_names())
    assert num_bands_temperature == len(land_surface_temperature.get_band_names())
    assert num_bands_daymet == len(daymet.get_band_names())

    for product_enum in ModisProductEnum:
        factory.get_product_by_enum(product_enum)

