from datetime import datetime

from geopy.geocoders import Nominatim

from susse import ModisProductEnum
from susse.api_clients import ModisDataFetcher, ModisProductFactory


def test_modis_data_fetcher():

    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010, 1, 30)

    data_fetcher = ModisDataFetcher()

    factory = ModisProductFactory()
    product = factory.get_product_by_enum(ModisProductEnum.LAND_SURFACE_TEMPERATURE)
    result = data_fetcher.fetch_temp_day(
        location.latitude, location.longitude, start_date, end_date
    )
    average_day_temperature = result.get_time_average()

    available_dates = data_fetcher.get_available_dates_for_product_and_location(
        product, longitude=location.longitude, latitude=location.latitude
    )

    reflectance = data_fetcher.fetch_surface_reflectance(
        latitude=location.latitude,
        longitude=location.longitude,
        start_date=start_date,
        end_date=end_date,
    )
    reflectance_value = reflectance.get_time_average()

    expected_reflectance = 0.15466800000000003
    assert expected_reflectance == reflectance_value
