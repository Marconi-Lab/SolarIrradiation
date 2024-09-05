from datetime import datetime

from geopy.geocoders import Nominatim

from susse.api_clients import ModisDataFetcher


def test_modis_data_fetcher():

    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    target_date = datetime(2020, 1, 10)

    reference_temperature = 25
    data_fetcher = ModisDataFetcher()
    # surf_temp = data_fetcher.surface_temperature(
    #     location.latitude, location.longitude, [str(target_date)]
    # )
    # assert surf_temp == reference_temperature
