from susse.api_clients import Merra2Config, MerraProducts, MerraDownloadManager, MerraDataFetcher
from geopy.geocoders import Nominatim
from datetime import datetime
from typing import Tuple

import os

def get_test_credentials() -> Tuple[str, str]:
    username = os.getenv('NASA_USERNAME')
    password = os.getenv('NASA_PASSWORD')
    if not username or not password:
        raise ValueError("NASA credentials not set in environment variables.")
    return username, password

def test_merra_config():

    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    start_date = datetime(2010, 1, 1)
    download_url = Merra2Config.generate_download_link(start_date, MerraProducts.HUSS.value, location.latitude,
                                                       location.longitude)
    expected_url = ""
    assert expected_url == download_url

def test_merra_downloader():

    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    start_date = datetime(2010, 1, 1)
    download_url = Merra2Config.generate_download_link(start_date, MerraProducts.HUSS.value, location.latitude,
                                                       location.longitude)

    download_manager = MerraDownloadManager()
    username, password = get_test_credentials()
    download_manager.set_username_pw(username, password)
    download_manager.download_from_urls(download_url, ".")


def test_download_fetcher():
    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Zurich")
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010, 1, 3)
    data_fetcher = MerraDataFetcher()
    data = data_fetcher.fetch_product_result(MerraProducts.PR.value, location, start_date, end_date)