import os
import shutil
from datetime import datetime
from typing import Tuple

import numpy as np
from geopy.geocoders import Nominatim

from susse.api_clients import (
    Merra2Config,
    MerraDataFetcher,
    MerraDataStreamFetcher,
    MerraDownloadManager,
    MerraProducts,
)


def get_test_credentials() -> Tuple[str, str]:
    username = os.getenv("NASA_USERNAME")
    password = os.getenv("NASA_PASSWORD")
    if not username or not password:
        raise ValueError("NASA credentials not set in environment variables.")
    return username, password


def test_merra_config():
    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    start_date = datetime(2010, 1, 1)
    download_url = Merra2Config.generate_download_link(
        start_date,
        MerraProducts.SPECIFIC_HUMIDITY.value,
        location.latitude,
        location.longitude,
    )
    expected_url = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/2010/01/MERRA2_300.tavg1_2d_slv_Nx.20100101.nc4.nc4?QV2M[0:1:23][181:1:181][340:1:340]"
    assert expected_url == download_url


def test_merra_downloader():

    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    start_date = datetime(2010, 1, 1)
    download_url = Merra2Config.generate_download_link(
        start_date,
        MerraProducts.SPECIFIC_HUMIDITY.value,
        location.latitude,
        location.longitude,
    )

    download_manager = MerraDownloadManager()
    try:
        username, password = get_test_credentials()
        download_manager.set_username_pw(username, password)
    except ValueError as e:
        pass
    download_folder = "./tmp"
    download_manager.download_from_urls(download_url, download_folder)
    filename = download_manager.extract_filename_from_url(download_url)
    file_path = os.path.join(download_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)


def test_download_fetcher():
    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Zurich")
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010, 1, 3)

    merra_product = MerraProducts.SURFACE_PRESSURE.value
    download_folder = "./tmp"
    data_fetcher = MerraDataFetcher(base_download_folder=download_folder)
    try:
        username, password = get_test_credentials()
        data_fetcher.set_username_pw(username, password)
    except ValueError as e:
        pass
    data = data_fetcher.fetch_product_result(
        merra_product, location, start_date, end_date
    )
    shutil.rmtree(download_folder)

    var_names = data.get_variable_names()
    expected_var_names = [merra_product.product_name]
    assert var_names == expected_var_names

    var_np = data.to_np()
    assert len(var_np) == 72


def test_merra_stream():
    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")

    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 1, 1)

    merra_stream = MerraDataStreamFetcher()

    try:
        username, password = get_test_credentials()
        merra_stream._authenticate._set_username_pw(username, password)
    except ValueError:
        pass

    result = merra_stream.fetch_data(
        start_date=start_date,
        end_date=end_date,
        product_data=MerraProducts.AIR_TEMPERATURE.value,
        location=location,
    )

    cleaned_data = result.data

    flattened_data = [value for day_data in cleaned_data.values() for value in day_data]
    mean_temperature = np.mean(flattened_data)

    assert (
        mean_temperature == 295.7949358622233
    ), f"Expected 295.7949358622233 but got {mean_temperature}"
