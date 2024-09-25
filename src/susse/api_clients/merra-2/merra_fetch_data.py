''' This part of the code is unfinished'''


import numpy as np
import logging
from .merra_download_manager import DownloadManager
from .merra_config import Merra2Config
from .merra_tools import Merra2tools
from .merra_product import MerraProductEnum
from datetime import datetime


class MerraFetchData:
    def __init__(self):
        self.temp_product = MerraProductEnum.TEMPERATURE
        self.aerosol_product = MerraProductEnum.AEROSOL_OPTICAL_DEPTH

    def fetch_data(self, dates, product, latitude, longitude) -> np.array:
        """
            Fetches data from the MERRA-2 dataset for the specified dates, product, and location.

        """
        try:
            lat_coord = Merra2tools.translate_lat_to_geos5_native(latitude)
            lon_coord = Merra2tools.translate_lon_to_geos5_native(longitude)
# Find the closest coordinate in the grid.
            lat_closest = Merra2tools.find_closest_coordinate(
                lat_coord, Merra2tools.MERRA_LATITUDE_ARRAY)
            lon_closest = Merra2tools.find_closest_coordinate(
                lon_coord, Merra2tools.MERRA_LONGITUDE_ARRAY)
# Generate URLs for scraping
            requested_lat = '[{lat}:1:{lat}]'.format(lat=lat_closest)
            requested_lon = '[{lon}:1:{lon}]'.format(lon=lon_closest)
            parameter = Merra2Config.generate_url_params(
                [product.value('field_id')], '[0:1:23]', requested_lat, requested_lon)
            generated_URL = Merra2Config.generate_download_links(
                years, database_url, database_id, parameter)
            download_manager = DownloadManager()
            download_manager.set_username_and_password(username, password)
            download_manager.download_path = field_name + '/' + loc
            download_manager.download_urls = generated_URL

        except Exception as e:
            logging.error(f"An error occurred while fetching data: {e}")
            return np.array([])

    def fetch_temperature_data(self, dates: datetime, latitude: float, longitude: float) -> np.array:
        """
        Fetches temperature data from the MERRA-2 dataset for the specified dates and location.
        """
        return self.fetch_data(dates, self.temp_product, latitude, longitude)

    def fetch_aerosol_optical_depth(self, dates, latitude, longitude) -> np.array:
        """
        Fetches aerosol optical depth data from the MERRA-2 dataset for the specified dates and location.
        """
        return self.fetch_data(dates, self.aerosol_product, latitude, longitude)
