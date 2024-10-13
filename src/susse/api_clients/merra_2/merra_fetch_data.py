''' This part of the code is unfinished'''


import numpy as np
import logging
from .merra_download_manager import DownloadManager
from .merra_config import Merra2Config
from .merra_tools import Merra2Tools
from .merra_product import Dictionarys, Keys, Products
from datetime import datetime


class MerraFetchData():

    TIME_SLICE = '[0:1:23]'

    @staticmethod
    def fetch_data(dates, product_name, database_name, lat_1, lon_1, database_id):
        """
        Fetches data from the MERRA-2 dataset for the specified dates, product, and location.

        """
        lat_coord_1 = Merra2Tools.translate_lat_to_geos5_native(lat_1)
        lon_coord_1 = Merra2Tools.translate_lon_to_geos5_native(lon_1)
        # lat_coord_2 = Merra2Tools.translate_lon_to_geos5_native(lat_2)
        # lon_coord_2 = Merra2Tools.translate_lon_to_geos5_native(lon_2)
#       Find the closest coordinate in the grid.
        lat_closest_1 = Merra2Tools.find_closest_coordinate(
            lat_coord_1, Merra2Tools.MERRA_LATITUDE_ARRAY)
        lon_closest_1 = Merra2Tools.find_closest_coordinate(
            lon_coord_1, Merra2Tools.MERRA_LONGITUDE_ARRAY)
        # lat_closest_2 = Merra2Tools.find_closest_coordinate(
        #
        #     lat_coord_2, Merra2Tools.MERRA_LATITUDE_ARRAY)
        # lon_closest_2 = Merra2Tools.find_closest_coordinate(
        #     lon_coord_2, Merra2Tools.MERRA_LONGITUDE_ARRAY)
        #

#       Generate URLs for scraping
        requested_lat = '[{lat_1}:1:{lat_2}]'.format(
            lat_1=lat_closest_1, lat_2=lat_closest_1)
        requested_lon = '[{lon_1}:1:{lon_2}]'.format(
            lon_1=lon_closest_1, lon_2=lon_closest_1)
#       Generate url parameters

        parameters = Merra2Config.generate_url_parameter(
            product_name, MerraFetchData.TIME_SLICE, requested_lat, requested_lon)
#       print(parameters)
#       Generate link

        generated_URL = Merra2Config.generate_download_links(
            dates=dates, database_name=database_name, database_id=database_id, url_parameters=parameters)
        print(generated_URL)

#       Downlaod Manager
        download_manager = DownloadManager(link=generated_URL)
        download_manager.read_credentials_from_yaml()
        # download_manager.start_download(
        #     ave_file=True, product_name=product_name)
        #
        download_manager.xarrray_try()

    @staticmethod
    def fetch_surface_pressure(dates: datetime, lat_1: float, lon_1: float):
        """
        Fetches temperature data from the MERRA-2 dataset for the specified dates and location.
        """
        return MerraFetchData.fetch_data(dates=dates, product_name=Products.SURFACE_PRESSURE.value, database_name=Dictionarys.SLV[Keys.DATABASE_NAME.value], database_id=Dictionarys.SLV[Keys.DATABASE_ID.value], lat_1=lat_1, lon_1=lon_1)

    def fetch_aerosol_optical_depth(self, dates, latitude, longitude) -> np.array:
        """
        Fetches aerosol optical depth data from the MERRA-2 dataset for the specified dates and location.
        """
        return self.fetch_data(dates, Products.AEROSOL_OPTICAL_DEPTH.value, latitude, longitude, Dictionarys.RAD[keys.DATABASE_ID.value])
