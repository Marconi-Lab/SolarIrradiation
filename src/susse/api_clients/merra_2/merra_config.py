from datetime import datetime

import numpy as np

from .merra_product import MerraProductData, MerraProducts


class Merra2Config:
    """
    This class contains the URLs used and the functions for generating download URLs for MERRA-2 data. It also contains
    functions to convert longitude and latitude coordinates to merra2 coordinates, as well as other transformation.
    Some of these functions are based on the implementation from here:
    https://github.com/emilylaiken/merradownload
    """

    BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2"
    MERRA_LAT_COORDS = np.arange(0, 361, dtype=int)
    MERRA_LON_COORDS = np.arange(0, 576, dtype=int)

    @staticmethod
    def generate_database_url(product_data: MerraProductData) -> str:
        return f"{Merra2Config.BASE_URL}/{product_data.database_name}"

    @staticmethod
    def generate_download_link(
        date: datetime, product_data: MerraProductData, lat: float, lon: float
    ) -> str:
        file_name = Merra2Config.create_file_name(date, product_data)
        m_str = str(date.month).zfill(2)
        y_str = str(date.year)
        lat_geos5 = Merra2Config._translate_lat_to_geos5_native(lat)
        lon_geos5 = Merra2Config._translate_lon_to_geos5_native(lon)
        merra_lat = Merra2Config._find_closest_merra_coordinate(
            lat_geos5, Merra2Config.MERRA_LAT_COORDS
        )
        merra_lon = Merra2Config._find_closest_merra_coordinate(
            lon_geos5, Merra2Config.MERRA_LON_COORDS
        )
        suffix = f"{product_data.product_name}[0:1:23][{merra_lat}:1:{merra_lat}][{merra_lon}:1:{merra_lon}]"
        url = f"{Merra2Config.generate_database_url(product_data)}/{y_str}/{m_str}/{file_name}.nc4?{suffix}"
        return url

    @staticmethod
    def create_file_name(date: datetime, product_data: MerraProductData) -> str:
        d_str = str(date.day).zfill(2)
        m_str = str(date.month).zfill(2)
        y_str = str(date.year)
        file_num = Merra2Config._create_year_url(date.year)
        return f"MERRA2_{file_num}.{product_data.database_id}.{y_str}{m_str}{d_str}.nc4"

    @staticmethod
    def _create_year_url(year: int):
        """
        The file names consist of a number and a meta data string.
        The number changes over the years. 1980 until 1991 it is 100,
        1992 until 2000 it is 200, 2001 until 2010 it is  300
        and from 2011 until now it is 400.
        """
        file_number = ""

        if year >= 1980 and year < 1992:
            file_number = "100"
        elif year >= 1992 and year < 2001:
            file_number = "200"
        elif year >= 2001 and year < 2011:
            file_number = "300"
        elif year >= 2011:
            file_number = "400"
        else:
            raise Exception("The specified year is out of range.")
        return file_number

    @staticmethod
    def _translate_lat_to_geos5_native(latitude: float) -> float:
        """
        The source for this formula is in the MERRA2
        Variable Details - File specifications for GEOS pdf file.
        The Grid in the documentation has points from 1 to 361 and 1 to 576.
        The MERRA-2 Portal uses 0 to 360 and 0 to 575.
        latitude: float Needs +/- instead of N/S
        """
        return (latitude + 90) / 0.5

    @staticmethod
    def _translate_lon_to_geos5_native(longitude: float) -> float:
        """See function above"""
        return (longitude + 180) / 0.625

    @staticmethod
    def _find_closest_merra_coordinate(calc_coord, coord_array):
        """
        Since the resolution of the grid is 0.5 x 0.625, the 'real world'
        coordinates will not be matched 100% correctly. This function matches
        the coordinates as close as possible.
        """
        index = np.abs(coord_array - calc_coord).argmin()
        return coord_array[index]
