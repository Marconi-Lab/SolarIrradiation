"""GES DISC MERRA-2 endpoint configuration + lat/lon → grid-index helpers.

The OPeNDAP URL itself is built by :class:`MerraDailyFetcher`; this module
only owns the parts that are independent of the request shape: the base
URL, the per-collection database URL prefix, the YYYYMMDD file-name
convention, and the conversion from real lat/lon to MERRA-2's native grid
indices.
"""

from datetime import date

import numpy as np

from .merra_product import MerraProductData


class Merra2Config:
    """Static configuration for MERRA-2 OPeNDAP access.

    Some of the lat/lon-to-grid translation logic is adapted from
    https://github.com/emilylaiken/merradownload.
    """

    BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2"
    MERRA_LAT_COORDS = np.arange(0, 361, dtype=int)
    MERRA_LON_COORDS = np.arange(0, 576, dtype=int)

    @staticmethod
    def generate_database_url(product_data: MerraProductData) -> str:
        return f"{Merra2Config.BASE_URL}/{product_data.database_name}"

    @staticmethod
    def create_file_name(d: date, product_data: MerraProductData) -> str:
        # ``date`` accepts both ``datetime.date`` and ``datetime.datetime``
        # (datetime subclasses date). The function only reads .day/.month/.year.
        d_str = str(d.day).zfill(2)
        m_str = str(d.month).zfill(2)
        y_str = str(d.year)
        file_num = Merra2Config._create_year_url(d.year)
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
