import numpy as np


class Merra2Tools:
    """
    This class contains the utility functions used for MERRA-2 data processing.
    """

    # Native MERRA-2 grid arrays
    MERRA_LATITUDE_ARRAY = np.arange(0, 361, dtype=int)
    MERRA_LONGITUDE_ARRAY = np.arange(0, 576, dtype=int)

    @staticmethod
    def translate_lat_to_geos5_native(latitude):
        """
        Converts a latitude value into its corresponding MERRA-2 native grid point.

        The MERRA-2 grid spans from -90° to +90° (divided into 361 grid points with 0.5-degree resolution).
        This function calculates the corresponding grid index.

        :param latitude: Latitude value in degrees (-90 to +90).
        :return: Index in the MERRA-2 grid corresponding to the latitude.
        """
        return ((latitude + 90) / 0.5)

    @staticmethod
    def translate_lon_to_geos5_native(longitude):
        """
        Converts a longitude value into its corresponding MERRA-2 native grid point.

        The MERRA-2 grid spans from -180° to +180° (divided into 576 grid points with 0.625-degree resolution).
        This function calculates the corresponding grid index.

        :param longitude: Longitude value in degrees (-180 to +180).
        :return: Index in the MERRA-2 grid corresponding to the longitude.
        """
        return ((longitude + 180) / 0.625)

    @staticmethod
    def find_closest_coordinate(calc_coord, coord_array):
        """
        Finds the closest coordinate in the MERRA-2 grid for a given real-world coordinate.

        Since the MERRA-2 grid has finite resolution, real-world coordinates will not match perfectly.
        This function finds the closest matching grid point.

        :param calc_coord: The calculated coordinate (latitude/longitude).
        :param coord_array: The array of grid points to compare against (latitude/longitude grid).
        :return: The closest grid point to the provided coordinate.
        """
        index = np.abs(coord_array - calc_coord).argmin()
        return coord_array[index]

    @staticmethod
    def translate_year_to_file_number(year):
        """
        Translates a year into the corresponding MERRA-2 file number series.

        MERRA-2 files are organized by decades, with different file number series:
        - 1980-1991 -> 100 series
        - 1992-2000 -> 200 series
        - 2001-2010 -> 300 series
        - 2011 onwards -> 400 series

        :param year: The year for which the file number is needed.
        :return: The MERRA-2 file number series for the given year.
        :raises ValueError: If the year is outside the supported range.
        """
        if 1980 <= year < 1992:
            return '100'
        elif 1992 <= year < 2001:
            return '200'
        elif 2001 <= year < 2011:
            return '300'
        elif year >= 2011:
            return '400'
        else:
            raise ValueError('The specified year is out of range.')
