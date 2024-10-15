from datetime import datetime

import numpy as np

from .merra_product import MerraProductData, MerraProducts


class Merra2Config:
    """
    This class contains the URLs used and the functions for generating download URLs for MERRA-2 data.
    """

    BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2"
    MERRA_LAT_COORDS = np.arange(0, 361, dtype=int)
    MERRA_LON_COORDS = np.arange(0, 576, dtype=int)

    @staticmethod
    def generate_database_url(product_data: MerraProductData):
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

    # @staticmethod
    # def generate_download_link(date, database_name: str, database_id: str, url_parameters):
    #     #       print(date)
    #     parsed_date = datetime.strptime(date, '%d-%m-%Y')
    #     y_str = parsed_date.strftime('%Y')  # Full year as string
    #     m_str = parsed_date.strftime('%m')  # Month as 2-digit string
    #
    #     # Get the file number based on the year
    #     if database_id.startswith("const"):
    #         merra_stream = "101"
    #     else:
    #         merra_stream = Merra2Tools.translate_year_to_file_number(
    #             int(y_str))
    #     # date string
    #     # Format the date as 'yyyymmdd'
    #     date_string = parsed_date.strftime('%Y%m%d')
    #
    #     # Create the file name string matching the structure MERRA2_400.inst3_2d_gas_Nx.20200207.nc4
    #     file_name = 'MERRA2_{num}.{name}.{date}.nc4'.format(
    #         num=merra_stream, name=database_id, date=date_string)
    #
    #     # Create the query URL
    #     query_url = '{base}/{database_name}/{y}/{m}/{file_name}?{params}'.format(
    #         base=Merra2Config.BASE_URL, database_name=database_name, y=y_str, m=m_str, file_name=file_name, params=url_parameters)
    #     return query_url


# Parse the date string (dd-mm-yy) into a datetime object
#     @staticmethod
#     def generate_download_links(dates, database_name, database_id, url_parameters):
#         """
#         Generates download links for MERRA-2 data.
#
#         :param dates: List of dates in 'dd-mm-yy' format.
#         :param base_url: The base URL for the MERRA-2 database.
#         :param dataset_name: Name of the dataset (e.g., 'inst3_2d_gas_Nx').
#         :param url_params: URL parameters for querying specific variables.
#         :return: List of URLs for downloading the data.
#         """
#         return [Merra2Config.generate_download_link(date, database_name, database_id, url_parameters) for date in dates]
#
#     @staticmethod
#     def generate_url_params(parameter, time_slice, latitude, longitude):
#         """
#         Creates a query string containing all the parameters in query form.
#
#         :param parameter: List of MERRA-2 variables to request (e.g., 'T2M', 'PS').
#         :param time_slice: Time slice string for querying time range (e.g., '[0:1:23]').
#         :param latitude: Latitude slice string (e.g., '[0:360]').
#         :param longitude: Longitude slice string (e.g., '[0:575]').
#         :return: Comma-separated URL query string for the parameters.
#         """
#         params = []
#         for param in parameter:
#             # For each parameter, add the time, latitude, and longitude slices
#             param_with_slices = Merra2Config.generate_url_parameter(
#                 prouduct_name=param, time_slice=time_slice, latitude=latitude, longitude=longitude)
#             params.append(param_with_slices)
#
# # Return all parameters joined by a comma
#         return ','.join(params)
#
#     @staticmethod
#     def generate_url_parameter(prouduct_name, time_slice, latitude, longitude):
#         parameter = str(prouduct_name)
#         return f'{parameter}{time_slice}{latitude}{longitude}'
#
#     @staticmethod
#     def generate_plain_link(date, database_name, database_id):
#         parsed_date = datetime.strptime(date, '%d-%m-%Y')
#         y_str = parsed_date.strftime('%Y')  # Full year as string
#         m_str = parsed_date.strftime('%m')  # Month as 2-digit string
#
#         # Get the file number based on the year
#         if database_id.startswith("const"):
#             merra_stream = "101"
#         else:
#             merra_stream = Merra2Tools.translate_year_to_file_number(
#                 int(y_str))
#         # date string
#         # Format the date as 'yyyymmdd'
#         date_string = parsed_date.strftime('%Y%m%d')
#
#         # Create the file name string matching the structure MERRA2_400.inst3_2d_gas_Nx.20200207.nc4
#         file_name = 'MERRA2_{num}.{name}.{date}.nc4'.format(
#             num=merra_stream, name=database_id, date=date_string)
#
#         # Create the query URL
#         query_url = '{base}/{database_name}/{y}/{m}/{file_name}?{params}'.format(
#             base=Merra2Config.BASE_URL, database_name=database_name, y=y_str, m=m_str, file_name=file_name, params=url_parameters)
#         return query_url
