from datetime import datetime
from merra_tools import Merra2Tools


class Merra2Config:
    """
    This class contains the URLs used and the functions for generating download URLs for MERRA-2 data.
    """

    base_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2'

    @staticmethod
    def generate_download_link(date, database_name, database_id, url_parameters):
        #       print(date)
        parsed_date = datetime.strptime(date, '%d-%m-%Y')
        y_str = parsed_date.strftime('%Y')  # Full year as string
        m_str = parsed_date.strftime('%m')  # Month as 2-digit string

        # Get the file number based on the year
        if database_id.startswith("const"):
            merra_stream = "101"
        else:
            merra_stream = Merra2Tools.translate_year_to_file_number(
                int(y_str))
        # date string
        # Format the date as 'yyyymmdd'
        date_string = parsed_date.strftime('%Y%m%d')

        # Create the file name string matching the structure MERRA2_400.inst3_2d_gas_Nx.20200207.nc4
        file_name = 'MERRA2_{num}.{name}.{date}.nc4'.format(
            num=merra_stream, name=database_id, date=date_string)

        # Create the query URL
        query_url = '{base}/{database_name}/{y}/{m}/{file_name}?{params}'.format(
            base=Merra2Config.base_url, database_name=database_name, y=y_str, m=m_str, file_name=file_name, params=url_parameters)
        return query_url

# Parse the date string (dd-mm-yy) into a datetime object
    @staticmethod
    def generate_download_links(dates, database_name, database_id, url_parameters):
        """
        Generates download links for MERRA-2 data.

        :param dates: List of dates in 'dd-mm-yy' format.
        :param base_url: The base URL for the MERRA-2 database.
        :param dataset_name: Name of the dataset (e.g., 'inst3_2d_gas_Nx').
        :param url_params: URL parameters for querying specific variables.
        :return: List of URLs for downloading the data.
        """
        return [Merra2Config.generate_download_link(date, database_name, database_id, url_parameters) for date in dates]

    @staticmethod
    def generate_url_params(parameter, time_slice, latitude, longitude):
        """
        Creates a query string containing all the parameters in query form.

        :param parameter: List of MERRA-2 variables to request (e.g., 'T2M', 'PS').
        :param time_slice: Time slice string for querying time range (e.g., '[0:1:23]').
        :param latitude: Latitude slice string (e.g., '[0:360]').
        :param longitude: Longitude slice string (e.g., '[0:575]').
        :return: Comma-separated URL query string for the parameters.
        """
        params = []
        for param in parameter:
            # For each parameter, add the time, latitude, and longitude slices
            param_with_slices = Merra2Config.generate_url_parameter(
                prouduct_name=param, time_slice=time_slice, latitude=latitude, longitude=longitude)
            params.append(param_with_slices)

# Return all parameters joined by a comma
        return ','.join(params)

    @staticmethod
    def generate_url_parameter(prouduct_name, time_slice, latitude, longitude):
        parameter = str(prouduct_name)
        return f'{parameter}{time_slice}{latitude}{longitude}'

    @staticmethod
    def generate_plain_link(date, database_name, database_id):
        parsed_date = datetime.strptime(date, '%d-%m-%Y')
        y_str = parsed_date.strftime('%Y')  # Full year as string
        m_str = parsed_date.strftime('%m')  # Month as 2-digit string

        # Get the file number based on the year
        if database_id.startswith("const"):
            merra_stream = "101"
        else:
            merra_stream = Merra2Tools.translate_year_to_file_number(
                int(y_str))
        # date string
        # Format the date as 'yyyymmdd'
        date_string = parsed_date.strftime('%Y%m%d')

        # Create the file name string matching the structure MERRA2_400.inst3_2d_gas_Nx.20200207.nc4
        file_name = 'MERRA2_{num}.{name}.{date}.nc4'.format(
            num=merra_stream, name=database_id, date=date_string)

        # Create the query URL
        query_url = '{base}/{database_name}/{y}/{m}/{file_name}?{params}'.format(
            base=Merra2Config.base_url, database_name=database_name, y=y_str, m=m_str, file_name=file_name, params=url_parameters)
        return query_url
