from datetime import datetime
from merra_tools import Merra2Tools


class Merra2Config:
    """
    This class contains the URLs used and the functions for generating download URLs for MERRA-2 data.
    """

    def __init__(self, base_url: str = ""):
        self._base_url = base_url

    def generate_download_link(self, date_str: datetime, dataset_name: str, url_params):

        parsed_date = datetime.strptime(date_str, '%d-%m-%Y')
        y_str = parsed_date.strftime('%Y')  # Full year as string
        m_str = parsed_date.strftime('%m')  # Month as 2-digit string

        # Get the file number based on the year
        file_num = Merra2Tools.translate_year_to_file_number(int(y_str))

        # Format the date as 'yyyymmdd'
        formatted_date = parsed_date.strftime('%Y%m%d')

        # Create the file name string matching the structure MERRA2_400.inst3_2d_gas_Nx.20200207.nc4
        file_name = 'MERRA2_{num}.{name}.{date}.nc4'.format(
            num=file_num, name=dataset_name, date=formatted_date)

        # Create the query URL
        query = '{base}{y}/{m}/{file_name}?{params}'.format(
            base=self._base_url, y=y_str, m=m_str, file_name=file_name, params=url_params)
        return query

 # Parse the date string (dd-mm-yy) into a datetime object

    def generate_download_links(self, dates, dataset_name, url_params):
        """
        Generates download links for MERRA-2 data.

        :param dates: List of dates in 'dd-mm-yy' format.
        :param base_url: The base URL for the MERRA-2 database.
        :param dataset_name: Name of the dataset (e.g., 'inst3_2d_gas_Nx').
        :param url_params: URL parameters for querying specific variables.
        :return: List of URLs for downloading the data.
        """
        return [self.generate_download_link(date_str, dataset_name, url_params) for date_str in dates]

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
            param_with_slices = f'{param}{time_slice}{latitude}{longitude}'
            params.append(param_with_slices)

# Return all parameters joined by a comma
        return ','.join(params)
