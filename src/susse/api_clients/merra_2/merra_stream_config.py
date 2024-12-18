import re
from datetime import datetime, timedelta
from typing import List

from geopy import Location

from .merra_config import Merra2Config
from .merra_product import MerraProductData


class MerraStreamConfig:
    """
    A configuration class for generating MERRA-2 data stream URLs.

    This class manages configuration settings and provides methods for generating
    download links for MERRA-2 data.

    """

    def __init__(self):
        self._merra_config = Merra2Config()

    def _generate_stream_link(
        self, date: datetime, product_data: MerraProductData, location: Location
    ) -> str:
        """
        Generates a URL for streaming MERRA-2 data for a specific date and location.

        Parameters:
        ----------
        date : datetime
            The date for which the data is being fetched.
        product_data : MerraProductData
            The product information including product name and metadata.
        location : Location
            The geographical coordinates (latitude and longitude) of the data request.
        """
        file_name = self._merra_config.create_file_name(date, product_data)
        m_str = str(date.month).zfill(2)
        y_str = str(date.year)

        lat_geos5 = self._merra_config._translate_lat_to_geos5_native(location.latitude)
        lon_geos5 = self._merra_config._translate_lon_to_geos5_native(
            location.longitude
        )

        merra_lat = self._merra_config._find_closest_merra_coordinate(
            lat_geos5, self._merra_config.MERRA_LAT_COORDS
        )
        merra_lon = self._merra_config._find_closest_merra_coordinate(
            lon_geos5, self._merra_config.MERRA_LON_COORDS
        )

        suffix = f"dap4.ce=/{product_data.product_name}[0:1:23][{merra_lat}:1:{merra_lat}][{merra_lon}:1:{merra_lon}]"

        url = f"{self._merra_config.generate_database_url(product_data)}/{y_str}/{m_str}/{file_name}?{suffix}"

        return url

    def generate_stream_links(
        self,
        start_date: datetime,
        end_date: datetime,
        product_data: MerraProductData,
        location: Location,
    ) -> List[str]:
        date_range = [
            start_date + timedelta(days=x)
            for x in range((end_date - start_date).days + 1)
        ]

        # Generate links for each date in the date range
        return [
            self._generate_stream_link(date, product_data, location)
            for date in date_range
        ]
    
    @staticmethod
    def _extract_date_from_url(url: str) -> str:
        """
        Extracts the date from a URL with separate year, month, and day segments.

        :param url: The dataset URL.
        :return: The date in 'YYYY-MM-DD' format as a string.
        :raises ValueError: If the date cannot be extracted from the URL.
        """
        match = re.search(r"(\d{4})(\d{2})(\d{2})", url)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
        else:
            raise ValueError(f"Date could not be extracted from URL: {url}")
