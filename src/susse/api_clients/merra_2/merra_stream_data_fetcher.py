from datetime import datetime

import numpy as np
from geopy import location as Glocation
from pydap.client import open_url

from .merra_data_stream_result import MerraStreamDataResult
from .merra_product import MerraProductData, MerraProducts
from .merra_stream_config import MerraStreamConfig
from .merra_stream_session_manager import StreamSessionManager


class MerraDataStreamFetcher:
    """
    A class responsible for fetching MERRA-2 data streams from URLs.

    This class handles session authentication and configuration setup for
    downloading MERRA-2 data. It provides a method for fetching single-day
    data from specified URLs and product configurations.

    Attributes:
    ----------
    authenticate : StreamSessionManager
        An instance responsible for managing session authentication.
    config : MerraStreamConfig
        A configuration object that manages settings for data streaming.
    """

    def __init__(self):
        self._authenticate = StreamSessionManager()
        self._config = MerraStreamConfig()

    def fetch_surface_albedo(
        self, start_date: datetime, end_date: datetime, location: Glocation
    ) -> MerraStreamDataResult:
        return self.fetch_data(
            start_date, end_date, MerraProducts.SURFACE_ALBEDO.value, location
        )

    def fetch_air_temperature(
        self, start_date: datetime, end_date: datetime, location: Glocation
    ) -> MerraStreamDataResult:
        return self.fetch_data(
            start_date, end_date, MerraProducts.AIR_TEMPERATURE.value, location
        )

    def fetch_surface_pressure(
        self, start_date: datetime, end_date: datetime, location: Glocation
    ) -> MerraStreamDataResult:
        return self.fetch_data(
            start_date, end_date, MerraProducts.SURFACE_PRESSURE.value, location
        )

    def fetch_aerosol_extinction(
        self, start_date: datetime, end_date: datetime, location: Glocation
    ) -> MerraStreamDataResult:
        return self.fetch_data(
            start_date, end_date, MerraProducts.AEROSOL_EXTINCTION_550nm.value, location
        )

    def _fetch_single_day_data_from_url(
        self, dataset_url: str, product_data: MerraProductData
    ):
        session = self._authenticate.authenticate(url=dataset_url)

        if session is None:
            raise Exception("Session is not authenticated")

        try:
            dataset = open_url(dataset_url, session=session, protocol="dap4")

            data = dataset[product_data.product_name][:]

            return np.array(data)

        except AttributeError as e:
            print(f"Error fetching data from {dataset_url}: {e}")
            return None

    def fetch_data(
        self,
        start_date: datetime,
        end_date: datetime,
        product_data: MerraProductData,
        location: Glocation,
    ) -> MerraStreamDataResult:
        """
        Fetches the data for a range of dates by first generating URLs and then fetching data from each URL.

        start_date: Start date for data fetching
        end_date: End date for data fetching
        product_data: MERRA product metadata
        location: Location of the place
        product_data: Product name; as defined in the MerraProductData dictionary
        :return: Python dictionary with date as the key and the corresponding data
        """

        # Generate all the URLs for the date range
        dataset_urls = self._config.generate_stream_links(
            start_date=start_date,
            end_date=end_date,
            product_data=product_data,
            location=location,
        )

        data_dict = {}

        for url in dataset_urls:
            current_date = MerraStreamConfig._extract_date_from_url(url)

            single_day_data = self._fetch_single_day_data_from_url(
                dataset_url=url, product_data=product_data
            )

            if single_day_data is not None:
                data_dict[current_date] = single_day_data

        if data_dict:
            return MerraStreamDataResult.from_data(
                data=data_dict,
                product_data=product_data,
                location=location,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            raise Exception("No data was fetched for the specified URLs.")
