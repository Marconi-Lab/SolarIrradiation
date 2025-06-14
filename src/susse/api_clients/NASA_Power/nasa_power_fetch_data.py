from datetime import datetime
from typing import List

import requests
from geopy.location import Location as GeopyLocation

from .nasa_power_config import NASAPowerConfig
from .nasa_power_result import NASAPowerDataResult, NASAPowerMultiDataResult
from .nasa_products import NASAPowerProduct, TemporalResolution


class NASAPowerFetchData:
    """
    Handles fetching data from the NASA POWER API.
    """

    @staticmethod
    def fetch_data(
        start_date: datetime,
        end_date: datetime,
        location: GeopyLocation,
        product: NASAPowerProduct,
        temporal_resolution: TemporalResolution = TemporalResolution.DAILY,
    ) -> NASAPowerDataResult:
        """
        Fetches data for a single product from the NASA POWER API.

        Args:
            temporal_resolution: The temporal resolution for the data.
            start_date: The start date for the data query.
            end_date: The end date for the data query.
            location: A location object with latitude and longitude.
            product: The specific NASA POWER product to fetch.

        Returns:
            A NASAPowerDataResult object containing the fetched data.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        url = NASAPowerConfig.generate_download_link(
            temporal_resolution=temporal_resolution,
            start_date=start_date,
            end_date=end_date,
            location=location,
            products=product,
        )
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()

        parameter_data = json_data["properties"]["parameter"][product.value]

        return NASAPowerDataResult.from_data(
            data=parameter_data,
            product=product,
            location=location,
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def fetch_multiple_parameters(
        start_date: datetime,
        end_date: datetime,
        location: GeopyLocation,
        products: List[NASAPowerProduct],
        temporal_resolution: TemporalResolution = TemporalResolution.DAILY,
    ) -> NASAPowerMultiDataResult:
        """Fetch multiple parameters in one request."""
        url = NASAPowerConfig.generate_download_link(
            temporal_resolution=temporal_resolution,
            start_date=start_date,
            end_date=end_date,
            location=location,
            products=products,
        )
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()

        all_parameters = json_data["properties"]["parameter"]

        return NASAPowerMultiDataResult(
            data=all_parameters,
            products=products,
            location=location,
            start_date=start_date,
            end_date=end_date,
        )
