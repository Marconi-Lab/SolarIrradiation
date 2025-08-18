from datetime import datetime
from typing import List, Union

import requests
from geopy.location import Location as GeopyLocation

from .nasa_power_config import NASAPowerConfig
from .nasa_power_result import NASAPowerResult
from .nasa_products import NASAPowerProduct, TemporalResolution


class NASAPowerFetchData:
    """
    Handles fetching data from the NASA POWER API.

    This class provides methods to fetch single or multiple parameters from the NASA POWER API
    while respecting the API's limitation of maximum 20 parameters per request.
    """

    @staticmethod
    def _fetch_parameter_batch(
        start_date: datetime,
        end_date: datetime,
        location: GeopyLocation,
        products: List[NASAPowerProduct],
        temporal_resolution: TemporalResolution,
    ) -> dict:
        """
        Fetch a single batch of parameters (max 20) in one request.

        Args:
            start_date: The start date for the data query.
            end_date: The end date for the data query.
            location: A geopy Location object containing latitude and longitude.
            products: List of NASA POWER products to fetch (max 20 per batch).
            temporal_resolution: The temporal resolution for the data (e.g., DAILY, MONTHLY).

        Returns:
            dict: Raw parameter data from the NASA POWER API response.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
            ValueError: If more than 20 products are passed in a single batch.
        """
        if len(products) > 20:
            raise ValueError("Maximum 20 parameters allowed per batch")

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
        return json_data["properties"]["parameter"]

    @staticmethod
    def fetch_multiple_parameters(
        start_date: datetime,
        end_date: datetime,
        location: GeopyLocation,
        products: Union[NASAPowerProduct, List[NASAPowerProduct]],
        temporal_resolution: TemporalResolution = TemporalResolution.DAILY,
    ) -> NASAPowerResult:
        """
        Fetch one or multiple parameters from the NASA POWER API.

        This method handles both single and multiple parameter requests, automatically
        batching requests into groups of 20 parameters to comply with API limitations.

        Args:
            start_date: The start date for the data query.
            end_date: The end date for the data query.
            location: A geopy Location object containing latitude and longitude.
            products: Single NASAPowerProduct or list of products to fetch.
            temporal_resolution: The temporal resolution for the data (default: DAILY).

        Returns:
            NASAPowerResult: Contains the fetched data for all requested products.

        Raises:
            requests.exceptions.HTTPError: If any API request fails.
            ValueError: If invalid parameters are provided.
        """
        # Convert single product to list for uniform handling
        if isinstance(products, NASAPowerProduct):
            products = [products]

        # Process products in batches of 20
        BATCH_SIZE = 20
        all_parameters = {}

        for i in range(0, len(products), BATCH_SIZE):
            product_batch = products[i : i + BATCH_SIZE]
            batch_parameters = NASAPowerFetchData._fetch_parameter_batch(
                start_date=start_date,
                end_date=end_date,
                location=location,
                products=product_batch,
                temporal_resolution=temporal_resolution,
            )
            all_parameters.update(batch_parameters)

        return NASAPowerResult(
            data=all_parameters,
            products=products,
            location=location,
            start_date=start_date,
            end_date=end_date,
        )
