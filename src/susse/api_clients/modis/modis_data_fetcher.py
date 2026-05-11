"""Thin HTTP wrapper around the ORNL DAAC MODIS subset endpoint.

Single public entry point: :meth:`ModisDataFetcher.fetch_product_result`,
which fetches one (product, band, location, date-range) slice and returns
a :class:`ModisDataResult`. The richer batching and parallelism live in
:class:`susse.api_clients.modis.ModisLongFetcher`.
"""

import logging
from datetime import datetime
from typing import Optional

import requests
from geopy import location as Glocation

from .modis_api_config import ModisConfig
from .modis_data_result import ModisDataResult
from .modis_product import ModisProduct, ModisProductEnum, ModisProductFactory


class ModisDataFetcher:
    """Single-request HTTP client for the ORNL DAAC MODIS subset API."""

    def __init__(self) -> None:
        self._product_factory = ModisProductFactory()

    def fetch_product_result(
        self,
        product_enum: ModisProductEnum,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
        band_name: Optional[str] = None,
    ) -> ModisDataResult:
        product = self._product_factory.get_product_by_enum(product_enum)
        band_name = self._validate_band_for_product(product, band_name)
        request_url = ModisConfig.get_product_request_url(
            product_name=product.name,
            location=location,
            band_name=band_name,
            start_date=start_date,
            end_date=end_date,
        )
        response = requests.get(request_url)
        if response.status_code == 200:
            return ModisDataResult.from_request_response(response.json())
        message = (
            f"Failed to fetch data for product {product.name}, coordinates "
            f"{location.latitude} {location.longitude} between dates: "
            f"{start_date} and {end_date}: \n{response.text}"
        )
        raise requests.exceptions.HTTPError(message)

    @staticmethod
    def _validate_band_for_product(
        product: ModisProduct, band_name: Optional[str] = None
    ) -> str:
        if band_name is None:
            if product.default_band is None:
                raise ValueError(
                    f"No band provided for product {product.name}, and no "
                    f"default band available. Available bands: "
                    f"{product.get_band_names()}"
                )
            return product.default_band.name
        if product.has_band(band_name):
            return band_name
        if product.default_band is not None:
            logging.warning(
                "Band %s not found in product %s, using default band %s instead",
                band_name,
                product.name,
                product.default_band.name,
            )
            return product.default_band.name
        raise ValueError(
            f"Band {band_name} not found in product {product.name}, and no "
            f"default band available. Available bands: "
            f"{product.get_band_names()}"
        )
