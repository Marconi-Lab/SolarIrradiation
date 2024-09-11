import logging
from datetime import datetime
from typing import List

import requests

from ..api_data_fetcher import ApiDataFetcher
from .modis_api_config import ModisConfig
from .modis_data_result import ModisDataResult
from .modis_product import ModisProduct, ModisProductEnum, ModisProductFactory


class ModisDataFetcher(ApiDataFetcher):
    """
    This class handles the data fetching through the Modis API
    """

    def __init__(self):
        super().__init__()
        self._product_factory = ModisProductFactory()

    def fetch_surface_reflectance(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        band_name: str = None,
    ):
        return self.fetch_product_result(
            ModisProductEnum.LAND_SURFACE_TEMPERATURE,
            latitude,
            longitude,
            start_date,
            end_date,
            band_name,
        )

    def fetch_product_result(
        self,
        product_enum: ModisProductEnum,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        band_name: str = None,
    ):
        product = self._product_factory.get_product_by_enum(product_enum)
        band_name = self._validate_band_for_product(product, band_name)
        request_url = ModisConfig.get_product_request_url(
            product_name=product.name,
            latitude=latitude,
            longitude=longitude,
            band_name=band_name,
            start_date=start_date,
            end_date=end_date,
        )

        response = requests.get(request_url)
        if response.status_code == 200:
            product_data = response.json()
            return ModisDataResult.from_request_response(product_data)
        else:
            raise requests.exceptions.HTTPError(
                f"Failed to fetch data for product {product.name}, coordinates {latitude}, {longitude} between dates: {start_date} and {end_date}: \n{response.text}"
            )

    def _validate_band_for_product(
        self, product: ModisProduct, band_name: str = None
    ) -> str:
        if band_name is None:
            if product.default_band is not None:
                return product.default_band.name
            else:
                raise ValueError(
                    f"No band provided for  product {product.name}, and no default band available. "
                    f"Available bands: {product.get_band_names()}"
                )
        else:
            if not product.has_band(band_name):
                if product.default_band is not None:
                    logging.warning(
                        f"Band {band_name} not found in product {product.name}, using default band {product.default_band.name} instead"
                    )
                    return product.default_band.name
                else:
                    raise ValueError(
                        f"Band {band_name} not found in product {product.name}, and no default band available. "
                        f"Available bands: {product.get_band_names()}"
                    )
        return band_name

    def get_available_dates_for_product_and_location(
        self, product: ModisProduct, latitude: float, longitude: float
    ) -> List[datetime]:
        available_dates_url = ModisConfig.get_available_date_url(
            product.name, latitude, longitude
        )
        response = requests.get(available_dates_url)

        if response.status_code == 200:
            available_dates_json = response.json()
            available_dates = [
                datetime.strptime(date_dict["calendar_date"], "%Y-%m-%d")
                for date_dict in available_dates_json["dates"]
            ]
            return available_dates
        else:
            raise requests.exceptions.HTTPError(
                f"Failed to fetch available dates for product {product.name} and coordinates {latitude}, {longitude}: \n{response.text}"
            )
