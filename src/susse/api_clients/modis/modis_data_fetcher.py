import logging
from datetime import datetime
from typing import List, Optional

import requests
from geopy import location as Glocation

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

    def fetch_temp_day(
        self,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
    ) -> ModisDataResult:
        return self.fetch_product_result(
            ModisProductEnum.LAND_SURFACE_TEMPERATURE,
            location,
            start_date,
            end_date,
            "LST_Day_1KM",
        )

    def fetch_temp_night(
        self,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
    ) -> ModisDataResult:
        return self.fetch_product_result(
            ModisProductEnum.LAND_SURFACE_TEMPERATURE,
            location,
            start_date,
            end_date,
            "LST_Night_1KM",
        )

    def fetch_surface_reflectance(
        self,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
        band_name: str = None,
    ) -> ModisDataResult:
        return self.fetch_product_result(
            ModisProductEnum.SURFACE_REFLACTANCE,
            location,
            start_date,
            end_date,
            band_name,
        )

    def fetch_emissivity(
        self,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
        band_name: str = None,
    ) -> ModisDataResult:
        return self.fetch_product_result(
            ModisProductEnum.EMISSIVITY,
            location,
            start_date,
            end_date,
            band_name,
        )

    def fetch_product_result(
        self,
        product_enum: ModisProductEnum,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
        band_name: str = None,
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
            product_data = response.json()
            return ModisDataResult.from_request_response(product_data)
        else:
            message = (
                "Failed to fetch data for product {}, coordinates {} {} "
                "between dates: {} and {}: \n{}"
            ).format(
                product.name,
                location.latitude,
                location.longitude,
                start_date,
                end_date,
                response.text,
            )
            raise requests.exceptions.HTTPError(message)

    def _validate_band_for_product(
        self, product: ModisProduct, band_name: Optional[str] = None
    ) -> str:
        if band_name is None:
            if product.default_band is not None:
                return product.default_band.name
            else:
                message = (
                    "No band provided for product {}, and no default band available."
                    "Available bands {}"
                ).format(product.name, product.get_band_names())
                raise ValueError(message)
        else:
            if not product.has_band(band_name):
                if product.default_band is not None:
                    message = (
                        "Band {} not found in product {}, using default band {} instead"
                    ).format(band_name, product.name, product.default_band.name)
                    logging.warning(message)

                    return product.default_band.name
                else:
                    message = (
                        "Band {} not found in product {}, and no default band available. "
                        "Available bands: {}"
                    ).format(band_name, product.name, product.get_band_names())
                    raise ValueError(message)

        return band_name

    def get_available_dates_for_product_and_location(
        self, product: ModisProduct, location: Glocation
    ) -> List[datetime]:
        available_dates_url = ModisConfig.get_available_date_url(
            product.name,
            location,
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
            message = (
                "Failed to fetch available dates for product {}, "
                "and coordinates {}, {}: \n{}"
            ).format(product.name, location.latitude, location.longitude, response.text)
            raise requests.exceptions.HTTPError(message)
