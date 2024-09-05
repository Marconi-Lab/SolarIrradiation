import datetime
import json
import logging
from calendar import monthrange
from datetime import timedelta
from typing import List

import certifi
import numpy as np
import requests
import urllib3

from ..api_data_fetcher import ApiDataFetcher


class ModisDataFetcher(ApiDataFetcher):
    """
    This class handles the data fetching through the Modis API
    """

    _MODIS_URL = "https://modis.ornl.gov/rst/api/v1/"

    def __init__(
        self,
        band: str = "",
        product: str = "",
        kmAB: int = 1,
        kmLR: int = 1,
        prod_data: list = [],
    ):
        """
        :param band: put description and explanation here!
        :param product:
        :param kmAB:
        :param kmLR:
        :param prod_data:
        """

        super().__init__(self._MODIS_URL)

        self._product = product
        self._band = band  # Why do you need to save the band?
        self._kmAB = kmAB
        self._kmLR = kmLR
        self._prod_data = prod_data if prod_data is not None else []

    # What does this function do? do you really need this one-line function? if yes, rename it so that it is clear what it does! does it need to be public?
    def cal_to_modis(self, cal_date: str) -> str:
        return "A" + datetime.datetime.strptime(cal_date, "%Y-%m-%d").strftime("%Y%j")

    # same here, what does this function do? why is it public?
    def request_URL(self, latitude: float, longitude: float, dates: List[str]) -> str:
        return str(
            self.base_url
            + self._product
            + "/subset?"
            + "latitude="
            + str(latitude)
            + "&longitude="
            + str(longitude)
            + "&band="
            + self._band
            + "&startDate="
            + dates[0]
            + "&endDate="
            + dates[-1]
            + "&kmAboveBelow="
            + str(self._kmAB)
            + "&kmLeftRight="
            + str(self._kmLR)
        )

    # rename to get_surface_reflectance. date inputs should be List[datetime.datetime] not str. why does it need to have inputs band and product? the user will generally not know what these are. Avoid using strings. make instaed Enums out of them
    def surface_reflectance(
        self,
        latitude: float,
        longitude: float,
        dates: List[str],
        band: str = "sur_refl_b01",
        product: str = "MOD09A1",
    ) -> np.ndarray:
        self._band = band  # why do you save all this?
        self._product = product
        start = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
        while start <= datetime.datetime.strptime(dates[-1], "%Y-%m-%d"):
            subset = requests.get(
                self.request_URL(
                    latitude,
                    longitude,
                    dates=[self.cal_to_modis(start.strftime("%Y-%m-%d"))],
                )
            )
            if subset.status_code == 400:  # avoid magic numbers, use enums instead!
                start += timedelta(days=1)
                print(
                    "Skipping date"
                )  # dont print messages, use logging.warning("...") or logging.info("..." ) where needed
                continue
            else:
                data = json.loads(subset.text)
                self._prod_data.append(data["subset"][0]["data"])
                scale = float(data["scale"])
                scaled_data = scale * np.array(self._prod_data)
                start += timedelta(days=1)
        return scaled_data

    def vegetation_index(
        self,
        latitude: float,
        longitude: float,
        dates: list,
        band: str = "500_m_16_days_NDVI",
        product: str = "VNP13A1",
    ) -> np.ndarray:
        self._band = band
        self._product = product
        start = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
        while start <= datetime.datetime.strptime(dates[-1], "%Y-%m-%d"):
            subset = requests.get(
                self.request_URL(
                    latitude,
                    longitude,
                    dates=[self.cal_to_modis(start.strftime("%Y-%m-%d"))],
                )
            )
            if subset.status_code == 400:
                start += timedelta(days=1)
                print("Skipping date")
                continue
            else:
                data = json.loads(subset.text)
                self._prod_data.append(data["subset"][0]["data"])
                scale = float(data["scale"])
                scaled_data = scale * np.array(self._prod_data)
                start += timedelta(days=1)
        return scaled_data

    def surface_temperature(
        self,
        latitude: float,
        longitude: float,
        dates: list,
        band: str = "LST_Day_1KM",
        product: str = "MOD21A2",
    ) -> np.ndarray:
        self._band = band
        self._product = product
        start = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
        while start <= datetime.datetime.strptime(dates[-1], "%Y-%m-%d"):
            subset = requests.get(
                self.request_URL(
                    latitude,
                    longitude,
                    dates=[self.cal_to_modis(start.strftime("%Y-%m-%d"))],
                )
            )
            if subset.status_code == 400:
                start += timedelta(days=1)
                print("Skipping date")
                continue
            else:
                data = json.loads(subset.text)
                self._prod_data.append(data["subset"][0]["data"])
                scale = float(data["scale"])
                scaled_data = scale * np.array(self._prod_data)
                start += timedelta(days=1)
        return scaled_data
