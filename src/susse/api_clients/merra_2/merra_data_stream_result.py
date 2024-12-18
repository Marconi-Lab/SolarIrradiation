from datetime import datetime
from typing import List, Union
import numpy as np
from geopy import location as Glocation

from .merra_product import MerraProducts, MerraProductData


class MerraStreamDataResult:
    def __init__(
        self,
        data: dict[str, list[float]],
        product_name:str,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Initialize the result object with data, product name, location, and dates.
        """
        self._raw_data = data
        self.product_name = product_name
        self.location = location
        self.start_date = start_date
        self.end_date = end_date
        self.fill_value = 999999986991104.0

    @classmethod
    def from_data(
        cls,
        data: dict,
        product_data: MerraProductData,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Class method to create an instance from raw data.
        """
        return cls(data, product_data.product_name, location, start_date, end_date)

    @property
    def data(self) -> dict[str, list[float]]:
        """
        Return the cleaned data where fill values are replaced with NaN.
        """
        return {
            date: self._clean_day_data(day_data)
            for date, day_data in self._raw_data.items()
        }

    def _clean_day_data(self, day_data: list[float]):
        """
        Clean the data for a single day by replacing fill values with NaN.
        """
        day_data_np = np.array(day_data)
        cleaned_data = np.where(day_data_np == self.fill_value, np.nan, day_data_np)
        return cleaned_data.tolist()
