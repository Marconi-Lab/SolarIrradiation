from datetime import datetime
from typing import Dict, List, Union

import numpy as np
from geopy import location as Glocation

from .merra_product import MerraProductData, MerraProducts


class MerraStreamDataResult:

    _FILL_VALUE = 999999986991104.0

    def __init__(
        self,
        data: Dict[str, List[float]],
        product_name: str,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Initialize the result object with data, product name, location, and dates.
        """
        self._raw_data = data
        self._product_name = product_name
        self._location = location
        self._start_date = start_date
        self._end_date = end_date
        self._fill_value = self._FILL_VALUE

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
    def product_name(self) -> str:
        return self._product_name

    @property
    def data(self) -> Dict[str, List[float]]:
        """
        Return the cleaned data where fill values are replaced with NaN.
        """
        return {
            date: self._clean_day_data(day_data)
            for date, day_data in self._raw_data.items()
        }

    def _clean_day_data(self, day_data: List[float]) -> List[float]:
        """
        Clean the data for a single day by replacing fill values with NaN.
        """
        day_data_np = np.array(day_data)
        cleaned_data = np.where(day_data_np == self._fill_value, np.nan, day_data_np)
        return cleaned_data.tolist()
