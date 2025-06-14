from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from geopy.location import Location as GeopyLocation

from .nasa_products import NASAPowerProduct


class NASAPowerDataResult:
    """
    Represents the result of a NASA POWER data request for a single product.
    """

    def __init__(
        self,
        data: Dict[str, Union[float, int]],
        product: NASAPowerProduct,
        location: GeopyLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        self._raw_data = data
        self._product = product
        self._location = location
        self._start_date = start_date
        self._end_date = end_date

    @classmethod
    def from_data(
        cls,
        data: Dict[str, Union[float, int]],
        product: NASAPowerProduct,
        location: GeopyLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> "NASAPowerDataResult":
        """
        Factory method to create an instance from raw data.
        """
        return cls(data, product, location, start_date, end_date)

    def to_numpy(self) -> np.ndarray:
        """
        Converts the raw data for the product to a NumPy array.

        Assumes timestamps in raw_data are sortable and correspond to values.
        """
        sorted_timestamps = sorted(self._raw_data.keys())
        return np.array([self._raw_data[ts] for ts in sorted_timestamps])


class NASAPowerMultiDataResult:
    """
    Represents the result of a NASA POWER data request for multiple products.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, Union[float, int]]],
        products: List[NASAPowerProduct],
        location: GeopyLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        self._raw_data = data
        self._products = products
        self._location = location
        self._start_date = start_date
        self._end_date = end_date

    def get_parameter_data(
        self, product: NASAPowerProduct
    ) -> Optional[Dict[str, Union[float, int]]]:
        """
        Retrieves the raw time-series data for a specific product.

        Args:
            product: The NASAPowerProduct to retrieve data for.

        Returns:
            A dictionary of timestamp-value pairs for the product, or None if not found.
        """
        return self._raw_data.get(product.value)

    def to_numpy(self, product: NASAPowerProduct) -> np.ndarray:
        """
        Converts the raw data for a specific product to a NumPy array.

        Args:
            product: The NASAPowerProduct to convert.

        Returns:
            A NumPy array of the data values for the specified product.

        Raises:
            ValueError: If the specified product is not found in the data.
        """
        product_data = self.get_parameter_data(product)
        if product_data is None:
            raise ValueError(f"Parameter {product.value} not found in data")

        sorted_timestamps = sorted(product_data.keys())
        return np.array([product_data[ts] for ts in sorted_timestamps])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts all product data into a single Pandas DataFrame.

        The DataFrame will have a 'timestamp' index and columns for each product.
        Assumes that all products share the same set of sorted timestamps.
        """
        df = pd.DataFrame()

        first_product_data = None
        if self._products:
            first_product_data = self.get_parameter_data(self._products[0])

        if not first_product_data:
            return df

        shared_sorted_timestamps_str = sorted(first_product_data.keys())

        for product in self._products:
            data = self.get_parameter_data(product)
            if data:
                df[product.name] = [data.get(ts) for ts in shared_sorted_timestamps_str]
            else:
                df[product.name] = [np.nan] * len(shared_sorted_timestamps_str)

        try:
            df["timestamp"] = pd.to_datetime(
                shared_sorted_timestamps_str, format="%Y%m%d"
            )
        except ValueError:
            df["timestamp"] = pd.to_datetime(shared_sorted_timestamps_str)

        return df.set_index("timestamp")
