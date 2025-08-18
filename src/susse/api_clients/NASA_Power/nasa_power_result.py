from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from geopy.location import Location as GeopyLocation

from .nasa_products import NASAPowerProduct


class NASAPowerResult:
    """
    Represents the result of a NASA POWER data request for one or multiple products.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, Union[float, int]]],
        products: List[NASAPowerProduct],
        location: GeopyLocation,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Initialize a NASA POWER result.

        Args:
            data: Dictionary mapping product values to their time series data.
                Each time series is a dict mapping timestamps to values.
            products: List of NASA POWER products included in this result.
            location: The location for which data was fetched.
            start_date: Start date of the data period.
            end_date: End date of the data period.
        """
        self._raw_data = data
        self._products = products
        self._location = location
        self._start_date = start_date
        self._end_date = end_date

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all parameter data to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with timestamps as index and products as columns.
        """
        dfs = []
        for product in self._products:
            if product.value not in self._raw_data:
                continue

            product_data = self._raw_data[product.value]
            df = pd.DataFrame.from_dict(
                product_data,
                orient="index",
                columns=[product.name],
            )
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=1)

    def to_numpy(self, product: Optional[NASAPowerProduct] = None) -> np.ndarray:
        """
        Convert data to a NumPy array for a specific product or all products.

        Args:
            product: Optional product to get data for. If None, returns data for all products.

        Returns:
            np.ndarray: Array of values, sorted by timestamp.
            If no product specified, returns 2D array with shape (timestamps, products).

        Raises:
            ValueError: If the specified product is not in the result.
        """
        if product is not None:
            if product.value not in self._raw_data:
                raise ValueError(f"Product {product.name} not found in result")
            sorted_timestamps = sorted(self._raw_data[product.value].keys())
            return np.array(
                [self._raw_data[product.value][ts] for ts in sorted_timestamps]
            )

        # Get all unique timestamps
        all_timestamps: set[str] = set()
        for product_data in self._raw_data.values():
            all_timestamps.update(product_data.keys())
        sorted_timestamps = sorted(all_timestamps)

        # Create 2D array with NaN for missing values
        result = np.full((len(sorted_timestamps), len(self._products)), np.nan)
        for i, product in enumerate(self._products):
            if product.value not in self._raw_data:
                continue
            product_data = self._raw_data[product.value]
            for j, ts in enumerate(sorted_timestamps):
                if ts in product_data:
                    result[j, i] = product_data[ts]

        return result

    def get_parameter_data(
        self, product: NASAPowerProduct
    ) -> Optional[Dict[str, Union[float, int]]]:
        """
        Get the raw data for a specific parameter.

        Args:
            product: The NASA POWER product to retrieve data for.

        Returns:
            Optional[Dict[str, Union[float, int]]]: The raw data for the parameter,
            or None if the parameter was not in the request.
        """
        return self._raw_data.get(product.value)

    @property
    def products(self) -> List[NASAPowerProduct]:
        """Get the list of products in this result."""
        return self._products

    @property
    def location(self) -> GeopyLocation:
        """Get the location for this result."""
        return self._location

    @property
    def start_date(self) -> datetime:
        """Get the start date of the data period."""
        return self._start_date

    @property
    def end_date(self) -> datetime:
        """Get the end date of the data period."""
        return self._end_date
