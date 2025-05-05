from datetime import datetime
import numpy as np
from geopy import location as Glocation
from .nasa_products import NASAPowerProducts
from typing import List
import pandas as pd

class NASAPowerDataResult:
    def __init__(
        self,
        data, 
        product: NASAPowerProducts,
        location: Glocation,
        start_date: datetime,
        end_date: datetime
    ):
        self._raw_data = data
        self._product = product
        self._location = location
        self._start_date = start_date
        self._end_date = end_date

    @classmethod
    def from_data(
            cls, 
            data,
            product, 
            location: Glocation,
            start_date: datetime,
            end_date: datetime
    ):
        return cls(data, product, location, start_date, end_date)

    def to_numpy(self):
        sorted_timestamps = sorted(self._raw_data.keys())
        return np.array([self._raw_data[ts] for ts in sorted_timestamps])

    def to_csv(self):
        pass

    def units(self):
        pass



class NASAPowerMultiDataResult:
    def __init__(
        self,
        data: dict,
        products: List[NASAPowerProducts],
        location: Glocation,
        start_date: datetime,
        end_date: datetime
    ):
        self._raw_data = data
        self._products = products
        self._location = location
        self._start_date = start_date
        self._end_date = end_date

    def get_parameter_data(self, product: NASAPowerProducts):
        return self._raw_data.get(product.value)

    def to_numpy(self, product: NASAPowerProducts):
        data = self.get_parameter_data(product)
        if data is None:
            raise ValueError(f"Parameter {product.value} not found in data")
        sorted_ts = sorted(data.keys())
        return np.array([data[ts] for ts in sorted_ts])

    def to_dataframe(self):
        df = pd.DataFrame()
        for product in self._products:
            data = self.get_parameter_data(product)
            sorted_ts = sorted(data.keys())
            df[product.name] = [data[ts] for ts in sorted_ts]
        df['timestamp'] = sorted_ts
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d')
        return df.set_index('timestamp')


