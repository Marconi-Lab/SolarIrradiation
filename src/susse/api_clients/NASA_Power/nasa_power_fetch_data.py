from .nasa_power_config import NASAPowerConfig
from .nasa_products import NASAPowerProducts
import requests
from .nasa_power_result import NASAPowerDataResult, NASAPowerMultiDataResult


class NASAPowerFetchData:
    def __init__(self):
        self._config = NASAPowerConfig()
    


    def fetch_data(self, temporal_resolution, start_date, end_date, location, product):
        url = NASAPowerConfig.generate_download_link(
            temporal_resolution, start_date, end_date, location, product
        )
        response = requests.get(url)
        response.raise_for_status() 
        json_data = response.json()
        parameter_data = json_data['properties']['parameter'][product.value]
        data_units = json_data['parameters'][product.value]['units']
        return NASAPowerDataResult.from_data(
            data=parameter_data,
            product=product,
            location=location,
            start_date=start_date,
            end_date=end_date
        )

    def fetch_multiple_parameters(self, temporal_resolution, start_date, end_date, 
                                location, products):
        """Fetch multiple parameters in one request"""
        url = NASAPowerConfig.generate_download_link(
            temporal_resolution, start_date, end_date, location, products
        )
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        all_parameters = json_data['properties']['parameter']
        return NASAPowerMultiDataResult(
            data=all_parameters,
            products=products,
            location=location,
            start_date=start_date,
            end_date=end_date
        )
