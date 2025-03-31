from datetime import datetime
from .nasa_products import NASAPowerProducts , TemporalResolution
from geopy import location as Glocation
from typing import List, Union


class NASAPowerConfig:

    BASE_URL = 'https://power.larc.nasa.gov/api/temporal'

    @staticmethod
    def generate_download_link(
        temporal_resolution: TemporalResolution,
        start_date: datetime,
        end_date: datetime,
        location: Glocation,
        products: Union[NASAPowerProducts, List[NASAPowerProducts]],
        format: str = 'JSON'
    ) -> str:
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        if isinstance(products, NASAPowerProducts):
            param_str = products.value
        else:
            param_str = ','.join([p.value for p in products])
            
        return (
            f"{NASAPowerConfig.BASE_URL}/{temporal_resolution.value}/point?"
            f"parameters={param_str}&community=RE&"
            f"longitude={location.longitude}&latitude={location.latitude}&"
            f"start={start_date_str}&end={end_date_str}&format={format}"
        )
