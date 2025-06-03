from datetime import datetime
from typing import List, Union

from geopy.location import Location as GeopyLocation

from .nasa_products import NASAPowerProducts, TemporalResolution


class NASAPowerConfig:
    """
    Configuration and utility class for accessing NASA POWER data.
    """

    BASE_URL = "https://power.larc.nasa.gov/api/temporal"

    @staticmethod
    def generate_download_link(
        temporal_resolution: TemporalResolution,
        start_date: datetime,
        end_date: datetime,
        location: GeopyLocation,
        products: Union[NASAPowerProducts, List[NASAPowerProducts]],
        output_format: str = "JSON",
    ) -> str:
        """
        Generates the download link for NASA POWER data.

        Args:
            temporal_resolution: The temporal resolution for the data.
            start_date: The start date for the data query.
            end_date: The end date for the data query.
            location: A geopy.location.Location object with latitude and longitude.
            products: A single NASAPowerProduct or a list of NASAPowerProducts.
            output_format: The desired output format (e.g., 'JSON', 'CSV').

        Returns:
            A string representing the complete URL to download the data.
        """
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        if isinstance(products, NASAPowerProducts):
            param_str = products.value
        else:
            param_str = ",".join([p.value for p in products])

        return (
            f"{NASAPowerConfig.BASE_URL}/{temporal_resolution.value}/point"
            f"?parameters={param_str}"
            f"&community=RE"
            f"&longitude={location.longitude}"
            f"&latitude={location.latitude}"
            f"&start={start_date_str}"
            f"&end={end_date_str}"
            f"&format={output_format}"
        )
