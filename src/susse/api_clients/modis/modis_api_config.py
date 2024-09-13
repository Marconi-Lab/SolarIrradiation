from datetime import datetime


class ModisConfig:
    """
    This class provides the url calls for the Modis API and also save all keywords and urls related to basic api calls
    """

    BASE_URL = "https://modis.ornl.gov/rst/api/v1"
    HEADERS = {"Accept": "application/json"}

    PRODUCTS = "/products"
    BANDS = "/bands"

    MODIS_DATE_FORMAT = "%Y%j"

    @staticmethod
    def get_product_url() -> str:
        return f"{ModisConfig.BASE_URL}{ModisConfig.PRODUCTS}"

    @staticmethod
    def get_band_url(product_name: str) -> str:
        return f"{ModisConfig.BASE_URL}/{product_name}{ModisConfig.BANDS}"

    @staticmethod
    def get_product_request_url(
        product_name: str,
        latitude: float,
        longitude: float,
        band_name: str,
        start_date: datetime,
        end_date: datetime,
        kmAB: int = 1,
        kmLR: int = 1,
    ) -> str:
        return f"{ModisConfig.BASE_URL}/{product_name}/subset?latitude={latitude}&longitude={longitude}&band={band_name}&startDate=A{start_date.strftime(ModisConfig.MODIS_DATE_FORMAT)}&endDate=A{end_date.strftime(ModisConfig.MODIS_DATE_FORMAT)}&kmAboveBelow={kmAB}&kmLeftRight={kmLR}"

    @staticmethod
    def get_available_date_url(
        product_name: str, latitude: float, longitude: float
    ) -> str:
        return f"{ModisConfig.BASE_URL}/{product_name}/dates?latitude={latitude}&longitude={longitude}"
