from datetime import datetime

from mypyc.primitives.misc_ops import stop_async_iteration_op


class ModisConfig:
    BASE_URL = "https://modis.ornl.gov/rst/api/v1"
    HEADERS = {"Accept": "application/json"}

    PRODUCTS = "/products"
    BANDS = "/bands"

    DATE_FORMAT = "%Y%j"

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
        return f"{ModisConfig.BASE_URL}/{product_name}/subset?latitude={latitude}&longitude={longitude}&band={band_name}&startDate=A{start_date.strftime(ModisConfig.DATE_FORMAT)}&endDate=A{end_date.strftime(ModisConfig.DATE_FORMAT)}&kmAboveBelow={kmAB}&kmLeftRight={kmLR}"

    @staticmethod
    def get_available_date_url(
        product_name: str, latitude: float, longitude: float
    ) -> str:
        return f"{ModisConfig.BASE_URL}/{product_name}/dates?latitude={latitude}&longitude={longitude}"
