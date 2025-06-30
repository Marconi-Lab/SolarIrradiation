from susse.api_clients import (
    Merra2Config,
    MerraDataFetcher,
    MerraDataStreamFetcher,
    MerraDownloadManager,
    MerraProducts,
)
from datetime import datetime
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="SuSSe")
location = geolocator.geocode("Zurich")

start_date = datetime(2010, 1, 1)
end_date = datetime(2010, 1, 3)

merra_product = MerraProducts.AIR_TEMPERATURE.value
download = "tmp"

data_fetcher = MerraDataFetcher(base_download_folder=download)

data = data_fetcher.fetch_product_result(
    merra_product, location, start_date, end_date
)


var_names = data.get_variable_names()

var_np = data.to_np()


print(f"Product Name:{var_names}")
print('###########################')
print(f"Values:{var_np}")
