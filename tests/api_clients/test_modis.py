from datetime import datetime

import numpy as np
import pytest
from geopy.geocoders import Nominatim

from susse import ModisProductEnum
from susse.api_clients import ModisDataFetcher, ModisProductFactory


# @pytest.mark.skip(
#     reason="Modis service seems to be temporarily unavailable. This needs to be re-evaluated"
# )
def test_modis_data_fetcher():

    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010, 1, 30)

    data_fetcher = ModisDataFetcher()

    factory = ModisProductFactory()
    product = factory.get_product_by_enum(ModisProductEnum.LAND_SURFACE_TEMPERATURE)
    result = data_fetcher.fetch_temp_day(location, start_date, end_date)
    average_day_temperature = result.get_time_average()

    assert average_day_temperature is not None

    available_dates = data_fetcher.get_available_dates_for_product_and_location(
        product, location=location
    )

    assert available_dates

    # test Reflectance
    reflectance = data_fetcher.fetch_surface_reflectance(
        location=location,
        start_date=start_date,
        end_date=end_date,
    )
    reflectance_value = reflectance.get_time_average()
    expected_reflectance = 0.15466800000000003
    assert (
        expected_reflectance == reflectance_value
    ), f"Reflectance mismatch: {reflectance_value} != {expected_reflectance}"

    reflectance_array = reflectance.to_np()
    reflectance_times = reflectance.time
    expected_reflectance_np = np.asarray([0.22294, 0.140864, 0.162352, 0.092516])
    assert type(reflectance_times[0]) is datetime
    assert len(reflectance_times) == len(reflectance_array)
    assert np.allclose(reflectance_array, expected_reflectance_np)

    # test Emissivity
    emissivity = data_fetcher.fetch_emissivity(
        location=location,
        start_date=start_date,
        end_date=end_date,
    )
    emissivity_value = emissivity.get_time_average()
    expected_emissivity = 0.4783888888888889
    assert (
        expected_emissivity == emissivity_value
    ), f"Emissivity mismatch: {emissivity_value} != {expected_emissivity}"
