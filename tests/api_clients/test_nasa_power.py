from datetime import datetime

import numpy as np
import pytest
import requests
from geopy import Point

from susse.api_clients import (
    NASAPowerConfig,
    NASAPowerDataResult,
    NASAPowerFetchData,
    NASAPowerProduct,
    TemporalResolution,
)


@pytest.fixture
def dates() -> tuple[datetime, datetime]:
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 3)
    return start, end


@pytest.fixture
def location() -> Point:
    return Point(latitude=33.6, longitude=1.3)


@pytest.fixture
def fetcher() -> NASAPowerFetchData:
    return NASAPowerFetchData()


@pytest.mark.parametrize(
    "resolution, code",
    [
        (TemporalResolution.HOURLY, "hourly"),
        (TemporalResolution.DAILY, "daily"),
    ],
)
def test_url_generation(resolution, code, dates, location):
    start, end = dates

    url = NASAPowerConfig.generate_download_link(
        resolution,
        start,
        end,
        location,
        NASAPowerProduct.SURFACE_PRESSURE,
    )

    assert f"/temporal/{code}/point" in url
    assert "parameters=PS" in url
    assert f"latitude={location.latitude}" in url
    assert f"start={start.strftime('%Y%m%d')}" in url
    assert f"end={end.strftime('%Y%m%d')}" in url


def test_fetch_data_success(requests_mock, fetcher, dates, location):
    start, end = dates
    product = NASAPowerProduct.SURFACE_PRESSURE

    # Prepare mock URL and JSON response
    url = NASAPowerConfig.generate_download_link(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        product,
    )
    mock_data = {
        "properties": {
            "parameter": {
                "PS": {
                    "2020010100": 89.0,
                    "2020010101": 89.0,
                    "2020010102": 88.99,
                }
            }
        }
    }
    requests_mock.get(url, json=mock_data)

    result = fetcher.fetch_data(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        product,
    )

    assert isinstance(result, NASAPowerDataResult)
    assert result._product is product
    assert len(result._raw_data) == 3


def test_fetch_data_http_error(requests_mock, fetcher, dates, location):
    start, end = dates
    product = NASAPowerProduct.SURFACE_PRESSURE

    url = NASAPowerConfig.generate_download_link(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        product,
    )
    requests_mock.get(url, exc=requests.exceptions.RequestException)

    with pytest.raises(requests.exceptions.RequestException):
        fetcher.fetch_data(
            TemporalResolution.HOURLY,
            start,
            end,
            location,
            product,
        )


@pytest.fixture
def sample_result(dates, location) -> NASAPowerDataResult:
    start, end = dates
    data = {
        "2020010100": 89.0,
        "2020010101": 89.5,
        "2020010102": 88.9,
    }
    return NASAPowerDataResult(
        data=data,
        product=NASAPowerProduct.SURFACE_PRESSURE,
        location=location,
        start_date=start,
        end_date=end,
    )


def test_to_numpy_conversion(sample_result):
    arr = sample_result.to_numpy()

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    np.testing.assert_array_equal(arr, np.array([89.0, 89.5, 88.9]))


def test_timestamp_ordering(location, dates):
    start, end = dates
    unsorted = {
        "2020010102": 88.9,
        "2020010100": 89.0,
        "2020010101": 89.5,
    }
    result = NASAPowerDataResult(
        data=unsorted,
        product=NASAPowerProduct.SURFACE_PRESSURE,
        location=location,
        start_date=start,
        end_date=end,
    )

    arr = result.to_numpy()
    np.testing.assert_array_equal(arr, np.array([89.0, 89.5, 88.9]))
