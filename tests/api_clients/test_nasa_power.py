from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import requests
from geopy import Point

from susse.api_clients import (
    NASAPowerConfig,
    NASAPowerFetchData,
    NASAPowerProduct,
    NASAPowerResult,
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


def test_fetch_single_parameter_success(requests_mock, fetcher, dates, location):
    start, end = dates
    product = NASAPowerProduct.SURFACE_PRESSURE

    # Prepare mock URL and JSON response
    url = NASAPowerConfig.generate_download_link(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        [product],
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

    result = fetcher.fetch_multiple_parameters(
        start,
        end,
        location,
        product,
        TemporalResolution.HOURLY,
    )

    assert isinstance(result, NASAPowerResult)
    assert len(result._products) == 1
    assert result._products[0] is product
    assert len(result._raw_data[product.value]) == 3


def test_fetch_multiple_parameters_success(requests_mock, fetcher, dates, location):
    start, end = dates
    products = [
        NASAPowerProduct.SURFACE_PRESSURE,
        NASAPowerProduct.TEMPERATURE,
    ]

    # Prepare mock URL and JSON response
    url = NASAPowerConfig.generate_download_link(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        products,
    )
    mock_data = {
        "properties": {
            "parameter": {
                "PS": {
                    "2020010100": 89.0,
                    "2020010101": 89.0,
                    "2020010102": 88.99,
                },
                "T2M": {
                    "2020010100": 25.0,
                    "2020010101": 26.0,
                    "2020010102": 27.0,
                },
            }
        }
    }
    requests_mock.get(url, json=mock_data)

    result = fetcher.fetch_multiple_parameters(
        start,
        end,
        location,
        products,
        TemporalResolution.HOURLY,
    )

    assert isinstance(result, NASAPowerResult)
    assert len(result._raw_data) == 2
    assert all(product.value in result._raw_data for product in products)


def test_fetch_parameters_batch_limit(requests_mock, fetcher, dates, location):
    start, end = dates
    # Create more than 20 products to test batching
    products = list(NASAPowerProduct)[:40]  # Take first 40 products

    # Prepare mock responses for batches
    mock_data_batch1 = {
        "properties": {
            "parameter": {p.value: {"2020010100": 1.0} for p in products[:20]}
        }
    }
    mock_data_batch2 = {
        "properties": {
            "parameter": {p.value: {"2020010100": 1.0} for p in products[20:]}
        }
    }

    # Mock both batch requests
    url1 = NASAPowerConfig.generate_download_link(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        products[:20],
    )
    url2 = NASAPowerConfig.generate_download_link(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        products[20:],
    )
    requests_mock.get(url1, json=mock_data_batch1)
    requests_mock.get(url2, json=mock_data_batch2)

    result = fetcher.fetch_multiple_parameters(
        start,
        end,
        location,
        products,
        TemporalResolution.HOURLY,
    )

    assert isinstance(result, NASAPowerResult)
    assert len(result._raw_data) == len(products)
    assert all(product.value in result._raw_data for product in products)


def test_fetch_parameters_http_error(requests_mock, fetcher, dates, location):
    start, end = dates
    products = [NASAPowerProduct.SURFACE_PRESSURE]

    url = NASAPowerConfig.generate_download_link(
        TemporalResolution.HOURLY,
        start,
        end,
        location,
        products,
    )
    requests_mock.get(url, exc=requests.exceptions.RequestException)

    with pytest.raises(requests.exceptions.RequestException):
        fetcher.fetch_multiple_parameters(
            start,
            end,
            location,
            products,
            TemporalResolution.HOURLY,
        )


@pytest.fixture
def sample_result(dates, location) -> NASAPowerResult:
    start, end = dates
    data = {
        "PS": {
            "2020010100": 89.0,
            "2020010101": 89.5,
            "2020010102": 88.9,
        }
    }
    return NASAPowerResult(
        data=data,
        products=[NASAPowerProduct.SURFACE_PRESSURE],
        location=location,
        start_date=start,
        end_date=end,
    )


def test_to_numpy_conversion(sample_result):
    arr = sample_result.to_numpy(NASAPowerProduct.SURFACE_PRESSURE)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    np.testing.assert_array_equal(arr, np.array([89.0, 89.5, 88.9]))


def test_timestamp_ordering(location, dates):
    start, end = dates
    unsorted = {
        "PS": {
            "2020010102": 88.9,
            "2020010100": 89.0,
            "2020010101": 89.5,
        }
    }
    result = NASAPowerResult(
        data=unsorted,
        products=[NASAPowerProduct.SURFACE_PRESSURE],
        location=location,
        start_date=start,
        end_date=end,
    )

    arr = result.to_numpy(NASAPowerProduct.SURFACE_PRESSURE)
    np.testing.assert_array_equal(arr, np.array([89.0, 89.5, 88.9]))


def test_to_dataframe_conversion(sample_result):
    df = sample_result.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Three timestamps
    assert list(df.columns) == [
        "SURFACE_PRESSURE"
    ]  # Column name should be product name
    np.testing.assert_array_equal(
        df["SURFACE_PRESSURE"].values, np.array([89.0, 89.5, 88.9])
    )


def test_to_numpy_all_parameters(location, dates):
    start, end = dates
    data = {
        "PS": {
            "2020010100": 89.0,
            "2020010101": 89.5,
        },
        "T2M": {
            "2020010100": 25.0,
            "2020010101": 26.0,
        },
    }
    result = NASAPowerResult(
        data=data,
        products=[NASAPowerProduct.SURFACE_PRESSURE, NASAPowerProduct.TEMPERATURE],
        location=location,
        start_date=start,
        end_date=end,
    )

    # Test converting all parameters to 2D numpy array
    arr = result.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)  # 2 timestamps × 2 parameters
    np.testing.assert_array_equal(arr, np.array([[89.0, 25.0], [89.5, 26.0]]))


def test_get_parameter_data(sample_result):
    # Test getting existing parameter
    data = sample_result.get_parameter_data(NASAPowerProduct.SURFACE_PRESSURE)
    assert data is not None
    assert len(data) == 3
    assert data["2020010100"] == 89.0

    # Test getting non-existent parameter
    data = sample_result.get_parameter_data(NASAPowerProduct.TEMPERATURE)
    assert data is None


def test_result_properties(sample_result, location, dates):
    start, end = dates

    # Test products property
    assert len(sample_result.products) == 1
    assert sample_result.products[0] == NASAPowerProduct.SURFACE_PRESSURE

    # Test location property
    assert sample_result.location == location

    # Test date properties
    assert sample_result.start_date == start
    assert sample_result.end_date == end


def test_invalid_product_to_numpy(sample_result):
    with pytest.raises(ValueError, match="Product TEMPERATURE not found in result"):
        sample_result.to_numpy(NASAPowerProduct.TEMPERATURE)
