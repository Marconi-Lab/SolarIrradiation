import unittest
from unittest.mock import patch, Mock
from datetime import datetime
import numpy as np
from geopy import Point
from nasa_power_config import NASAPowerConfig
from nasa_power_fetch_data import NASAPowerFetchData
from nasa_power_result import NASAPowerDataResult
from nasa_products import NASAPowerProducts, TemporalResolution

class TestNASAPowerConfig(unittest.TestCase):
    def test_url_generation(self):
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 3)
        location = Point(latitude=33.6, longitude=1.3)
        
        url = NASAPowerConfig.generate_download_link(
            TemporalResolution.HOURLY,
            start_date,
            end_date,
            location,
            NASAPowerProducts.SURFACE_PRESSURE
        )
        
        expected = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point"
            "?parameters=PS&community=RE"
            "&longitude=1.3&latitude=33.6"
            "&start=20200101&end=20200103&format=CSV"
        )
        self.assertEqual(url, expected)

class TestNASAPowerFetchData(unittest.TestCase):
    def setUp(self):
        self.fetch = NASAPowerFetchData()
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2020, 1, 3)
        self.location = Point(latitude=33.6, longitude=1.3)
        self.mock_response = {
            "properties": {
                "parameter": {
                    "PS": {
                        "2020010100": 89.0,
                        "2020010101": 89.0,
                        "2020010102": 88.99
                    }
                }
            }
        }

    @patch('requests.get')
    def test_fetch_data_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = self.mock_response
        mock_get.return_value = mock_response

        result = self.fetch.fetch_data(
            TemporalResolution.HOURLY,
            self.start_date,
            self.end_date,
            self.location,
            NASAPowerProducts.SURFACE_PRESSURE
        )

        self.assertIsInstance(result, NASAPowerDataResult)
        self.assertEqual(result._product, NASAPowerProducts.SURFACE_PRESSURE)
        self.assertEqual(len(result._raw_data), 3)

    @patch('requests.get')
    def test_fetch_data_http_error(self, mock_get):
        mock_get.side_effect = Exception("HTTP Error")
        
        with self.assertRaises(Exception):
            self.fetch.fetch_data(
                TemporalResolution.HOURLY,
                self.start_date,
                self.end_date,
                self.location,
                NASAPowerProducts.SURFACE_PRESSURE
            )

class TestNASAPowerDataResult(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            "2020010100": 89.0,
            "2020010101": 89.5,
            "2020010102": 88.9
        }
        self.result = NASAPowerDataResult(
            data=self.sample_data,
            product=NASAPowerProducts.SURFACE_PRESSURE,
            location=Point(1.3, 33.6),
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 3)
        )

    def test_to_numpy_conversion(self):
        array = self.result.to_numpy()
        
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, (3,))
        np.testing.assert_array_equal(
            array,
            np.array([89.0, 89.5, 88.9])
        )

    def test_timestamp_ordering(self):
        unsorted_data = {
            "2020010102": 88.9,
            "2020010100": 89.0,
            "2020010101": 89.5
        }
        result = NASAPowerDataResult(
            unsorted_data,
            NASAPowerProducts.SURFACE_PRESSURE,
            Point(1.3, 33.6),
            datetime(2020, 1, 1),
            datetime(2020, 1, 3)
        )
        
        array = result.to_numpy()
        np.testing.assert_array_equal(
            array,
            np.array([89.0, 89.5, 88.9])
        )
