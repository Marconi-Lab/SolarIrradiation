from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import the classes to be tested
from susse.estimators.clearsky import ClearSkyEstimate, ClearSkyEstimatorPVlib


def test_from_np_valid_input():
    """Test creating ClearSkyEstimate from valid numpy arrays."""
    ghi = np.array([100, 200, 300])
    dhi = np.array([50, 100, 150])
    dni = np.array([75, 150, 225])
    timestamps = [
        datetime(2021, 1, 1, 12, 0),
        datetime(2021, 1, 1, 13, 0),
        datetime(2021, 1, 1, 14, 0),
    ]
    estimate = ClearSkyEstimate.from_np(ghi, dhi, dni, timestamps)
    assert isinstance(estimate, ClearSkyEstimate)
    np.testing.assert_array_equal(estimate.ghi, ghi)
    np.testing.assert_array_equal(estimate.dhi, dhi)
    np.testing.assert_array_equal(estimate.dni, dni)
    assert estimate.timestamp == timestamps


def test_from_np_mismatched_lengths():
    """Test creating ClearSkyEstimate with mismatched array lengths."""
    ghi = np.array([100, 200])
    dhi = np.array([50])
    dni = np.array([75, 150])
    timestamps = [datetime(2021, 1, 1, 12, 0)]
    with pytest.raises(ValueError):
        ClearSkyEstimate.from_np(ghi, dhi, dni, timestamps)


def test_clear_sky_estimate_properties():
    """Test the property methods of ClearSkyEstimate."""
    ghi = np.array([100])
    dhi = np.array([50])
    dni = np.array([75])
    timestamps = [datetime(2021, 1, 1, 12, 0)]
    estimate = ClearSkyEstimate.from_np(ghi, dhi, dni, timestamps)
    assert estimate.ghi.tolist() == [100]
    assert estimate.dhi.tolist() == [50]
    assert estimate.dni.tolist() == [75]
    assert estimate.timestamp == timestamps


def test_estimate_clear_sky_valid_input():
    """Test estimate_clear_sky with valid input."""
    from geopy import Nominatim

    with (
        patch("pvlib.location.Location") as mock_Location,
        patch("pvlib.location.lookup_altitude") as mock_lookup_altitude,
    ):
        # Mock the altitude lookup
        mock_lookup_altitude.return_value = 100
        mock_pv_location = MagicMock()
        mock_Location.return_value = mock_pv_location

        # Mock the get_clearsky method
        mock_clear_sky_df = pd.DataFrame(
            {
                "ghi": [100, 200],
                "dhi": [50, 100],
                "dni": [75, 150],
            },
            index=pd.date_range(
                start=datetime(2021, 1, 1, 12, 0, tzinfo=timezone.utc),
                periods=2,
                freq="h",
            ),
        )
        mock_pv_location.get_clearsky.return_value = mock_clear_sky_df

        # Create an instance of the estimator
        estimator = ClearSkyEstimatorPVlib()

        # Mock location
        geolocator = Nominatim(user_agent="SuSSe")
        location = geolocator.geocode("Kampala")

        # Estimate clear sky
        start_date = datetime(2021, 1, 1, 12, 0, tzinfo=timezone.utc)
        end_date = datetime(2021, 1, 1, 13, 0, tzinfo=timezone.utc)
        estimate = estimator.estimate_clear_sky(location, start_date, end_date)

        # Assertions
        assert isinstance(estimate, ClearSkyEstimate)
        np.testing.assert_array_equal(estimate.ghi, [100, 200])
        np.testing.assert_array_equal(estimate.dhi, [50, 100])
        np.testing.assert_array_equal(estimate.dni, [75, 150])


def test_estimate_clear_sky_invalid_dates():
    """Test estimate_clear_sky with end_date before start_date."""
    from geopy import Nominatim

    estimator = ClearSkyEstimatorPVlib()
    geolocator = Nominatim(user_agent="SuSSe")
    location = geolocator.geocode("Kampala")
    start_date = datetime(2021, 1, 2, 12, 0, tzinfo=timezone.utc)
    end_date = datetime(2021, 1, 1, 12, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        estimator.estimate_clear_sky(location, start_date, end_date)


def test_set_timedelta():
    """Test setting a custom timedelta."""
    estimator = ClearSkyEstimatorPVlib()
    custom_timedelta = timedelta(minutes=30)
    estimator.set_timedelta(custom_timedelta)
    assert estimator._timedelta == custom_timedelta


def test_set_timezone():
    """Test setting a custom timezone."""
    estimator = ClearSkyEstimatorPVlib()
    custom_timezone = timezone(timedelta(hours=-5))
    estimator.set_timezone(custom_timezone)
    assert estimator._tz == custom_timezone
