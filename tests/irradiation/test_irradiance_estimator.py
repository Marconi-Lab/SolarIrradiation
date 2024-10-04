import pandas as pd
import pytest

from susse.irradiation import IrradianceEstimator, Location


@pytest.fixture
def kampala_location():
    return Location(latitude=0.347594, longitude=32.58210, timezone="Africa/Kampala")


@pytest.fixture
def estimator(kampala_location):
    return IrradianceEstimator(kampala_location)


def test_generate_time_range(estimator):
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    time_range = estimator.generate_time_range(start_date, end_date)

    assert isinstance(time_range, pd.DatetimeIndex)
    assert time_range[0] == pd.Timestamp("2023-01-01 00:00:00+03:00")
    assert time_range[-1] == pd.Timestamp("2023-01-03 00:00:00+03:00")
    assert len(time_range) == 49  # 2 days * 24 hours and 1 hour for the end date


def test_estimate_clearsky(estimator):
    times = pd.date_range("2023-01-01", "2023-01-02", freq="1h", tz="Africa/Kampala")
    clearsky = estimator.estimate_clearsky(times)

    assert isinstance(clearsky, pd.DataFrame)
    assert "ghi" in clearsky.columns
    assert "dni" in clearsky.columns
    assert "dhi" in clearsky.columns
    assert len(clearsky) == len(times)


def test_adjust_for_cloud_cover(estimator):
    times = pd.date_range("2023-01-01", "2023-01-02", freq="1h", tz="Africa/Kampala")
    clearsky = estimator.estimate_clearsky(times)
    cloud_cover = 0.2

    adjusted_ghi = estimator.adjust_for_cloud_cover(clearsky, cloud_cover)

    assert isinstance(adjusted_ghi, pd.Series)
    assert len(adjusted_ghi) == len(clearsky)
    assert all(adjusted_ghi <= clearsky["ghi"])


def test_get_solar_position(estimator):
    times = pd.date_range("2023-01-01", "2023-01-02", freq="1h", tz="Africa/Kampala")
    solar_position = estimator.get_solar_position(times)

    assert isinstance(solar_position, pd.DataFrame)
    assert "zenith" in solar_position.columns
    assert "azimuth" in solar_position.columns
    assert len(solar_position) == len(times)


def test_decompose_irradiance(estimator):
    times = pd.date_range("2023-01-01", "2023-01-02", freq="1h", tz="Africa/Kampala")
    clearsky = estimator.estimate_clearsky(times)
    solar_position = estimator.get_solar_position(times)

    decomposed = estimator.decompose_irradiance(
        clearsky["ghi"], solar_position["zenith"], times
    )

    assert isinstance(decomposed, pd.DataFrame)
    assert "dni" in decomposed.columns
    assert "dhi" in decomposed.columns
    assert len(decomposed) == len(times)


def test_calculate_poa_irradiance(estimator):
    times = pd.date_range("2023-01-01", "2023-01-02", freq="1h", tz="Africa/Kampala")
    clearsky = estimator.estimate_clearsky(times)
    solar_position = estimator.get_solar_position(times)
    decomposed = estimator.decompose_irradiance(
        clearsky["ghi"], solar_position["zenith"], times
    )

    surface_tilts = [0, 20, 40]
    surface_azimuth = 180

    poa_irradiance = estimator.calculate_poa_irradiance(
        surface_tilts,
        surface_azimuth,
        decomposed["dni"],
        decomposed["dhi"],
        clearsky["ghi"],
        solar_position["zenith"],
        solar_position["azimuth"],
    )

    assert isinstance(poa_irradiance, dict)
    assert all(tilt in poa_irradiance for tilt in surface_tilts)
    assert all(isinstance(poa_irradiance[tilt], pd.Series) for tilt in surface_tilts)
    assert all(len(poa_irradiance[tilt]) == len(times) for tilt in surface_tilts)
