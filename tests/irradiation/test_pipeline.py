import numpy as np
import pandas as pd
import pytest

from susse.irradiation import IrradianceEstimator, IrradiancePipeline, Location


@pytest.fixture
def kampala_location():
    return Location(latitude=0.347594, longitude=32.58210, timezone="Africa/Kampala")


@pytest.fixture
def estimator(kampala_location):
    return IrradianceEstimator(kampala_location)


@pytest.fixture
def pipeline(estimator):
    return IrradiancePipeline(estimator)


def test_pipeline_initialization(pipeline):
    assert isinstance(pipeline.estimator, IrradianceEstimator)
    assert pipeline.dni_name == "dni"
    assert pipeline.dhi_name == "dhi"
    assert pipeline.ghi_name == "ghi"
    assert pipeline.poa_name == "poa_irradiance"
    assert pipeline.zenith_name == "zenith"
    assert pipeline.azimuth_name == "azimuth"
    assert pipeline.clearness_index_name == "kt"


def test_pipeline_run(pipeline):
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    cloud_cover_fraction = 0.2
    surface_tilts = [0, 20]
    surface_azimuth = 180

    pipeline.run(
        start_date, end_date, cloud_cover_fraction, surface_tilts, surface_azimuth
    )

    assert isinstance(pipeline.times, pd.DatetimeIndex)
    assert isinstance(pipeline.clear_sky, pd.DataFrame)
    assert isinstance(pipeline.adjusted_ghi, np.ndarray)
    assert isinstance(pipeline.solar_position, pd.DataFrame)
    assert isinstance(pipeline.dni, np.ndarray)
    assert isinstance(pipeline.dhi, np.ndarray)
    assert isinstance(pipeline.clearness_index, np.ndarray)
    assert isinstance(pipeline.poa_irradiance, dict)

    assert len(pipeline.times) == 49  # 2 days + 1 hour (inclusive of end date)
    assert all(tilt in pipeline.poa_irradiance for tilt in surface_tilts)


def test_pipeline_property_accessors(pipeline):
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    cloud_cover_fraction = 0.2
    surface_tilts = [0]
    surface_azimuth = 180

    pipeline.run(
        start_date, end_date, cloud_cover_fraction, surface_tilts, surface_azimuth
    )

    assert pipeline.times is not None
    assert pipeline.clear_sky is not None
    assert pipeline.adjusted_ghi is not None
    assert pipeline.solar_position is not None
    assert pipeline.dni is not None
    assert pipeline.dhi is not None
    assert pipeline.clearness_index is not None
    assert pipeline.poa_irradiance is not None


def test_pipeline_individual_steps(pipeline):
    start_date = "2023-01-01"
    end_date = "2023-01-03"
    cloud_cover_fraction = 0.2
    surface_tilts = [0]
    surface_azimuth = 180

    pipeline.generate_time_range(start_date, end_date)
    assert isinstance(pipeline.times, pd.DatetimeIndex)

    pipeline.estimate_clear_sky_irradiance()
    assert isinstance(pipeline.clear_sky, pd.DataFrame)

    pipeline.adjust_irradiance_for_cloud_cover(cloud_cover_fraction)
    assert isinstance(pipeline.adjusted_ghi, np.ndarray)

    pipeline.calculate_solar_position()
    assert isinstance(pipeline.solar_position, pd.DataFrame)

    pipeline.decompose_irradiance()
    assert isinstance(pipeline.dni, np.ndarray)
    assert isinstance(pipeline.dhi, np.ndarray)

    pipeline.calculate_poa_irradiance(surface_tilts, surface_azimuth)
    assert isinstance(pipeline.poa_irradiance, dict)


def test_pipeline_with_different_parameters(pipeline):
    start_date = "2023-06-01"
    end_date = "2023-06-02"
    cloud_cover_fraction = 0.5
    surface_tilts = [10, 30, 50]
    surface_azimuth = 225

    pipeline.run(
        start_date, end_date, cloud_cover_fraction, surface_tilts, surface_azimuth
    )

    assert len(pipeline.times) == 25  # 1 day + 1 hour (inclusive of end date)
    assert all(tilt in pipeline.poa_irradiance for tilt in surface_tilts)
    assert pipeline.poa_irradiance[10].shape == pipeline.dni.shape
