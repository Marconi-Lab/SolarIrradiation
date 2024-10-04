from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .irradiance_estimator import IrradianceEstimator


class IrradiancePipeline:
    def __init__(self, estimator: IrradianceEstimator):
        self.estimator = estimator
        self.dni_name = "dni"
        self.dhi_name = "dhi"
        self.ghi_name = "ghi"
        self.poa_name = "poa_irradiance"
        self.zenith_name = "zenith"
        self.azimuth_name = "azimuth"

        # Initialize attributes to store results with appropriate default values
        self.times: Optional[pd.DatetimeIndex] = None
        self.clear_sky: Optional[pd.DataFrame] = None
        self.adjusted_ghi: Optional[np.ndarray] = np.array([])
        self.solar_position: Optional[pd.DataFrame] = None
        self.dni: Optional[np.ndarray] = np.array([])
        self.dhi: Optional[np.ndarray] = np.array([])
        self.poa_irradiance: Optional[Dict[float, np.ndarray]] = {}
        self.solar_zenith: Optional[np.ndarray] = np.array([])
        self.solar_azimuth: Optional[np.ndarray] = np.array([])

    def generate_time_range(
        self, start_date: str, end_date: str, freq: str = "1h"
    ) -> None:
        self.times = self.estimator.generate_time_range(start_date, end_date, freq)

    def estimate_clear_sky_irradiance(self) -> None:
        self.clear_sky = self.estimator.estimate_clearsky(self.times)

    def adjust_irradiance_for_cloud_cover(self, cloud_cover_fraction: float) -> None:
        self.adjusted_ghi = self.estimator.adjust_for_cloud_cover(
            self.clear_sky, cloud_cover_fraction
        )

    def calculate_solar_position(self) -> None:
        self.solar_position = self.estimator.get_solar_position(self.times)

    def decompose_irradiance(self) -> None:
        self.dni, self.dhi = (
            self.estimator.decompose_irradiance(
                self.adjusted_ghi,
                self.solar_position[self.zenith_name].values,
                self.times,
            )[self.dni_name],
            self.estimator.decompose_irradiance(
                self.adjusted_ghi,
                self.solar_position[self.zenith_name].values,
                self.times,
            )[self.dhi_name],
        )

    def calculate_poa_irradiance(
        self, surface_tilts: List[float], surface_azimuth: float
    ) -> None:
        self.poa_irradiance = self.estimator.calculate_poa_irradiance(
            surface_tilts,
            surface_azimuth,
            self.dni,
            self.dhi,
            self.adjusted_ghi,
            self.solar_position[self.zenith_name].values,
            self.solar_position[self.azimuth_name].values,
        )

    def run(
        self,
        start_date: str,
        end_date: str,
        cloud_cover_fraction: float,
        surface_tilts: List[float],
        surface_azimuth: float,
    ) -> None:
        self.generate_time_range(start_date, end_date)
        self.estimate_clear_sky_irradiance()
        self.adjust_irradiance_for_cloud_cover(cloud_cover_fraction)
        self.calculate_solar_position()
        self.decompose_irradiance()
        self.calculate_poa_irradiance(surface_tilts, surface_azimuth)

    # Property accessors
    @property
    def times(self) -> pd.DatetimeIndex:
        return self._times

    @times.setter
    def times(self, value: pd.DatetimeIndex):
        self._times = value

    @property
    def clear_sky(self) -> pd.DataFrame:
        return self._clear_sky

    @clear_sky.setter
    def clear_sky(self, value: pd.DataFrame):
        self._clear_sky = value

    @property
    def adjusted_ghi(self) -> np.ndarray:
        return self._adjusted_ghi

    @adjusted_ghi.setter
    def adjusted_ghi(self, value: np.ndarray):
        self._adjusted_ghi = value

    @property
    def solar_position(self) -> pd.DataFrame:
        return self._solar_position

    @solar_position.setter
    def solar_position(self, value: pd.DataFrame):
        self._solar_position = value

    @property
    def dni(self) -> np.ndarray:
        return self._dni

    @dni.setter
    def dni(self, value: np.ndarray):
        self._dni = value

    @property
    def dhi(self) -> np.ndarray:
        return self._dhi

    @dhi.setter
    def dhi(self, value: np.ndarray):
        self._dhi = value

    @property
    def poa_irradiance(self) -> Dict[float, np.ndarray]:
        return self._poa_irradiance

    @poa_irradiance.setter
    def poa_irradiance(self, value: Dict[float, np.ndarray]):
        self._poa_irradiance = value

    @property
    def solar_zenith(self) -> np.ndarray:
        return self._solar_zenith

    @solar_zenith.setter
    def solar_zenith(self, value: np.ndarray):
        self._solar_zenith = value

    @property
    def solar_azimuth(self) -> np.ndarray:
        return self._solar_azimuth

    @solar_azimuth.setter
    def solar_azimuth(self, value: np.ndarray):
        self._solar_azimuth = value
