from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .irradiance_estimator import IrradianceEstimator


class IrradiancePipeline:
    """
    A class to run the pipeline for estimating solar irradiance
    """

    def __init__(self, estimator: IrradianceEstimator):
        self.estimator = estimator
        self.dni_name = "dni"
        self.dhi_name = "dhi"
        self.ghi_name = "ghi"
        self.poa_name = "poa_irradiance"
        self.zenith_name = "zenith"
        self.azimuth_name = "azimuth"
        self.clearness_index_name = "kt"

        # Initialize attributes to store results with appropriate default values
        self._times: Optional[pd.DatetimeIndex] = None
        self._clear_sky: Optional[pd.DataFrame] = None
        self._adjusted_ghi: Optional[np.ndarray] = np.array([])
        self._solar_position: Optional[pd.DataFrame] = None
        self._dni: Optional[np.ndarray] = np.array([])
        self._dhi: Optional[np.ndarray] = np.array([])
        self._clearness_index: Optional[np.ndarray] = np.array([])
        self._poa_irradiance: Optional[Dict[float, np.ndarray]] = {}

    def generate_time_range(
        self, start_date: str, end_date: str, freq: str = "1h"
    ) -> None:
        self._times = self.estimator.generate_time_range(start_date, end_date, freq)

    def estimate_clear_sky_irradiance(self) -> None:
        self._clear_sky = self.estimator.estimate_clearsky(self._times)

    def adjust_irradiance_for_cloud_cover(self, cloud_cover_fraction: float) -> None:
        self._adjusted_ghi = self.estimator.adjust_for_cloud_cover(
            self._clear_sky, cloud_cover_fraction
        ).values

    def calculate_solar_position(self) -> None:
        self._solar_position = self.estimator.get_solar_position(self._times)

    def decompose_irradiance(self) -> None:
        if self._adjusted_ghi is None:
            raise ValueError(
                "Adjusted GHI is empty. Run adjust_irradiance_for_cloud_cover first."
            )
        if self._solar_position is None:
            raise ValueError(
                "Solar position is empty. Run calculate_solar_position first."
            )
        if self._times is None:
            raise ValueError("Times are empty. Run generate_time_range first.")

        self._dni = self.estimator.decompose_irradiance(
            self._adjusted_ghi,
            self._solar_position[self.zenith_name].values,
            self._times,
        )[self.dni_name].values
        self._dhi = self.estimator.decompose_irradiance(
            self._adjusted_ghi,
            self._solar_position[self.zenith_name].values,
            self._times,
        )[self.dhi_name].values
        self._clearness_index = self.estimator.decompose_irradiance(
            self._adjusted_ghi,
            self._solar_position[self.zenith_name].values,
            self.times,
        )[self.clearness_index_name].values

    def calculate_poa_irradiance(
        self, surface_tilts: List[float], surface_azimuth: float
    ) -> None:
        if self._solar_position is None:
            raise ValueError(
                "Solar position is empty. Run calculate_solar_position first."
            )

        self._poa_irradiance = self.estimator.calculate_poa_irradiance(
            surface_tilts,
            surface_azimuth,
            self._dni,
            self._dhi,
            self._adjusted_ghi,
            self._solar_position[self.zenith_name].values,
            self._solar_position[self.azimuth_name].values,
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

    @property
    def clear_sky(self) -> pd.DataFrame:
        return self._clear_sky

    @property
    def adjusted_ghi(self) -> np.ndarray:
        if self._adjusted_ghi is not None:
            return self._adjusted_ghi
        else:
            raise ValueError(
                "Adjusted GHI is empty. Run adjust_irradiance_for_cloud_cover first."
            )

    @property
    def solar_position(self) -> pd.DataFrame:
        return self._solar_position

    @property
    def dni(self) -> np.ndarray:
        if self._dni is not None:
            return self._dni
        else:
            raise ValueError("DNI is empty. Run decompose_irradiance first.")

    @property
    def dhi(self) -> np.ndarray:
        if self._dhi is not None:
            return self._dhi
        else:
            raise ValueError("DHI is empty. Run decompose_irradiance first.")

    @property
    def clearness_index(self) -> np.ndarray:
        if self._clearness_index is not None:
            return self._clearness_index
        else:
            raise ValueError(
                "Clearness index is empty. Run decompose_irradiance first."
            )

    @property
    def poa_irradiance(self) -> Dict[float, np.ndarray]:
        if self._poa_irradiance:
            return self._poa_irradiance
        else:
            raise ValueError(
                "POA irradiance is empty. Run calculate_poa_irradiance first."
            )
