from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pvlib
from farms.utilities import ti_to_radius
from geopy.location import Location as GLocation

from ...api_clients.NASA_Power import (
    NASAPowerFetchData,
    NASAPowerProduct,
    TemporalResolution,
)
from .rest2 import rest2


class REST2Model:
    """
    Compute clear-sky irradiance using the REST2 model.
    """

    def __init__(self) -> None:
        self.fetcher = NASAPowerFetchData()
        self.solar_constant = 1361.2
        self.sza_limit = 89.0

    def compute_irradiance(
        self,
        start_date: datetime,
        end_date: datetime,
        location: GLocation,
        resolution: TemporalResolution,
    ) -> np.ndarray:
        """
        Fetch parameters and run REST2 clear-sky irradiance model.

        Args:
            start_date: Start datetime (UTC).
            end_date: End datetime (UTC).
            location: Geopy Location with latitude and longitude.
            resolution: Temporal resolution (HOURLY, DAILY, MONTHLY).

        Returns:
            Array of clear-sky irradiance values.
        """
        params_df = self.fetcher.fetch_multiple_parameters(
            start_date,
            end_date,
            location,
            [
                NASAPowerProduct.SURFACE_PRESSURE,
                NASAPowerProduct.AEROSOL_OPTICAL_DEPTH_550nm,
                NASAPowerProduct.AEROSOL_OPTICAL_DEPTH_840nm,
                NASAPowerProduct.PRECIPITABLE_WATER,
                NASAPowerProduct.ALL_SKY_SURFACE_ALBEDO,
                NASAPowerProduct.TOTAL_COLUMN_OZONE,
            ],
            temporal_resolution=resolution,
        )

        timestamps = self._generate_timestamps(start_date, end_date, resolution)
        radius = self._calculate_radius_factors(timestamps)

        p = params_df.to_numpy(NASAPowerProduct.SURFACE_PRESSURE)
        aod550 = params_df.to_numpy(NASAPowerProduct.AEROSOL_OPTICAL_DEPTH_550nm)
        aod840 = params_df.to_numpy(NASAPowerProduct.AEROSOL_OPTICAL_DEPTH_840nm)
        pw = params_df.to_numpy(NASAPowerProduct.PRECIPITABLE_WATER)
        albedo = params_df.to_numpy(NASAPowerProduct.ALL_SKY_SURFACE_ALBEDO)
        ozone = params_df.to_numpy(NASAPowerProduct.TOTAL_COLUMN_OZONE) * 0.001
        z = self._get_solar_zenith_angle(timestamps, location)

        alpha, beta = self._calculate_angstrom_params(aod550, aod840)

        rest2_inputs = {
            "p": p * 10,
            "albedo": albedo,
            "ssa": -9.99,
            "g": -9.99,
            "z": z,
            "radius": radius,
            "alpha": alpha,
            "beta": beta,
            "ozone": ozone,
            "w": pw,
            "sza_lim": self.sza_limit,
        }

        return rest2(**rest2_inputs)

    def _generate_timestamps(
        self,
        start_date: datetime,
        end_date: datetime,
        resolution: TemporalResolution,
    ) -> pd.DatetimeIndex:
        """
        Generate timestamps matching NASA API's exclusive end behavior.

        Returns:
            DatetimeIndex of timestamps.
        """
        if resolution is TemporalResolution.HOURLY:
            end_date += timedelta(hours=1)
        elif resolution is TemporalResolution.DAILY:
            end_date += timedelta(days=1)

        freq_map = {
            TemporalResolution.HOURLY: "H",
            TemporalResolution.DAILY: "D",
            TemporalResolution.MONTHLY: "MS",
        }

        return pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq_map[resolution],
            inclusive="left",
        )

    @staticmethod
    def _calculate_radius_factors(
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Vectorized Earth–Sun distance calculation.
        """
        return ti_to_radius(timestamps)

    @staticmethod
    def _calculate_angstrom_params(
        aod550: np.ndarray,
        aod840: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Angstrom exponent (alpha) and turbidity (beta).
        """
        aod550 = np.asarray(aod550)
        aod840 = np.asarray(aod840)

        valid = (aod550 > 0) & (aod840 > 0)

        alpha = np.full_like(aod550, np.nan)
        beta = np.full_like(aod550, np.nan)

        alpha[valid] = -np.log(aod550[valid] / aod840[valid]) / np.log(550 / 840)
        beta[valid] = aod550[valid] * (0.55 ** alpha[valid])

        return alpha, beta

    def _get_solar_zenith_angle(
        self,
        timestamps: pd.DatetimeIndex | list[str],
        location: GLocation,
    ) -> np.ndarray:
        """
        Calculate solar zenith angle for given times and location.
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps, format="%Y%m%d%H")

        solpos = pvlib.solarposition.get_solarposition(
            timestamps,
            location.latitude,
            location.longitude,
        )
        return solpos["zenith"].to_numpy()
