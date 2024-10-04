from typing import Dict, List

import pandas as pd
import pvlib

from .location import Location


class IrradianceEstimator:
    def __init__(self, location: Location):
        self.location = location

    def generate_time_range(
        self, start_date: str, end_date: str, freq: str = "1h", **kwargs
    ) -> pd.DatetimeIndex:
        """
        Uses the pandas `date_range` function to generate a range of time points

        Returns the range of equally spaced time points (where the difference between any
        two adjacent points is specified by the given frequency) such that they all
        satisfy `start <[=] x <[=] end`, where the first one and the last one are, resp.,
        the first and last time points in that range that fall on the boundary of ``freq``
        (if given as a frequency string)

        Parameters
        ----------
        start_date : str
            The start date of the range
        end_date : str
            The end date of the range
        freq : str, default '1h'
            The frequency string for the time points
        **kwargs
            Additional keyword arguments to be passed to the `date_range` function

        Returns
        -------
        pd.DatetimeIndex
        """
        return pd.date_range(
            start=start_date, end=end_date, freq=freq, tz=self.location.tz, **kwargs
        )

    # TODO Add more clearsky models as needed after model is implemented
    def estimate_clearsky(
        self, times: pd.DatetimeIndex, model: str = "simplified_solis"
    ) -> pd.DataFrame:
        return self.location.get_clearsky(times, model=model)

    # TODO implement cloud model or use cloud index from satellite data
    def adjust_for_cloud_cover(
        self, clear_sky: pd.DataFrame, cloud_cover_fraction: float
    ) -> pd.Series:
        return clear_sky["ghi"] * (1 - cloud_cover_fraction)

    def get_solar_position(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        return pvlib.solarposition.get_solarposition(
            time=times,
            latitude=self.location.latitude,
            longitude=self.location.longitude,
            altitude=self.location.altitude,
        )

    def decompose_irradiance(
        self,
        ghi: pd.Series,
        solar_zenith: pd.Series,
        times: pd.DatetimeIndex,
        model: str = "erbs",
    ) -> pd.DataFrame:
        if model == "erbs":
            data = pvlib.irradiance.erbs(
                ghi=ghi, zenith=solar_zenith, datetime_or_doy=times
            )

        return data
        # TODO Add more decomposition models here as needed

    def calculate_poa_irradiance(
        self,
        surface_tilts: List[float],
        surface_azimuth: float,
        dni: pd.Series,
        dhi: pd.Series,
        ghi: pd.Series,
        solar_zenith: pd.Series,
        solar_azimuth: pd.Series,
    ) -> Dict[float, pd.Series]:
        poa_irradiance_tilts = {}
        for surface_tilt in surface_tilts:
            poa_irradiance = pvlib.irradiance.get_total_irradiance(
                surface_tilt=surface_tilt,
                surface_azimuth=surface_azimuth,
                dni=dni,
                dhi=dhi,
                ghi=ghi,
                solar_zenith=solar_zenith,
                solar_azimuth=solar_azimuth,
            )
            poa_irradiance_tilts[surface_tilt] = poa_irradiance["poa_global"]
        return poa_irradiance_tilts
