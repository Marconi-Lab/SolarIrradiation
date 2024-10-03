from typing import Dict, List

import pandas as pd

from .irradiance_estimator import IrradianceEstimator


class IrradiancePipeline:
    def __init__(self, estimator: IrradianceEstimator):
        self.estimator = estimator

    def run(
        self,
        start_date: str,
        end_date: str,
        cloud_cover_fraction: float,
        surface_tilts: List[float],
        surface_azimuth: float,
    ) -> pd.DataFrame:
        times = self.estimator.generate_time_range(start_date, end_date)
        clear_sky = self.estimator.estimate_clearsky(times)
        adjusted_ghi = self.estimator.adjust_for_cloud_cover(
            clear_sky, cloud_cover_fraction
        )

        solar_position = self.estimator.get_solar_position(times)
        solar_zenith = solar_position["zenith"]
        solar_azimuth = solar_position["azimuth"]

        irrad_data = self.estimator.decompose_irradiance(
            adjusted_ghi, solar_zenith, times
        )

        poa_irradiance_tilts = self.estimator.calculate_poa_irradiance(
            surface_tilts,
            surface_azimuth,
            irrad_data["dni"],
            irrad_data["dhi"],
            adjusted_ghi,
            solar_zenith,
            solar_azimuth,
        )

        poa_df = pd.DataFrame(poa_irradiance_tilts)
        poa_df.index = times
        return poa_df
