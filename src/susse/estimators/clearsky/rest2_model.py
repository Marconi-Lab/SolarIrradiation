import numpy as np 
import pandas as pd
import pvlib
from farms.utilities import ti_to_radius
from ...api_clients.NASA_Power import NASAPowerProducts, TemporalResolution, NASAPowerFetchData
from datetime import datetime, timedelta
from geopy import location as Glocation

from .rest2 import rest2


class REST2Model:
    def __init__(self):
        self.fetcher = NASAPowerFetchData()
        self.solar_constant = 1361.2
        self.sza_limit = 89.0
        
    def compute_irradiance(self, start_date: datetime, end_date: datetime, 
                             location: Glocation, resolution: TemporalResolution):
        """
        Compute clear-sky irradiance using REST2 model for any temporal resolution.
        """
        # Fetch all required parameters in one request
        required_params = [
            NASAPowerProducts.SURFACE_PRESSURE,
            NASAPowerProducts.AEROSOL_OPTICAL_DEPTH_550nm,
            NASAPowerProducts.AEROSOL_OPTICAL_DEPTH_840nm,
            NASAPowerProducts.PRECIPITABLE_WATER,
            NASAPowerProducts.ALL_SKY_SURFACE_ALBEDO,
            NASAPowerProducts.TOTAL_COLUMN_OZONE
        ]
        
        data = self.fetcher.fetch_multiple_parameters(
            resolution,
            start_date,
            end_date,
            location,
            required_params
        )
        
        # Generate appropriate timestamps and radius factors
        timestamps = self._generate_timestamps(start_date, end_date, resolution)
        #print(timestamps)
        radius_factors = self._calculate_radius_factors(timestamps)
        
        # Prepare parameters dictionary
        params = {
            'p': data.to_numpy(NASAPowerProducts.SURFACE_PRESSURE),
            'aod550': data.to_numpy(NASAPowerProducts.AEROSOL_OPTICAL_DEPTH_550nm),
            'aod840': data.to_numpy(NASAPowerProducts.AEROSOL_OPTICAL_DEPTH_840nm),
            'pw': data.to_numpy(NASAPowerProducts.PRECIPITABLE_WATER),
            'albedo': data.to_numpy(NASAPowerProducts.ALL_SKY_SURFACE_ALBEDO),
            'z': self._get_solar_zenith_angle(timestamps, location),
            'ozone': data.to_numpy(NASAPowerProducts.TOTAL_COLUMN_OZONE) * 0.001,
            'radius': radius_factors
        }

        # Calculate Angstrom parameters
        alpha, beta = self._calculate_angstrom_params(params['aod550'], params['aod840'])
 
        # Prepare and execute REST2 inputs
        rest2_args = {
            'p': params['p'] * 10, 
            'albedo': params['albedo'],
            'ssa':-9.99,
            'g': -9.99,
            'z': params['z'],
            'radius': params['radius'],
            'alpha': alpha,
            'beta': beta,
            'ozone': params['ozone'],
            'w': params['pw'],
            'sza_lim': self.sza_limit
        }
        #print(rest2_args)
        return rest2(**rest2_args)

   
    def _generate_timestamps(self, start_date, end_date, resolution):
        """Generate timestamps matching NASA API's exclusive end behavior"""
        if resolution == TemporalResolution.HOURLY:
            # Add 1 hour to include final hour
            end_date += timedelta(days=1)
        elif resolution == TemporalResolution.DAILY:
            # Add 1 day to include final day
            end_date += timedelta(days=1)
        
        freq_map = {
            TemporalResolution.HOURLY: 'h',
            TemporalResolution.DAILY: 'D',
            TemporalResolution.MONTHLY: 'MS'
        }
        return pd.date_range(start_date, end_date, freq=freq_map[resolution], inclusive='left')  



    def _calculate_radius_factors(self, timestamps):
        """Vectorized Earth-Sun distance calculation."""
        return ti_to_radius(timestamps)
    
    
    def _calculate_angstrom_params(self, aod550, aod840):
        """Calculate Angstrom exponent (alpha) and turbidity (beta)."""
        aod550 = np.array(aod550)
        aod840 = np.array(aod840)

        valid = (aod550 > 0) & (aod840 > 0)

        alpha = np.full(aod550.shape, np.nan)
        beta = np.full(aod550.shape, np.nan)

        alpha[valid] = -np.log(aod550[valid] / aod840[valid]) / np.log(550 / 840)

        beta[valid] = aod550[valid] * (0.55) ** alpha[valid]

        return alpha, beta



    
    def _get_solar_zenith_angle(self, timestamps, location):
        if isinstance(timestamps[0], str):
            dt_index = pd.to_datetime(timestamps, format='%Y%m%d%H')
        else:
            dt_index = pd.DatetimeIndex(timestamps)
        
        solpos = pvlib.solarposition.get_solarposition(
            dt_index,
            location.latitude,
            location.longitude
        )
        return solpos['zenith'].to_numpy()
