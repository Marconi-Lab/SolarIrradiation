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
        radius_factors = self._calculate_radius_factors(timestamps)
        
        # Prepare parameters dictionary
        params = {
            'p': data.to_numpy(NASAPowerProducts.SURFACE_PRESSURE),
            'aod550': data.to_numpy(NASAPowerProducts.AEROSOL_OPTICAL_DEPTH_550nm),
            'aod840': data.to_numpy(NASAPowerProducts.AEROSOL_OPTICAL_DEPTH_840nm),
            'pw': data.to_numpy(NASAPowerProducts.PRECIPITABLE_WATER),
            'albedo': data.to_numpy(NASAPowerProducts.ALL_SKY_SURFACE_ALBEDO),
            'z': self._get_solar_zenith_angle(start_date, end_date, location, resolution),
            'ozone': data.to_numpy(NASAPowerProducts.TOTAL_COLUMN_OZONE) * 0.001,
            'radius': radius_factors
        }

        # Calculate Angstrom parameters
        alpha, beta = self._calculate_angstrom_params(params['aod550'], params['aod840'])
 
        # Prepare and execute REST2 inputs
        rest2_args = {
            'p': params['p'] * 10, 
            'albedo': params['albedo'],
            'ssa': 0.92,
            'g': 0.7,
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
        """Generate resolution-appropriate pandas timestamps."""
        freq_map = {
            TemporalResolution.HOURLY: 'H',
            TemporalResolution.DAILY: 'D',
            TemporalResolution.MONTHLY: 'MS'
        }
        return pd.date_range(start_date, end_date, freq=freq_map[resolution])

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



    def _get_solar_zenith_angle(self, start, end, location, resolution):
        """
        Compute solar zenith angle for the given time range and location using pvlib.
        The method ensures proper frequency based on resolution and timezone awareness.
        """
        freq_map = {
            TemporalResolution.HOURLY: 'H',
            TemporalResolution.DAILY: 'D',
            TemporalResolution.MONTHLY: 'MS'
        }
        freq = freq_map.get(resolution, 'H')

        times = pd.date_range(start=start, end=end, freq=freq, tz='UTC')
        
        if hasattr(location, 'tzinfo') and location.tzinfo is not None:
            times = times.tz_convert(location.tzinfo)

        if hasattr(location, 'latitude') and hasattr(location, 'longitude'):
            lat = location.latitude
            lon = location.longitude
        else:
            lat, lon = location  # assume location is a tuple

        solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
        sza = solpos['zenith'].to_numpy()
        return sza

