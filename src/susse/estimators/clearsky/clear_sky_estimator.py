
from abc import ABC, abstractmethod
from geopy import location as Glocation
import datetime

import pandas as pd

from overrides import override

import pvlib


class ClearSkyEstimator(ABC):

    def __init__(self, name: str):
        self._name = name
        self._timedelta = datetime.timedelta(hours=1)
        self.

    @abstractmethod
    def estimate_clear_sky(self, location: Glocation, start_date: datetime.datetime, end_date: datetime.datetime):
        pass

    def set_timedelta(self, timedelta: datetime.timedelta):
        self._timedelta = timedelta


class ClearSkyEstimatorPVlib(ClearSkyEstimator):

    def __init__(self):
        super().__init__("clear_sky_pvlib")

    @override
    def estimate_clear_sky(self, location: Glocation, start_date: datetime.datetime, end_date: datetime.datetime):
        location_altitude = pvlib.location.lookup_altitude(location.latitude, location.longitude)
        pv_location = pvlib.location.Location(location.latitude, location.longitude, altitude=location_altitude)

        times = pd.date_range(
            start=start_date, end=end_date, freq=self._timedelta, tz=self.location.tz
        )
        clear_sky_df = pv_location.get_clearsky()

