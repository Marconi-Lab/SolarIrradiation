import datetime
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
import pvlib
from geopy import location as Glocation
from overrides import override

from .clear_sky_data import ClearSkyEstimate


class ClearSkyEstimator(ABC):
    """
    This class is the baseclass for all estimators of Clear-Sky irradiance. it takes as inputs the location of interest
    as well as the start and end date. The return value is of type ClearSkyEstimate, which itself is a wrapper for a
    pandas dataframe
    """

    def __init__(self, name: str):
        self._name = name
        self._timedelta = datetime.timedelta(hours=1)
        self._tz = datetime.timezone.utc

    @abstractmethod
    def estimate_clear_sky(
        self,
        location: Glocation,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
    ) -> ClearSkyEstimate:
        pass

    def set_timedelta(self, timedelta: datetime.timedelta):
        self._timedelta = timedelta

    def set_timezone(self, tz: datetime.timezone):
        self._tz = tz


class ClearSkyEstimatorPVlib(ClearSkyEstimator):

    class Models(Enum):
        SOLIS = "simplified_solis"
        INEICHEN = "ineichen"
        HAURWITZ = "haurwitz"

    def __init__(self):
        super().__init__("clear_sky_pvlib")

    @override
    def estimate_clear_sky(
        self,
        location: Glocation,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
    ):
        location_altitude = pvlib.location.lookup_altitude(
            location.latitude, location.longitude
        )
        pv_location = pvlib.location.Location(
            location.latitude, location.longitude, altitude=location_altitude
        )

        times = pd.date_range(
            start=start_date, end=end_date, freq=self._timedelta, tz=self._tz
        )
        clear_sky_df = pv_location.get_clearsky(
            times=times, model=self.Models.SOLIS.value
        )
        return ClearSkyEstimate(clear_sky_df)
