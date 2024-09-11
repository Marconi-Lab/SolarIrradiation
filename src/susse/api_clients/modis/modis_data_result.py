from datetime import datetime
from typing import List

import numpy as np
from numpy import number

from .modis_product import ModisBand


class ModisDataPoint:
    """
    This class represents a single datapoint of a quantity of interest as obtained from Modis. The data is provided as
    a list, since the data is generally obtained from a grid across a location of interest. This class might need to
    be adjusted to be more flexible if it turns out that additional result structures exist
    """

    def __init__(self, date: datetime, band_name: str, data: List[number]):
        self.date = date
        self.band = band_name
        self.data = data
        self.data_avg = np.mean(np.asarray(data))


class ModisDataResult:
    """
    This class represents the result from a Modis API call. Each call will generally contain several datapoints for the
    provided time-interval and location
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        cellsize: float,
        nrows: int,
        ncols: int,
        units: str,
        data_points: List[ModisDataPoint],
    ):
        self._latitude = latitude
        self._longitude = longitude
        self._cellsize = cellsize
        self._nrows = nrows
        self._ncols = ncols
        self._data_points = data_points
        self._units = units

    @classmethod
    def from_request_response(cls, request_response: dict):
        latitude_str = request_response.get("latitude")
        longitude_str = request_response.get("longitude")
        cellsize_str = request_response.get("cellsize")
        nrows_str = request_response.get("nrows")
        ncols_str = request_response.get("ncols")
        units_str = request_response.get("units")

        data_points = []
        for data_dict in request_response.get("subset"):
            date = datetime.strptime(data_dict.get("calendar_date"), "%Y-%m-%d")
            band_name = data_dict.get("band")
            data = data_dict.get("data")
            data_points.append(ModisDataPoint(date, band_name, data))

        if not latitude_str or not longitude_str:
            raise ValueError(
                "Missing latitude and longitude parameters from Modis result"
            )

        latitude = float(latitude_str)
        longitude = float(longitude_str)
        cellsize = float(cellsize_str) if cellsize_str else None
        nrows = int(nrows_str) if nrows_str else None
        ncols = int(ncols_str) if ncols_str else None

        return cls(
            latitude=latitude,
            longitude=longitude,
            cellsize=cellsize,
            nrows=nrows,
            ncols=ncols,
            units=units_str,
            data_points=data_points,
        )

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def data_points(self):
        return self._data_points

    def get_time_average(self):
        values = [v.data_avg for v in self.data_points]
        return np.mean(np.asarray(values))
