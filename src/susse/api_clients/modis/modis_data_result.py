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

    _DATE_TAG = "calendar_date"
    _BAND_TAG = "band"
    _DATA_TAG = "data"
    _DATE_FORMAT = "%Y-%m-%d"

    def __init__(self, date: datetime, band_name: str, data: List[number]):
        self.date = date
        self.band = band_name
        self.data = data
        self.data_avg = np.mean(np.asarray(data))

    @classmethod
    def from_subset_dict(cls, subset_dict: dict, scale: float = 1.0):
        date = datetime.strptime(subset_dict.get(cls._DATE_TAG), cls._DATE_FORMAT)
        band_name = subset_dict.get(cls._BAND_TAG)
        data = subset_dict.get(cls._DATA_TAG, [])
        if not isinstance(data, list):
            raise ValueError(
                f"Expected '{cls._DATA_TAG}' to be a list but got {type(data)}"
            )
        data = [data_point * scale for data_point in data]
        return ModisDataPoint(date, band_name, data)


class ModisDataResult:
    """
    This class represents the result from a Modis API call. Each call will generally contain several datapoints for the
    provided time-interval and location
    """

    _LATITUDE_TAG = "latitude"
    _LONGITUDE_TAG = "longitude"
    _CELLSIZE_TAG = "cellsize"
    _NROWS_TAG = "nrows"
    _NCOLS_TAG = "ncols"
    _UNITS_TAG = "units"
    _SCALE_TAG = "scale"
    _SUBSET_TAG = "subset"

    def __init__(
        self,
        latitude: float,
        longitude: float,
        data_points: List[ModisDataPoint],
        cellsize: float = None,
        nrows: int = None,
        ncols: int = None,
        units: str = None,
        scale: float = None,
    ):
        self._latitude = latitude
        self._longitude = longitude
        self._cellsize = cellsize
        self._nrows = nrows
        self._ncols = ncols
        self._data_points = data_points
        self._units = units
        self._scale = scale

    @classmethod
    def from_request_response(cls, request_response: dict):
        latitude_str = request_response.get(cls._LATITUDE_TAG)
        longitude_str = request_response.get(cls._LONGITUDE_TAG)
        cellsize_str = request_response.get(cls._CELLSIZE_TAG)
        nrows_str = request_response.get(cls._NROWS_TAG)
        ncols_str = request_response.get(cls._NCOLS_TAG)
        units_str = request_response.get(cls._UNITS_TAG)
        scale_str = request_response.get(cls._SCALE_TAG)
        subset = request_response.get(cls._SUBSET_TAG, [])

        if not latitude_str or not longitude_str:
            raise ValueError(
                "Missing latitude and longitude parameters from Modis result"
            )

        latitude = float(latitude_str)
        longitude = float(longitude_str)
        cellsize = float(cellsize_str) if cellsize_str else None
        nrows = int(nrows_str) if nrows_str else None
        ncols = int(ncols_str) if ncols_str else None
        scale = float(scale_str) if scale_str else 1.0

        if not isinstance(subset, list):
            raise ValueError(
                f"Expected '{cls._SUBSET_TAG}' to be a list but got {type(subset)}"
            )

        data_points = []
        for data_dict in subset:
            data_points.append(ModisDataPoint.from_subset_dict(data_dict, scale=scale))

        return cls(
            latitude=latitude,
            longitude=longitude,
            data_points=data_points,
            cellsize=cellsize,
            nrows=nrows,
            ncols=ncols,
            units=units_str,
            scale=scale,
        )

    @property
    def latitude(self) -> float:
        return self._latitude

    @property
    def longitude(self) -> float:
        return self._longitude

    @property
    def data_points(self) -> List[ModisDataPoint]:
        return self._data_points

    def get_time_average(self) -> float:
        values = [v.data_avg for v in self.data_points]
        return np.mean(np.asarray(values))
