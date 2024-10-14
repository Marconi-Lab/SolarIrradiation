import logging
from datetime import timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from geopy import location as Glocation


class MerraDataMetadata:

    def __init__(self, filename: Union[str, List[str]], location: Glocation, name: str):
        self.filename = filename
        self.location = location
        self.name = name


class MerraDataResult:
    _TIME_KEY = "time"
    _FILENAME_KEY = "Filename"

    def __init__(self, metadata: MerraDataMetadata, data: pd.DataFrame):
        self._metadata = metadata
        self._data = data

    @staticmethod
    def from_xarray(ds: xr.Dataset, location: Glocation, name: str):

        filename = ds.attrs[MerraDataResult._FILENAME_KEY]
        metadata = MerraDataMetadata(filename, location, name)

        data_vars = list(ds.data_vars.keys())
        if not data_vars:
            logging.warning("No data variables found in the dataset.")
            raise ValueError("Dataset contains no data variables.")

        data_frames = []

        for var_name in data_vars:
            var_data = ds[var_name].values
            if not var_data.size:
                logging.warning(
                    f"Variable {var_name} for {name} in {location} read from {filename} is empty, skipping."
                )
                continue

            if MerraDataResult._TIME_KEY in ds.dims:
                time_values = ds[MerraDataResult._TIME_KEY].values
                if time_values.size == 0:
                    msg = f"Time values for {var_name} for {name} in {location} read from {filename} is empty."
                    logging.warning(msg)
                    raise ValueError(f"No time values found for variable {var_name}.")

                start_date = pd.to_datetime(
                    ds.attrs.get("RangeBeginningDate", "1970-01-01")
                )
                time_index = [start_date + timedelta(hours=int(t)) for t in time_values]
            else:
                time_index = np.arange(
                    len(var_data)
                ).tolist()  # Default to range if no time dimension

            df = pd.DataFrame(var_data.flatten(), index=time_index, columns=[var_name])

            data_frames.append(df)

        combined_df = pd.concat(data_frames, axis=1)
        return MerraDataResult(metadata, combined_df)

    def to_df(self) -> pd.DataFrame:
        return self._data

    @property
    def location(self) -> Glocation:
        """
        Returns the location for the dataset.
        """
        return self._metadata.location

    @property
    def filename(self) -> Union[str, List[str]]:
        """
        Returns the filename of the NetCDF file.
        """
        return self._metadata.filename

    @property
    def name(self) -> str:
        return self._metadata.name

    def get_variable_names(self) -> List[str]:
        """
        Returns a list of variable names in the dataset.
        """
        return self._data.columns.tolist()

    def to_np(self) -> np.ndarray:
        return self._data[self.get_variable_names()].values

    def get_variable_as_numpy(self, var_name) -> Optional[np.ndarray]:
        """
        Returns the specified variable as a NumPy array.
        If the variable does not exist, it returns None.
        """
        if var_name in self.get_variable_names():
            return self._data[var_name].values
        else:
            print(f"Variable '{var_name}' not found in data read from {self.filename}.")
            return None
