from datetime import datetime

import pandas as pd

from typing import List

import numpy as np

import logging


class ClearSkyEstimate:
    _GHI_COL = "ghi"
    _DHI_COL = "dhi"
    _DNI_COL = "dni"

    def __init__(self, clear_sky_df: pd.DataFrame):
        # Check that clear_sky_df is a DataFrame
        if not isinstance(clear_sky_df, pd.DataFrame):
            logging.error("The 'clear_sky_df' parameter must be a pandas DataFrame.")
            raise TypeError("clear_sky_df must be a pandas DataFrame.")
        else:
            logging.debug("Received a pandas DataFrame.")

        # Check that index is DatetimeIndex
        if not isinstance(clear_sky_df.index, pd.DatetimeIndex):
            logging.error("The DataFrame index must be a pandas DatetimeIndex.")
            raise ValueError("DataFrame index must be a pandas DatetimeIndex.")
        else:
            logging.debug("DataFrame index is a pandas DatetimeIndex.")

        # Check for necessary columns
        necessary_columns = [self._GHI_COL, self._DHI_COL, self._DNI_COL]
        missing_columns = [col for col in necessary_columns if col not in clear_sky_df.columns]
        if missing_columns:
            logging.error(f"Missing necessary columns: {missing_columns}")
            raise ValueError(f"Missing necessary columns: {missing_columns}")
        else:
            logging.debug("All necessary columns are present.")

        # Check for NaN values in necessary columns
        if clear_sky_df[necessary_columns].isnull().values.any():
            logging.warning("ClearSkyData contains NaN values.")
        else:
            logging.debug("No NaN values in necessary columns.")

        # Check data types of necessary columns
        for col in necessary_columns:
            if not np.issubdtype(clear_sky_df[col].dtype, np.number):
                logging.error(f"The column '{col}' must contain numeric data.")
                raise TypeError(f"Column '{col}' must contain numeric data.")
            else:
                logging.debug(f"Column '{col}' contains numeric data.")

        self._df = clear_sky_df

    def to_df(self) -> pd.DataFrame:
        return self._df

    @property
    def ghi(self) -> np.ndarray:
        return self._df[self._GHI_COL].values.to_numpy()

    @property
    def dhi(self) -> np.ndarray:
        return self._df[self._DHI_COL].values.to_numpy()

    @property
    def dni(self) -> np.ndarray:
        return self._df[self._DNI_COL].values.to_numpy()

    @property
    def timestamp(self) -> List[datetime]:
        return self._df.index.to_pydatetime().tolist()
