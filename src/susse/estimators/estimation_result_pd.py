import logging
from typing import List

import numpy as np
import pandas as pd


class EstimationResultDF:
    """
    This class is meant to be a convenience class to serve as base class for all the estimation results that store the
    data in a pd.Dataframe internally. This class simply contains some basic consistency checks
    """

    def __init__(self, df: pd.DataFrame, necessary_columns: List[str]):
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.error("The DataFrame index must be a pandas DatetimeIndex.")
            raise ValueError("DataFrame index must be a pandas DatetimeIndex.")
        else:
            logging.debug("DataFrame index is a pandas DatetimeIndex.")

        missing_columns = [col for col in necessary_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing necessary columns: {missing_columns}")
            raise ValueError(f"Missing necessary columns: {missing_columns}")
        else:
            logging.debug("All necessary columns are present.")

        if df[necessary_columns].isnull().values.any():
            logging.warning("ClearSkyData contains NaN values.")
        else:
            logging.debug("No NaN values in necessary columns.")

        for col in necessary_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                logging.error(f"The column '{col}' must contain numeric data.")
                raise TypeError(f"Column '{col}' must contain numeric data.")
            else:
                logging.debug(f"Column '{col}' contains numeric data.")

        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df
