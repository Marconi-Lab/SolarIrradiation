from datetime import datetime

import pandas as pd

from typing import List

import numpy as np

from ..estimation_result_pd import EstimationResultDF


class ClearSkyEstimate(EstimationResultDF):
    _GHI_COL = "ghi"
    _DHI_COL = "dhi"
    _DNI_COL = "dni"

    @staticmethod
    def from_np(ghi: np.ndarray, dhi: np.ndarray, dni: np.ndarray, timestamps: List[datetime]) -> 'ClearSkyEstimate':
        df = pd.DataFrame({ClearSkyEstimate._GHI_COL: ghi,
                           ClearSkyEstimate._DHI_COL: dhi,
                           ClearSkyEstimate._DNI_COL: dni,
                           "timestamps": timestamps})
        df.set_index("timestamps", inplace=True)
        return ClearSkyEstimate(df)

    def __init__(self, clear_sky_df: pd.DataFrame):
        super().__init__(clear_sky_df, [self._GHI_COL, self._DHI_COL, self._DNI_COL])

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
