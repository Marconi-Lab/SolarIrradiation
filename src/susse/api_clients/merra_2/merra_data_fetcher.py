import logging
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import xarray as xr
from geopy import location as Glocation

from .merra_config import Merra2Config
from .merra_download_manager import MerraDownloadManager
from .merra_product import MerraProductData, MerraProducts


class MerraDataFetcher:
    """
    This class handles the data fetching through the Merra 2
    """

    def __init__(self):
        self._download_manager = MerraDownloadManager()

    def fetch_product_result(
        self,
        product: MerraProductData,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        dates = [
            start_date + timedelta(days=x)
            for x in range((end_date - start_date).days + 1)
        ]

        self._download_data_if_necessary(dates, location, product)

        product_folder = self._download_manager.get_folder_for_product(product)

        if not os.path.exists(product_folder):
            logging.warning(f"Product folder '{product_folder}' does not exist.")
            return None

        dfs = []

        for date in dates:
            file_name = Merra2Config.create_file_name(date, product)
            file_path = os.path.join(product_folder, file_name)

            if os.path.exists(file_path):
                try:
                    with xr.open_mfdataset(
                        file_path, preprocess=MerraDataFetcher._extract_date
                    ) as df:
                        dfs.append(df.to_dataframe())
                except Exception as e:
                    logging.error(f"Issue with file {file_name}: {e}")
            else:
                logging.warning(
                    f"File '{file_name}' not found in '{product_folder}'. Skipping."
                )

        df_hourly = pd.concat(dfs)
        return df_hourly

    def _download_data_if_necessary(self, dates, location, product) -> None:
        urls: List[str] = []
        for date in dates:
            urls.append(
                Merra2Config.generate_download_link(
                    date, product, location.latitude, location.longitude
                )
            )
        self._download_manager.download_from_urls(urls)

    @staticmethod
    def _extract_date(data_set):
        """
        Extracts the date from the filename before merging the datasets.
        """
        if "HDF5_GLOBAL.Filename" in data_set.attrs:
            f_name = data_set.attrs["HDF5_GLOBAL.Filename"]
        elif "Filename" in data_set.attrs:
            f_name = data_set.attrs["Filename"]
        else:
            raise AttributeError("The attribute name has changed again!")
        # find a match between "." and ".nc4" that does not have "." .
        exp = r"(?<=\.)[^\.]*(?=\.nc4)"
        res = re.search(exp, f_name).group(0)
        # Extract the date.
        y, m, d = res[0:4], res[4:6], res[6:8]
        date_str = "%s-%s-%s" % (y, m, d)
        data_set = data_set.assign(date=date_str)
        return data_set
