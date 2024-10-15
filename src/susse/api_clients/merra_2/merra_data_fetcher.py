import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import xarray as xr
from geopy import location as Glocation

from .merra_config import Merra2Config
from .merra_data_result import MerraDataMetadata, MerraDataResult
from .merra_download_manager import MerraDownloadManager
from .merra_product import MerraProductData, MerraProducts


class MerraDataFetcher:
    """
    This class handles the data fetching through the MERRA2 and is meant as the primary interface for data fetching from
    MERRA2. It utilizes the MerraDownloadManager and Merra2Config to translate a request for a product at a specific
    location and time to a dowloadable link and downloads the file of interest. It also extracts the relevant data from
    the obtained file.
    """

    _DOWNLOAD_FOLDER = str(
        Path(__file__).resolve().parents[4] / "merra_files_downloaded"
    )

    def __init__(self, base_download_folder: str = None):
        self._download_manager = MerraDownloadManager()
        self._base_download_folder = base_download_folder or self._DOWNLOAD_FOLDER

    def set_username_pw(self, username: str, password: str) -> None:
        """
        Encrypts and stores the username and password.
        """
        self._download_manager.set_username_pw(username, password)

    def fetch_product_result(
        self,
        product: MerraProductData,
        location: Glocation,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[MerraDataResult]:
        dates = [
            start_date + timedelta(days=x)
            for x in range((end_date - start_date).days + 1)
        ]

        self._download_data_if_necessary(dates, location, product)

        merra_result = self._read_downloaded_files(dates, location, product)

        if not merra_result:
            logging.warning(
                f"Product '{product.name}' for location '{str(location)}' between {start_date} and {end_date} is empty."
            )
            return None
        else:
            return merra_result

    def _read_downloaded_files(
        self, dates: List[datetime], location: Glocation, product: MerraProductData
    ) -> Optional[MerraDataResult]:

        product_folder = self.get_product_folder(product, location)

        if not os.path.exists(product_folder):
            raise RuntimeError(f"Product folder '{product_folder}' does not exist.")

        data_frames = []
        file_names = []

        for date in dates:
            file_name = Merra2Config.create_file_name(date, product)
            file_path = os.path.join(product_folder, file_name)

            if not os.path.exists(file_path):
                logging.warning(
                    f"File '{file_name}' not found in '{product_folder}'. Skipping."
                )
                continue

            ds = xr.open_dataset(file_path)
            data_result = MerraDataResult.from_xarray(ds, location, product.name)
            ds.close()
            data_frames.append(data_result.to_df())
            file_names.append(file_name)

        if len(data_frames) > 0:
            concatenated = pd.concat(data_frames)
            metadata = MerraDataMetadata(file_names, location, product.name)
            return MerraDataResult(metadata, concatenated)
        else:
            return None

    def _download_data_if_necessary(
        self, dates: List[datetime], location: Glocation, product: MerraProductData
    ) -> None:
        urls: List[str] = []
        for date in dates:
            urls.append(
                Merra2Config.generate_download_link(
                    date, product, location.latitude, location.longitude
                )
            )
        self._download_manager.download_from_urls(
            urls, self.get_product_folder(product, location)
        )

    def get_product_folder(self, product: MerraProductData, location: Glocation) -> str:
        siimple_loc = re.split(", |_|-|!", str(location))[0]
        return str(
            Path(self._base_download_folder) / product.product_name / siimple_loc
        )
