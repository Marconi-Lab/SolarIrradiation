import random

import numpy as np
import pandas as pd
import pvlib
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder


class LocationHandler:
    def __init__(self, name: str, dataframe: pd.DataFrame):
        """
        Handles data for a specific location
        """
        self.name = name
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        df = self._dataframe.copy()
        df.set_index("MEASURE_DATE", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        df.drop(
            columns=[
                "DIM_SITE_ID",
                "SITE_LATITUDE",
                "SITE_LONGITUDE",
                "Location_Name",
                "SITE_COUNTRY",
            ],
            inplace=True,
        )

        return df

    @property
    def ghi(self) -> np.ndarray:
        return self._dataframe["MEASURE_VALUE"].values

    @property
    def coordinates(self) -> tuple:
        return (
            float(self._dataframe["SITE_LATITUDE"].values[0]),
            float(self._dataframe["SITE_LONGITUDE"].values[0]),
        )

    @property
    def timezone(self) -> str:
        tf = TimezoneFinder()
        return tf.timezone_at(lng=self.coordinates[1], lat=self.coordinates[0])

    @property
    def time_range(self) -> tuple:
        return (self.dataframe.index.min(), self.dataframe.index.max())

    @property
    def altitude(self) -> float:
        return pvlib.location.Location(
            self.coordinates[0], self.coordinates[1]
        ).altitude


class DataHandler:
    def __init__(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe
        self._locations = {}
        self._add_location_labels()

        for location_label in self._dataframe["Location_Label"].unique():
            location_df = dataframe[dataframe["Location_Label"] == location_label]
            self._locations[location_label] = LocationHandler(
                location_label, location_df
            )
            setattr(self, location_label, self._locations[location_label])

    def _add_location_labels(self):
        geolocator = Nominatim(user_agent="solar")

        def get_location_name(row):
            location = geolocator.reverse((row["SITE_LATITUDE"], row["SITE_LONGITUDE"]))
            location_str = (
                location.address
                if location
                else "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
            )
            location_components = location_str.split(", ")
            location_name = str(location_components[0]).replace(" ", "_").lower()
            location_name = str(location_name).replace("-", "_").lower()
            return location_name

        unique_site_ids = self._dataframe[
            ["DIM_SITE_ID", "SITE_LATITUDE", "SITE_LONGITUDE"]
        ].drop_duplicates()

        location_names = {}
        for _, row in unique_site_ids.iterrows():
            location_name = get_location_name(row)
            if location_name in location_names.values():
                suffix = "a"
                while f"{location_name}_{suffix}" in location_names.values():
                    suffix = chr(ord(suffix) + 1)
                location_name = f"{location_name}_{suffix}"
            location_names[row["DIM_SITE_ID"]] = location_name

        location_labels = {}
        for i, location in enumerate(location_names.values()):
            location_labels[location] = f"area_{chr(97 + i)}"

        self._dataframe["Location_Name"] = self._dataframe["DIM_SITE_ID"].map(
            location_names
        )
        self._dataframe["Location_Label"] = self._dataframe["Location_Name"].map(
            location_labels
        )

    @property
    def locations(self):
        return self._dataframe["Location_Label"].unique()
