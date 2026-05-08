import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

from .modis_api_config import ModisConfig


class ModisProdFrequency(Enum):
    """
    This enum represents the possible MODIS result frequencies
    """

    DAILY = "Daily"
    MONTHLY = "Monthly"
    YEARLY = "Yearly"
    FOUR_DAY = "4 - Day"
    EIGHT_DAY = "8-Day"
    SIXTEEN_DAY = "16-Day"
    VARIES = "Varies"

    @classmethod
    def from_str(cls, frequency_str: str):
        try:
            return cls(frequency_str)
        except ValueError:
            logging.warning(f"Could not find frequency for {frequency_str}")
            return None


class ModisProductEnum(Enum):
    """
    This class represents the different Modis products of interest. It also contains a function to provide a default
    band of interest. For a list of all possible products with their bands and documentation see
    https://modis.ornl.gov/documentation.html
    """
    SURFACE_REFLECTANCE_8DAY = "MOD09A1"  # 500 m, 8-day composite
    BRDF_PARAMETERS_16DAY = "MCD43A1"  # BRDF model parameters (RossThick/LiSparse)
    ALBEDO_16DAY = "MCD43A3"  # White/black-sky & shortwave albedos
    NBAR_16DAY = "MCD43A4"  # Nadir BRDF-Adjusted Reflectance (bands 1–7)

    # --- Land Surface Temperature & Emissivity ---
    LAND_SURFACE_TEMPERATURE_DAILY = "MOD11A1"  # 1 km, daily (includes LST_Day_1KM)
    LAND_SURFACE_TEMPERATURE_8DAY = "MOD11A2"  # 1 km, 8-day composite

    # --- Vegetation state & structure ---
    VEGETATION_INDEX_16DAY = "MOD13Q1"  # 250 m NDVI/EVI (16-day)
    LAI_FPAR_4DAY = "MCD15A3H"  # 500 m, 4-day LAI/FPAR
    LAND_COVER_YEARLY = "MCD12Q1"  # 500 m, yearly IGBP/UMD/etc. classes

    # --- Surface/flux proxies helpful for solar modeling ---
    EVAPOTRANSPIRATION_8DAY = "MOD16A2"  # 500 m ET (8-day)
    GPP_8DAY = "MOD17A2H"  # 500 m GPP (8-day)

    # --- Cryosphere / masking (affects albedo & irradiance) ---
    SNOW_COVER_DAILY = "MOD10A1"  # 500 m snow cover (daily)

    # Add relevant products here

    def default_band_name(self) -> Optional[str]:
        return {
            # ORNL DAAC's bands endpoint reports lowercase 'km' for the
            # 1 km LST band on MOD11A1 / MOD11A2; the previous default
            # ``LST_Day_1KM`` was case-mismatched and silently fell back
            # to whatever the validator picked.
            ModisProductEnum.LAND_SURFACE_TEMPERATURE_DAILY: "LST_Day_1km",
            ModisProductEnum.LAND_SURFACE_TEMPERATURE_8DAY: "LST_Day_1km",
            ModisProductEnum.SURFACE_REFLECTANCE_8DAY: "sur_refl_b01",  # red
            ModisProductEnum.VEGETATION_INDEX_16DAY: "250m_16_days_NDVI",
            ModisProductEnum.NBAR_16DAY: "Nadir_Reflectance_Band1",
            ModisProductEnum.ALBEDO_16DAY: "Albedo_BSA_shortwave",
            ModisProductEnum.LAI_FPAR_4DAY: "Lai_500m",
            ModisProductEnum.LAND_COVER_YEARLY: "LC_Type1",
            ModisProductEnum.SNOW_COVER_DAILY: "NDSI_Snow_Cover",
            ModisProductEnum.EVAPOTRANSPIRATION_8DAY: "ET_500m",
            ModisProductEnum.GPP_8DAY: "Gpp_500m",
        }.get(self)


class ModisBand:
    """
    This class contains the information about an individual band for a Modis product
    """

    _BAND_TAG = "band"
    _DESCRIPTION_TAG = "description"
    _VALID_RANGE_TAG = "valid_range"
    _SCALE_FACTOR_TAG = "scale_factor"
    _ADD_OFFSET_TAG = "add_offset"

    def __init__(
        self,
        band_name: str,
        description: Optional[str] = None,
        valid_range: Optional[Tuple[float, float]] = None,
        add_offset: Optional[float] = None,
        scale_factor: Optional[float] = None,
    ):
        self._band_name = band_name
        self._description = description
        self._valid_range = valid_range
        self._add_offset = add_offset
        self._scale_factor = scale_factor

    @property
    def scale_factor(self) -> Optional[float]:
        return self._scale_factor

    @property
    def add_offset(self) -> Optional[float]:
        return self._add_offset

    def __repr__(self):
        return f"{self._band_name}: {self._description}"

    @property
    def name(self):
        return self._band_name

    @property
    def description(self):
        return self._description

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, str]):
        add_offset_str = json_dict.get(cls._ADD_OFFSET_TAG)
        scale_factor_str = json_dict.get(cls._SCALE_FACTOR_TAG)
        valid_range_str = json_dict.get(cls._VALID_RANGE_TAG)

        add_offset = float(add_offset_str) if add_offset_str is not None else None
        scale_factor = float(scale_factor_str) if scale_factor_str is not None else None

        range_split = (
            valid_range_str.split(" to ") if valid_range_str is not None else None
        )
        valid_range = (
            (float(range_split[0]), float(range_split[1]))
            if range_split is not None
            else None
        )

        return cls(
            band_name=json_dict[cls._BAND_TAG],
            description=json_dict.get(cls._DESCRIPTION_TAG),
            valid_range=valid_range,
            add_offset=add_offset,
            scale_factor=scale_factor,
        )


class ModisProduct:
    """
    This class represents a full Modis Product with all required information
    """

    _RESPONSE_BANDS_TAG = "bands"
    _PRODUCT_TAG = "product"
    _FREQUENCY_TAG = "frequency"
    _RESOLUTION_METERS_TAG = "resolution_meters"
    _DESCRIPTION_TAG = "description"

    def __init__(  # Corrected method name
        self,
        product_name: str,
        resolution_meters: float,
        description: str,
        frequency: Optional[ModisProdFrequency] = None,
        default_band_name: Optional[str] = None,
    ):
        self._product_name = product_name
        self._frequency = frequency
        self._resolution_meters = resolution_meters
        self._description = description
        self._bands = self._fetch_bands()
        self._default_band = (
            None
            if default_band_name is None
            else self.get_band_by_name(default_band_name)
        )

    @staticmethod
    def product_tag() -> str:
        return ModisProduct._PRODUCT_TAG

    @property
    def default_band(self) -> Optional[ModisBand]:
        return self._default_band

    def get_band_by_name(self, band_name: str) -> Optional[ModisBand]:
        for band in self._bands:
            if band.name == band_name:
                return band
        logging.warning(
            f"Could not find band {band_name} for product {self._product_name}"
        )
        return None

    def get_band_names(self) -> List[str]:
        return [band.name for band in self._bands]

    @property
    def name(self) -> str:
        return self._product_name

    def has_band(self, band_name: str) -> bool:
        for band in self._bands:
            if band.name == band_name:
                return True
        return False

    @classmethod
    def from_json_dict(
        cls, json_dict: Dict[str, str], default_band_name: Optional[str] = None
    ):
        return cls(
            product_name=json_dict[cls._PRODUCT_TAG],
            frequency=ModisProdFrequency.from_str(json_dict[cls._FREQUENCY_TAG]),
            resolution_meters=float(json_dict[cls._RESOLUTION_METERS_TAG]),
            description=json_dict[cls._DESCRIPTION_TAG],
            default_band_name=default_band_name,
        )

    def get_product_name(self) -> str:
        """Returns the product name string (e.g., 'MOD21A2')."""
        return self._product_name

    def _fetch_bands(self) -> List[ModisBand]:
        request_url = ModisConfig.get_band_url(self._product_name)
        req_bands = requests.get(request_url)
        if req_bands.status_code == 200:
            bands_data = json.loads(req_bands.text)[self._RESPONSE_BANDS_TAG]
            return [ModisBand.from_json_dict(band_data) for band_data in bands_data]
        else:
            message = ("Failed to fetch bands for product {}: {}").format(
                self._product_name, req_bands.text
            )
            logging.warning(message)

            return []

    def __repr__(self):
        return f"{self._product_name}: {self._frequency} ({self._resolution_meters}m)\nDescription: {self._description}\nBands: {self._bands}"


class ModisProductFactory:
    """
    A Factory class that creates individual Modis products from the respective enums
    """

    _RESPONSE_PRODUCT_TAG = "products"

    def __init__(self):
        self._products = None

    def products_fetched(self) -> bool:
        return self._products is not None

    def _fetch_products(self) -> None:
        url = ModisConfig.get_product_url()
        header = ModisConfig.HEADERS
        response = requests.get(url, headers=header)
        if response.status_code == 200:
            json_dict = json.loads(response.text)
            self._products = json_dict[self._RESPONSE_PRODUCT_TAG]
        else:
            raise requests.exceptions.HTTPError(
                f"Failed to fetch MODIS products: {response.text}"
            )

    def get_product_by_enum(self, product_enum: ModisProductEnum) -> ModisProduct:
        if not self.products_fetched():
            self._fetch_products()

        for product_data in self._products:  # Iterate through the fetched product data
            # Ensure product_data is a dictionary and has the product tag
            if (
                isinstance(product_data, dict)
                and ModisProduct.product_tag() in product_data
            ):
                if (
                    product_enum.value == product_data[ModisProduct.product_tag()]
                ):  # Compare with product_enum.value
                    return ModisProduct.from_json_dict(
                        product_data, default_band_name=product_enum.default_band_name()
                    )
        raise ValueError(f"Product {product_enum.value} not found")
