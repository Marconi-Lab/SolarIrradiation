from enum import Enum


class Merra2(Enum):

    BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/'
    VERSION_ID = "5.12.4"


class MerraProductEnum(Enum):
    """
    This class represents the different MERRA-2 products of interest.
    """

    TEMPERATURE = {
        "field_id": "T2M",
        "database_name": "M2I1NXASM",
        "database_id": "inst1_2d_asm_Nx"
    }

    AEROSOL_OPTICAL_DEPTH = {
        "field_id": "AODANA",
        "database_name": "M2I1NXASM",
        "database_id": "inst1_2d_asm_Nx"
    }

    TOTAL_COLUMN_OZONE = {
        "field_id": "TO3",
        "database_name": "M2I1NXASM",
        "database_id": "inst1_2d_asm_Nx"
    }

    SURFACE_PRESURE = {
        "field_id": "PS",
    }

    SURFACE_ALBEDO = {
        "field_id": "ALBEDO",
        "database_name": "M2T1NXRAD",
        "database_id": "tavg1_2d_rad_Nx"
    }

    PRECIPITABLE_WATER = {
        "field_id": "TQV",
    }


class Merra2TimeEnum(Enum):
    """
    This class represents the time slice enum.
    """
    # Representing the time as an actual list or range
    TIME = list(range(24))  # This represents hourly slices from 0 to 23
