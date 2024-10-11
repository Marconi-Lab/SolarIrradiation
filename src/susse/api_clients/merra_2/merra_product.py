from enum import Enum


class Dictionarys():
    RAD = {
        "database_name": "M2T1NXRAD.5.12.4",
        "database_id": "tavg1_2d_rad_Nx",
        "product_name": ["ALBEDO", "CLDTOT", "SWGDN", "SWGDNCLR", "TAUTOT"],
        "standard_name": "radiation",
    }
    SLV = {
        "database_name": "M2T1NXSLV.5.12.4",
        "database_id": "tavg1_2d_slv_Nx",
        "product_name": ["TQV", "TO3", "PS"],
        "standard_name": "surface",
    }
    AER = {
        "database_name": "M2T1NXAER.5.12.4",
        "database_id": "tavg1_2d_aer_Nx",
        "product_name": ["TOTSCATAU", "TOTEXTTAU", "TOTANGSTR"],
        "standard_name": "aerosols",
    }
    ASM = {
        "database_name": "M2C0NXASM.5.12.4",
        "database_id": "const_2d_asm_Nx",
        "product_name": ["PHIS"],
        "standard_name": "parameters",
    }
    DEWPT = {
        "database_name": "M2T1NXSLV.5.12.4",
        "database_id": "tavg1_2d_slv_Nx",
        "product_name": ["T2MDEW"],
        "standard_name": "dew_point_temperature",
    }
    EVSPSBL = {
        "database_name": "M2T1NXFLX.5.12.4",
        "database_id": "tavg1_2d_flx_Nx",
        "product_name": ["EVAP"],
        "standard_name": "water_evaporation_flux",
    }
    HUR = {
        "database_name": "M2T3NPCLD.5.12.4",
        "database_id": "tavg3_3d_cld_Np",
        "product_name": ["RH"],
        "standard_name": "relative_humidity",
    }
    HUSS = {
        "database_name": "M2T1NXSLV.5.12.4",
        "database_id": "tavg1_2d_slv_Nx",
        "product_name": ["QV2M"],
        "standard_name": "specific_humidity",
    }
    PHIS = {
        "database_name": "M2C0NXASM.5.12.4",
        "database_id": "const_2d_asm_Nx",
        "product_name": ["PHIS"],
        "standard_name": "surface_geopotential",
    }
    PR = {
        "database_name": "M2T1NXFLX.5.12.4",
        "database_id": "tavg1_2d_flx_Nx",
        "product_name": ["PRECTOT"],
        "standard_name": "precipitation_flux",
    }
    PRC = {
        "database_name": "M2T1NXFLX.5.12.4",
        "database_id": "tavg1_2d_flx_Nx",
        "product_name": ["PRECCON"],
        "standard_name": "convective_precipitation_flux",
    }
    PRBC = {
        "database_name": "M2T1NXFLX.5.12.4",
        "database_id": "tavg1_2d_flx_Nx",
        "product_name": ["PRECTOTCORR"],
        "standard_name": "precipitation_flux_bias_corr",
    }
    PRMAX = {
        "database_name": "M2SDNXSLV.5.12.4",
        "database_id": "statD_2d_slv_Nx",
        "product_name": ["TPRECMAX"],
        "standard_name": "precipitation_flux",
    }
    PRSN = {
        "database_name": "M2T1NXFLX.5.12.4",
        "database_id": "tavg1_2d_flx_Nx",
        "product_name": ["PRECSNO"],
        "standard_name": "snowfall_flux",
    }
    RLS = {
        "database_name": "M2T1NXRAD.5.12.4",
        "database_id": "tavg1_2d_rad_Nx",
        "product_name": ["LWGNT"],
        "standard_name": "surface_net_downward_longwave_flux",
    }
    SFTGIF = {
        "database_name": "M2C0NXASM.5.12.4",
        "database_id": "const_2d_asm_Nx",
        "product_name": ["FRLANDICE"],
        "standard_name": "land_ice_area_fraction",
    }
    SFTKF = {
        "database_name": "M2C0NXASM.5.12.4",
        "database_id": "const_2d_asm_Nx",
        "product_name": ["FRLAKE"],
        "standard_name": "lake_area_fraction",
    }
    SFTLF = {
        "database_name": "M2C0NXASM.5.12.4",
        "database_id": "const_2d_asm_Nx",
        "product_name": ["FRLAND"],
        "standard_name": "land_area_fraction",
    }
    SFTOF = {
        "database_name": "M2C0NXASM.5.12.4",
        "database_id": "const_2d_asm_Nx",
        "product_name": ["FROCEAN"],
        "standard_name": "sea_area_fraction",
    }
    SHG = {
        "database_name": "M2C0NXASM.5.12.4",
        "database_id": "const_2d_asm_Nx",
        "product_name": ["SHG"],
        "standard_name": "isotropic_stdv_of_gravity_wave_drag_topography",
    }
    SIC = {
        "database_name": "M2T1NXOCN.5.12.4",
        "database_id": "tavg1_2d_ocn_Nx",
        "product_name": ["FRSEAICE"],
        "standard_name": "sea_ice_area_fraction",
    }
    TAS = {
        "database_name": "M2T1NXSLV.5.12.4",
        "database_id": "tavg1_2d_slv_Nx",
        "product_name": ["T2M"],
        "standard_name": "air_temperature",
    }
    TASMAS = {
        "database_name": "M2SDNXSLV.5.12.4",
        "database_id": "statD_2d_slv_Nx",
        "product_name": ["T2MMAX"],
        "standard_name": "air_temperature",
    }
    TASMIN = {
        "database_name": "M2SDNXSLV.5.12.4",
        "database_id": "statD_2d_slv_Nx",
        "product_name": ["T2MMIN"],
        "standard_name": "air_temperature",
    }
    UAS = {
        "database_name": "M2I1NXASM.5.12.4",
        "database_id": "inst1_2d_asm_Nx",
        "product_name": ["U10M"],
        "standard_name": "eastward_wind",
    }
    VAS = {
        "database_name": "M2I1NXASM.5.12.4",
        "database_id": "inst1_2d_asm_Nx",
        "product_name": ["V10M"],
        "standard_name": "northward_wind",
    }


class Products(Enum):
    AEROSOL_OPTICAL_DEPTH = Dictionarys.RAD['product_name'][-1]
    TOTAL_COLUMN_OZONE = Dictionarys.SLV['product_name'][0]
    ANGSTROM_EXPONENT = Dictionarys.AER['product_name'][2]
    SURFACE_PRESSURE = Dictionarys.SLV['product_name'][2]
    PRECIPITATBLE_WATER = Dictionarys.SLV['product_name'][0]
    TEMPERATURE = Dictionarys.TAS['product_name']
    SURFACE_ALBEDO = Dictionarys.RAD['product_name'][0]
    TOTAL_COLUD_COVER = Dictionarys.RAD['product_name'][1]


class Keys(Enum):
    DATABASE_NAME = 'database_name'
    DATABASE_ID = 'database_id'
    PRODUCT_NAMES = 'product_name'
    STANDARD_NAME = 'standard_name'
