from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class MerraProductData:
    name: str
    database_id: str
    database_name: str
    product_name: str


class MerraProducts(Enum):
    """
    These products represent the different merra products that can be checked here
    https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
    This class should be extended whenever new products are required
    """

    AER = MerraProductData(
        name="Total Aerosol Angstrom parameter 470-870 nm ",
        database_id="tavg1_2d_aer_Nx",
        database_name="M2T1NXAER.5.12.4",
        product_name="TOTANGSTR",
    )
    AERTOTEXT = MerraProductData(
        name="Total Aerosol Extinction AOT 550 nm ",
        database_id="tavg1_2d_aer_Nx",
        database_name="M2T1NXAER.5.12.4",
        product_name="TOTEXTTAU",
    )
    AERTOTSCAT = MerraProductData(
        name="Total Aerosol Scattering AOT 550 nm",
        database_id="tavg1_2d_aer_Nx",
        database_name="M2T1NXAER.5.12.4",
        product_name="TOTSCATAU",
    )
    DEWPT = MerraProductData(
        name="Dew Point Temperature",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="T2MDEW",
    )
    EVSPSBL = MerraProductData(
        name="Evaporation Flux",
        database_id="tavg1_2d_flx_Nx",
        database_name="M2T1NXFLX.5.12.4",
        product_name="EVAP",
    )
    HUR = MerraProductData(
        name="Relative Humidity",
        database_id="tavg3_3d_cld_Np",
        database_name="M2T3NPCLD.5.12.4",
        product_name="RH",
    )
    HUSS = MerraProductData(
        name="Specific Humidity",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="QV2M",
    )
    PR = MerraProductData(
        name="Total Precipitation",
        database_id="tavg1_2d_flx_Nx",
        database_name="M2T1NXFLX.5.12.4",
        product_name="PRECTOT",
    )
    PRBC = MerraProductData(
        name="Bias Corrected Precipitation",
        database_id="tavg1_2d_flx_Nx",
        database_name="M2T1NXFLX.5.12.4",
        product_name="PRECTOTCORR",
    )
    PRMAX = MerraProductData(
        name="Maximum Precipitation",
        database_id="statD_2d_slv_Nx",
        database_name="M2SDNXSLV.5.12.4",
        product_name="TPRECMAX",
    )
    SFTKF = MerraProductData(
        name="Lake Area Fraction",
        database_id="const_2d_asm_Nx",
        database_name="M2C0NXASM.5.12.4",
        product_name="FRLAKE",
    )
    SFTLF = MerraProductData(
        name="Land Area Fraction",
        database_id="const_2d_asm_Nx",
        database_name="M2C0NXASM.5.12.4",
        product_name="FRLAND",
    )
    SFTOF = MerraProductData(
        name="Sea Area Fraction",
        database_id="const_2d_asm_Nx",
        database_name="M2C0NXASM.5.12.4",
        product_name="FROCEAN",
    )
    SIC = MerraProductData(
        name="Sea Ice Area Fraction",
        database_id="tavg1_2d_ocn_Nx",
        database_name="M2T1NXOCN.5.12.4",
        product_name="FRSEAICE",
    )
    TAS = MerraProductData(
        name="Air Temperature at 2m",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="T2M",
    )
    TASMAX = MerraProductData(
        name="Maximum Air Temperature at 2m",
        database_id="statD_2d_slv_Nx",
        database_name="M2SDNXSLV.5.12.4",
        product_name="T2MMAX",
    )
    TASMIN = MerraProductData(
        name="Minimum Air Temperature at 2m",
        database_id="statD_2d_slv_Nx",
        database_name="M2SDNXSLV.5.12.4",
        product_name="T2MMIN",
    )
    UAS = MerraProductData(
        name="Eastward Wind at 10m",
        database_id="inst1_2d_asm_Nx",
        database_name="M2I1NXASM.5.12.4",
        product_name="U10M",
    )
    VAS = MerraProductData(
        name="Northward Wind at 10m",
        database_id="inst1_2d_asm_Nx",
        database_name="M2I1NXASM.5.12.4",
        product_name="V10M",
    )
