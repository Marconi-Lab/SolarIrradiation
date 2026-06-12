from dataclasses import dataclass
from enum import Enum


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

    AEROSOL_ANGSTROM_PARAMETER = MerraProductData(
        name="Total Aerosol Angstrom parameter 470-870 nm ",
        database_id="tavg1_2d_aer_Nx",
        database_name="M2T1NXAER.5.12.4",
        product_name="TOTANGSTR",
    )
    AEROSOL_OPTICAL_DEPTH_ANALYSIS = MerraProductData(
        name="Aerosol Optical Depth",
        database_id="inst3_2d_gas_Nx",
        # GES DISC OPeNDAP requires the version suffix on the collection
        # path segment; without it every URL 404s. Other entries above
        # already have ``.5.12.4`` baked in.
        database_name="M2I3NXGAS.5.12.4",
        product_name="AODANA",
    )
    AEROSOL_EXTINCTION_550nm = MerraProductData(
        name="Total Aerosol Extinction AOT 550 nm ",
        database_id="tavg1_2d_aer_Nx",
        database_name="M2T1NXAER.5.12.4",
        product_name="TOTEXTTAU",
    )
    AEROSOL_SCATTERING_550nm = MerraProductData(
        name="Total Aerosol Scattering AOT 550 nm",
        database_id="tavg1_2d_aer_Nx",
        database_name="M2T1NXAER.5.12.4",
        product_name="TOTSCATAU",
    )
    DEW_POINT_TEMPERATURE = MerraProductData(
        name="Dew Point Temperature",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="T2MDEW",
    )
    EVAPORATION_FLUX = MerraProductData(
        name="Evaporation Flux",
        database_id="tavg1_2d_flx_Nx",
        database_name="M2T1NXFLX.5.12.4",
        product_name="EVAP",
    )
    RELATIVE_HUMIDITY = MerraProductData(
        name="Relative Humidity",
        database_id="tavg3_3d_cld_Np",
        database_name="M2T3NPCLD.5.12.4",
        product_name="RH",
    )
    SPECIFIC_HUMIDITY = MerraProductData(
        name="Specific Humidity",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="QV2M",
    )
    SURFACE_ALBEDO = MerraProductData(
        name="Surface Albedo",
        database_id="tavg1_2d_rad_Nx",
        database_name="M2T1NXRAD.5.12.4",
        product_name="ALBEDO",
    )
    TOTAL_PRECIPITATION = MerraProductData(
        name="Total Precipitation",
        database_id="tavg1_2d_flx_Nx",
        database_name="M2T1NXFLX.5.12.4",
        product_name="PRECTOT",
    )
    TOTAL_COLUMN_OZONE = MerraProductData(
        name="total column ozne",
        database_id="inst1_2d_asm_Nx",
        database_name="M2I1NXASM.5.12.4",
        product_name="TO3",
    )
    TOTAL_PRECIPITATION_WATER_VAPOR = MerraProductData(
        name="Total precipitable water vapour",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="TQV",
    )
    BIAS_CORRECTED_PRECIPITATION = MerraProductData(
        name="Bias Corrected Precipitation",
        database_id="tavg1_2d_flx_Nx",
        database_name="M2T1NXFLX.5.12.4",
        product_name="PRECTOTCORR",
    )
    MAXIMUM_PRECIPTATION = MerraProductData(
        name="Maximum Precipitation",
        database_id="statD_2d_slv_Nx",
        database_name="M2SDNXSLV.5.12.4",
        product_name="TPRECMAX",
    )
    LAKE_AREA_FRACTION = MerraProductData(
        name="Lake Area Fraction",
        database_id="const_2d_asm_Nx",
        database_name="M2C0NXASM.5.12.4",
        product_name="FRLAKE",
    )
    lAND_AREA_FRACTION = MerraProductData(
        name="Land Area Fraction",
        database_id="const_2d_asm_Nx",
        database_name="M2C0NXASM.5.12.4",
        product_name="FRLAND",
    )
    SEA_AREA_FRACTION = MerraProductData(
        name="Sea Area Fraction",
        database_id="const_2d_asm_Nx",
        database_name="M2C0NXASM.5.12.4",
        product_name="FROCEAN",
    )
    SEA_ICE_AREA_FRACTION = MerraProductData(
        name="Sea Ice Area Fraction",
        database_id="tavg1_2d_ocn_Nx",
        database_name="M2T1NXOCN.5.12.4",
        product_name="FRSEAICE",
    )
    SURFACE_PRESSURE = MerraProductData(
        name="surface pressure",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="PS",
    )

    AIR_TEMPERATURE = MerraProductData(
        name="Air Temperature at 2m",
        database_id="tavg1_2d_slv_Nx",
        database_name="M2T1NXSLV.5.12.4",
        product_name="T2M",
    )
    MAX_AIR_TEMPERATURE = MerraProductData(
        name="Maximum Air Temperature at 2m",
        database_id="statD_2d_slv_Nx",
        database_name="M2SDNXSLV.5.12.4",
        product_name="T2MMAX",
    )
    MIN_AIR_TEMPERATURE = MerraProductData(
        name="Minimum Air Temperature at 2m",
        database_id="statD_2d_slv_Nx",
        database_name="M2SDNXSLV.5.12.4",
        product_name="T2MMIN",
    )
    EASTWARD_WIND = MerraProductData(
        name="Eastward Wind at 10m",
        database_id="inst1_2d_asm_Nx",
        database_name="M2I1NXASM.5.12.4",
        product_name="U10M",
    )
    NORTHWARD_WIND = MerraProductData(
        name="Northward Wind at 10m",
        database_id="inst1_2d_asm_Nx",
        database_name="M2I1NXASM.5.12.4",
        product_name="V10M",
    )
