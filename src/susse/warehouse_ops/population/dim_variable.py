"""Variable catalogue and ``dim_variable`` table population.

Defines every satellite variable SuSSE ingests, with the metadata that
goes into the ``dim_variable`` table (units, descriptions, valid ranges,
spatial resolution). New variables are added here once and pick up
across loaders, jobs, and downstream feature assembly.

Variable routing:

* ``ghi``, ``dhi``, ``dni`` (in either source) → wide ``irradiance_daily``.
* Everything else → the long-format companion table for that source
  (``nasa_daily_vars_long`` or ``cams_daily_vars_long``).

The variable_id strings for NASA POWER match what the warehouse stores
after migration A2 (the legacy ``arimass`` typo was renamed to
``airmass``). New variables added here must use the correct spelling.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import pandas as pd

from .types import Source, VariableSpec
from ..io.bq import BigQueryClient
from ..io.config import TableSchemas

_logger = logging.getLogger(__name__)


# Spatial resolutions (km) at the equator. Negative values in the existing
# warehouse rows are a known bug; we write the correct positive values.
_NASA_POWER_RESOLUTION_KM = 55.5  # 0.5° native grid
_CAMS_RESOLUTION_KM = 5.5  # 0.05° native grid
# MERRA-2 native grid is 0.5° lat × 0.625° lon; using the lat figure as
# the representative spatial resolution.
_MERRA_2_RESOLUTION_KM = 55.5
# MODIS resolutions are per-product. MOD11A1 LST is 1 km native; MCD43A3
# albedo is 500 m; MOD10A1 snow is 500 m; MOD13Q1 NDVI is 250 m.
_MODIS_LST_RESOLUTION_KM = 1.0
_MODIS_ALBEDO_RESOLUTION_KM = 0.5
_MODIS_SNOW_RESOLUTION_KM = 0.5
_MODIS_NDVI_RESOLUTION_KM = 0.25


class VariableCatalog:
    """Registry of every satellite variable SuSSE ingests.

    Two class-level lists hold the catalogue. The convention is:

    * The :attr:`IRRADIANCE_VARIABLE_IDS` set names the variables routed to
      the wide ``irradiance_daily`` table; everything else goes to the
      long-format companion.
    * Sources are listed separately so the same variable_id can exist for
      multiple sources (``ghi`` is produced by both NASA POWER and CAMS).
    """

    IRRADIANCE_VARIABLE_IDS: ClassVar[frozenset[str]] = frozenset({"ghi", "dhi", "dni"})

    NASA_POWER_VARIABLES: ClassVar[tuple[VariableSpec, ...]] = (
        # Irradiance — go to irradiance_daily.
        VariableSpec(
            variable_id="ghi",
            source=Source.NASA_POWER,
            api_code="ALLSKY_SFC_SW_DWN",
            display_name="All-sky GHI",
            unit="kWh/m^2/day",
            native_unit="kWh/m^2/day",
            description="All-sky surface shortwave downward irradiance.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="dhi",
            source=Source.NASA_POWER,
            api_code="ALLSKY_SFC_SW_DIFF",
            display_name="All-sky DHI",
            unit="kWh/m^2/day",
            native_unit="kWh/m^2/day",
            description="All-sky diffuse horizontal irradiance.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="dni",
            source=Source.NASA_POWER,
            api_code="ALLSKY_SFC_SW_DNI",
            display_name="All-sky DNI",
            unit="kWh/m^2/day",
            native_unit="kWh/m^2/day",
            description="All-sky direct normal irradiance.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        # Auxiliary — go to nasa_daily_vars_long.
        VariableSpec(
            variable_id="temperature",
            source=Source.NASA_POWER,
            api_code="T2M",
            display_name="2m Air Temperature",
            unit="degC",
            native_unit="degC",
            description="Daily near-surface air temperature.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="temperature_range",
            source=Source.NASA_POWER,
            api_code="T2M_RANGE",
            display_name="2m Air Temperature Range",
            unit="degC",
            native_unit="degC",
            description="Daily range (max - min) of 2m air temperature.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="specific_humidity",
            source=Source.NASA_POWER,
            api_code="QV2M",
            display_name="2m Specific Humidity",
            unit="kg/kg",
            native_unit="kg/kg",
            description="2m specific humidity.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="relative_humidity",
            source=Source.NASA_POWER,
            api_code="RH2M",
            display_name="2m Relative Humidity",
            unit="%",
            native_unit="%",
            description="2m relative humidity.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="surface_pressure",
            source=Source.NASA_POWER,
            api_code="PS",
            display_name="Surface Pressure",
            unit="kPa",
            native_unit="kPa",
            description="Daily surface-level pressure.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="aod_550",
            source=Source.NASA_POWER,
            api_code="AOD_55",
            display_name="AOD @ 550nm",
            unit="unitless",
            native_unit="unitless",
            description="Aerosol Optical Depth at 550 nm.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="aod_550_adj",
            source=Source.NASA_POWER,
            api_code="AOD_55_ADJ",
            display_name="AOD @ 550nm (adjusted)",
            unit="unitless",
            native_unit="unitless",
            description="Aerosol Optical Depth at 550 nm, terrain-adjusted.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="aod_840",
            source=Source.NASA_POWER,
            api_code="AOD_84",
            display_name="AOD @ 840nm",
            unit="unitless",
            native_unit="unitless",
            description="Aerosol Optical Depth at 840 nm.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="cloud_visible_optical_depth",
            source=Source.NASA_POWER,
            api_code="CLOUD_OD",
            display_name="Cloud Visible Optical Depth",
            unit="unitless",
            native_unit="unitless",
            description="Daily mean cloud optical depth at visible wavelengths.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="cloud_amount",
            source=Source.NASA_POWER,
            api_code="CLOUD_AMT",
            display_name="Cloud Amount",
            unit="fraction",
            native_unit="fraction",
            description="Daily mean total cloud amount.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="cloud_amount_dat",
            source=Source.NASA_POWER,
            api_code="CLOUD_AMT_DAY",
            display_name="Cloud Amount (Daytime)",
            unit="fraction",
            native_unit="fraction",
            description="Daytime cloud amount.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="total_column_ozone",
            source=Source.NASA_POWER,
            api_code="TO3",
            display_name="Total Column Ozone",
            unit="Dobson Units",
            native_unit="Dobson Units",
            description="Total column ozone.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="precipitable_water",
            source=Source.NASA_POWER,
            api_code="PW",
            display_name="Precipitable Water",
            unit="cm",
            native_unit="cm",
            description="Total precipitable water in the atmospheric column.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="precipitation_corrected",
            source=Source.NASA_POWER,
            api_code="PRECTOTCORR",
            display_name="Precipitation (Corrected)",
            unit="mm/day",
            native_unit="mm/day",
            description="Bias-corrected daily precipitation.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="surface_albedo",
            source=Source.NASA_POWER,
            api_code="SRF_ALB_ADJ",
            display_name="Surface Albedo (Terrain-Adjusted)",
            unit="unitless",
            native_unit="unitless",
            description="Terrain-adjusted surface albedo.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="all_sky_surface_albedo",
            source=Source.NASA_POWER,
            api_code="ALLSKY_SRF_ALB",
            display_name="All-sky Surface Albedo",
            unit="unitless",
            native_unit="unitless",
            description="All-sky surface albedo.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="clear_sky_albedo",
            source=Source.NASA_POWER,
            api_code="CLRSKY_SRF_ALB",
            display_name="Clear-sky Surface Albedo",
            unit="unitless",
            native_unit="unitless",
            description="Clear-sky surface albedo.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="clearness_index",
            source=Source.NASA_POWER,
            api_code="ALLSKY_KT",
            display_name="Clearness Index",
            unit="unitless",
            native_unit="unitless",
            description="All-sky clearness index (GHI / extraterrestrial).",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
            valid_min=0.0,
            valid_max=1.0,
        ),
        VariableSpec(
            variable_id="longwave_downward_irr",
            source=Source.NASA_POWER,
            api_code="CLRSKY_SFC_LW_DWN",
            display_name="Clear-sky Longwave Downward Irradiance",
            unit="W/m^2",
            native_unit="W/m^2",
            description="Clear-sky surface longwave downward irradiance.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="wind_speed",
            source=Source.NASA_POWER,
            api_code="WS2M",
            display_name="2m Wind Speed",
            unit="m/s",
            native_unit="m/s",
            description="Daily mean 2m wind speed.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="northern_wind",
            source=Source.NASA_POWER,
            api_code="V2M",
            display_name="2m Northward Wind",
            unit="m/s",
            native_unit="m/s",
            description="Daily mean 2m northward (V) wind component.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="surface_roughness",
            source=Source.NASA_POWER,
            api_code="Z0M",
            display_name="Surface Roughness",
            unit="m",
            native_unit="m",
            description="Surface aerodynamic roughness length.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="zero_plane_displacement",
            source=Source.NASA_POWER,
            api_code="DISPH",
            display_name="Zero-Plane Displacement",
            unit="m",
            native_unit="m",
            description="Zero-plane displacement height.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="planetary_boundary",
            source=Source.NASA_POWER,
            api_code="PBLTOP",
            display_name="Planetary Boundary Layer Height",
            unit="m",
            native_unit="m",
            description="Top of the planetary boundary layer.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="surface_air_density",
            source=Source.NASA_POWER,
            api_code="RHOA",
            display_name="Surface Air Density",
            unit="kg/m^3",
            native_unit="kg/m^3",
            description="Surface-level air density.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="airmass",
            source=Source.NASA_POWER,
            api_code="AIRMASS",
            display_name="Airmass",
            unit="unitless",
            native_unit="unitless",
            description="Atmospheric airmass.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="surface_soil_wetness",
            source=Source.NASA_POWER,
            api_code="GWETTOP",
            display_name="Surface Soil Wetness",
            unit="fraction",
            native_unit="fraction",
            description="Top-layer soil wetness fraction.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
            valid_min=0.0,
            valid_max=1.0,
        ),
        VariableSpec(
            variable_id="evaporation_land",
            source=Source.NASA_POWER,
            api_code="EVLAND",
            display_name="Evaporation (Land)",
            unit="mm/day",
            native_unit="mm/day",
            description="Daily evaporation over land.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="evapotranspiration_energy",
            source=Source.NASA_POWER,
            api_code="EVPTRNS",
            display_name="Evapotranspiration Energy",
            unit="W/m^2",
            native_unit="W/m^2",
            description="Energy consumed by evapotranspiration.",
            spatial_resolution_km=_NASA_POWER_RESOLUTION_KM,
        ),
        # NOTE: NASA POWER does *not* serve solar_zenith_angle (SZA) at
        # daily resolution — the API returns the -999 fill value, which
        # the loader filters out, leaving zero rows. Including SZA in
        # the catalog therefore caused the per-station coverage check
        # to think SZA was always missing and re-fetch the entire
        # station on every A6 invocation. Migration A10 removes the
        # corresponding dim_variable row. Re-add only if NASA POWER
        # starts serving SZA daily, which would need verification.
    )

    CAMS_VARIABLES: ClassVar[tuple[VariableSpec, ...]] = (
        # Irradiance — go to irradiance_daily.
        VariableSpec(
            variable_id="ghi",
            source=Source.CAMS,
            api_code="ghi",
            display_name="All-sky GHI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="All-sky global horizontal irradiance, daily total.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="dhi",
            source=Source.CAMS,
            api_code="dhi",
            display_name="All-sky DHI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="All-sky diffuse horizontal irradiance, daily total.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="dni",
            source=Source.CAMS,
            api_code="dni",
            display_name="All-sky DNI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="All-sky direct normal irradiance, daily total.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        # Auxiliary — go to cams_daily_vars_long.
        VariableSpec(
            variable_id="bhi",
            source=Source.CAMS,
            api_code="bhi",
            display_name="All-sky BHI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="All-sky beam (direct) horizontal irradiance.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="ghi_extra",
            source=Source.CAMS,
            api_code="ghi_extra",
            display_name="Extraterrestrial GHI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="Extraterrestrial GHI on a horizontal plane.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="ghi_clear",
            source=Source.CAMS,
            api_code="ghi_clear",
            display_name="Clear-sky GHI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="Clear-sky GHI from the McClear model.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="bhi_clear",
            source=Source.CAMS,
            api_code="bhi_clear",
            display_name="Clear-sky BHI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="Clear-sky beam horizontal irradiance.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="dhi_clear",
            source=Source.CAMS,
            api_code="dhi_clear",
            display_name="Clear-sky DHI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="Clear-sky diffuse horizontal irradiance.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
        VariableSpec(
            variable_id="dni_clear",
            source=Source.CAMS,
            api_code="dni_clear",
            display_name="Clear-sky DNI",
            unit="kWh/m^2/day",
            native_unit="Wh/m^2 (period)",
            description="Clear-sky direct normal irradiance.",
            spatial_resolution_km=_CAMS_RESOLUTION_KM,
        ),
    )

    MERRA_2_VARIABLES: ClassVar[tuple[VariableSpec, ...]] = (
        # All MERRA-2 entries land in the long-format companion table
        # `merra_daily_vars_long`; none are routed to `irradiance_daily`
        # (MERRA-2 doesn't expose a direct GHI estimate the way POWER or
        # CAMS do). Hourly source values are aggregated to one daily value
        # per (date, point) by cosine-zenith-weighted mean — see
        # `MerraStreamFetcher` for the aggregation logic.
        VariableSpec(
            variable_id="aod_550_extinction",
            source=Source.MERRA_2,
            api_code="TOTEXTTAU",
            display_name="Total Aerosol Extinction AOT @ 550 nm",
            unit="unitless",
            native_unit="unitless",
            description=(
                "Total aerosol extinction optical thickness at 550 nm "
                "(daily mean, cosine-zenith-weighted from hourly tavg1_2d_aer_Nx)."
            ),
            spatial_resolution_km=_MERRA_2_RESOLUTION_KM,
            valid_min=0.0,
        ),
        VariableSpec(
            variable_id="aod_550_scattering",
            source=Source.MERRA_2,
            api_code="TOTSCATAU",
            display_name="Total Aerosol Scattering AOT @ 550 nm",
            unit="unitless",
            native_unit="unitless",
            description=(
                "Total aerosol scattering optical thickness at 550 nm. "
                "Combined with `aod_550_extinction` gives single-scattering "
                "albedo (SSA = scattering / extinction), which discriminates "
                "absorbing aerosols (dust, smoke) from scattering ones "
                "(sulfate, sea salt). Daily mean, cosine-zenith-weighted "
                "from hourly tavg1_2d_aer_Nx."
            ),
            spatial_resolution_km=_MERRA_2_RESOLUTION_KM,
            valid_min=0.0,
        ),
        VariableSpec(
            variable_id="aod_550_analysis",
            source=Source.MERRA_2,
            api_code="AODANA",
            display_name="Aerosol Optical Depth (Analysis)",
            unit="unitless",
            native_unit="unitless",
            description=(
                "MERRA-2 aerosol analysis AOD field. Independent of POWER's "
                "AOD and of TOTEXTTAU; useful as a cross-check. Daily mean, "
                "cosine-zenith-weighted from 3-hourly inst3_2d_gas_Nx."
            ),
            spatial_resolution_km=_MERRA_2_RESOLUTION_KM,
            valid_min=0.0,
        ),
        VariableSpec(
            variable_id="precipitable_water",
            source=Source.MERRA_2,
            api_code="TQV",
            display_name="Total Precipitable Water Vapour",
            unit="kg/m^2",
            native_unit="kg/m^2",
            description=(
                "Total column precipitable water vapour. Same physical "
                "quantity as NASA POWER's `precipitable_water` (cm), provided "
                "here from MERRA-2 directly for cross-source comparison. "
                "Daily mean, cosine-zenith-weighted from hourly tavg1_2d_slv_Nx."
            ),
            spatial_resolution_km=_MERRA_2_RESOLUTION_KM,
            valid_min=0.0,
        ),
    )

    MODIS_VARIABLES: ClassVar[tuple[VariableSpec, ...]] = (
        # MODIS rows land in `modis_observations` (not the long-format
        # tables) because each value is associated with a (product_id,
        # band_id) pair that the long-format schema doesn't carry. The
        # `variable_id` here uses the convention `{PRODUCT}_{BAND}` so
        # `dim_variable` still has a unique key per row, and the
        # underlying observations table preserves product_id and band_id
        # as separate columns for downstream querying.
        #
        # Product list constrained by what ORNL DAAC's RST subset API
        # actually serves (verified 2026-05-08 against the live /products
        # endpoint). MCD43A3 (true albedo), MOD11A1 (daily LST), and
        # MOD10A1 (snow cover) were originally planned but are not on
        # this endpoint — they require AppEEARS or LAADS DAAC integration
        # which is deferred to a future phase. The three below are the
        # closest practical substitutes from ORNL's catalog.
        VariableSpec(
            variable_id="MCD43A4_Nadir_Reflectance_Band1",
            source=Source.MODIS,
            api_code="MCD43A4",
            display_name="MODIS Nadir BRDF-Adjusted Reflectance, Band 1 (Red)",
            unit="unitless",
            native_unit="unitless",
            description=(
                "Daily 500 m nadir BRDF-adjusted reflectance in MODIS "
                "band 1 (red, 620-670 nm). Surface-state proxy; "
                "alternative to direct broadband albedo (MCD43A3, not "
                "available at ORNL DAAC)."
            ),
            spatial_resolution_km=_MODIS_ALBEDO_RESOLUTION_KM,
            valid_min=0.0,
            valid_max=1.0,
        ),
        VariableSpec(
            variable_id="MOD11A2_LST_Day_1km",
            source=Source.MODIS,
            api_code="MOD11A2",
            display_name="MODIS Land Surface Temperature (Day, 8-day, 1 km)",
            unit="K",
            native_unit="K",
            description=(
                "Daytime land surface temperature from MOD11A2, 8-day "
                "composite at 1 km. The daily-cadence MOD11A1 is not "
                "available at ORNL DAAC; this 8-day composite is the "
                "best LST signal we can pull from this endpoint."
            ),
            spatial_resolution_km=_MODIS_LST_RESOLUTION_KM,
            valid_min=0.0,
            valid_max=400.0,
        ),
        VariableSpec(
            variable_id="MOD13Q1_250m_16_days_NDVI",
            source=Source.MODIS,
            api_code="MOD13Q1",
            display_name="MODIS Normalised Difference Vegetation Index",
            unit="unitless",
            native_unit="unitless",
            description=(
                "16-day NDVI composite from MOD13Q1 at 250 m. Land-cover "
                "proxy; useful as a slow-changing categorical-ish feature "
                "for surface-type discrimination."
            ),
            spatial_resolution_km=_MODIS_NDVI_RESOLUTION_KM,
            valid_min=-0.2,
            valid_max=1.0,
        ),
    )

    @classmethod
    def all_variables(cls) -> tuple[VariableSpec, ...]:
        return (
            cls.NASA_POWER_VARIABLES
            + cls.CAMS_VARIABLES
            + cls.MERRA_2_VARIABLES
            + cls.MODIS_VARIABLES
        )

    @classmethod
    def for_source(cls, source: Source) -> tuple[VariableSpec, ...]:
        return tuple(v for v in cls.all_variables() if v.source is source)

    @classmethod
    def auxiliary_for_source(cls, source: Source) -> tuple[VariableSpec, ...]:
        return tuple(
            v
            for v in cls.for_source(source)
            if v.variable_id not in cls.IRRADIANCE_VARIABLE_IDS
        )

    @classmethod
    def irradiance_for_source(cls, source: Source) -> tuple[VariableSpec, ...]:
        return tuple(
            v
            for v in cls.for_source(source)
            if v.variable_id in cls.IRRADIANCE_VARIABLE_IDS
        )

    @classmethod
    def get(cls, *, variable_id: str, source: Source) -> VariableSpec:
        for v in cls.all_variables():
            if v.variable_id == variable_id and v.source is source:
                return v
        raise KeyError(
            f"No variable '{variable_id}' for source {source.value}. "
            f"Add an entry to VariableCatalog or check the spelling."
        )


def variables_to_dataframe(variables: tuple[VariableSpec, ...]) -> pd.DataFrame:
    """Project a tuple of VariableSpec into a ``dim_variable``-shaped DataFrame."""
    return pd.DataFrame(
        [
            {
                "variable_id": v.variable_id,
                "source": v.source.value,
                "display_name": v.display_name,
                "unit": v.unit,
                "native_unit": v.native_unit,
                "description": v.description,
                "temporal_granularity": v.temporal_granularity,
                "spatial_resolution_km": v.spatial_resolution_km,
                "valid_min": v.valid_min,
                "valid_max": v.valid_max,
            }
            for v in variables
        ]
    )


def populate_dim_variable(
    bq: BigQueryClient, *, table_fqn: str, variables: tuple[VariableSpec, ...] | None = None
) -> int:
    """Upsert variable catalogue rows into the ``dim_variable`` table.

    Idempotent on ``(variable_id, source)``. Pass ``variables=None`` to
    upsert the entire registered catalogue (NASA POWER + CAMS).

    Returns the number of rows written.
    """
    from .loaders import MergeLoader, MergeSpec

    variables = variables if variables is not None else VariableCatalog.all_variables()
    df = variables_to_dataframe(variables)
    if df.empty:
        _logger.info("populate_dim_variable: no variables to write.")
        return 0

    loader = MergeLoader(
        bq=bq,
        table_fqn=table_fqn,
        spec=MergeSpec(schema=TableSchemas.DIM_VARIABLE),
    )
    written = loader.load(df)
    _logger.info("populate_dim_variable: upserted %d row(s) into %s.", written, table_fqn)
    return written
