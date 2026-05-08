"""Variable catalogue and ``dim_variable`` table population.

Defines every satellite variable SuSSE ingests, with the metadata that
goes into the ``dim_variable`` table (units, descriptions, valid ranges,
spatial resolution). New variables are added here once and pick up
across loaders, jobs, and downstream feature assembly.

Variable routing:

* ``ghi``, ``dhi``, ``dni`` (in either source) → wide ``irradiance_daily``.
* Everything else → the long-format companion table for that source
  (``nasa_daily_vars_long`` or ``cams_daily_vars_long``).

The variable_id strings for NASA POWER are kept consistent with what the
existing warehouse already stores (verified 2026-05-08), including the
``arimass`` and ``longwave_downward_irr`` typos. Renaming them is a
future migration task; we don't fix them silently here.
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
            api_code="SRF_ALB",
            display_name="Surface Albedo",
            unit="unitless",
            native_unit="unitless",
            description="Surface albedo.",
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
            description=(
                "Clear-sky surface longwave downward irradiance. "
                "Variable id retains the legacy 'downward → downard' typo "
                "for warehouse compatibility."
            ),
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
            variable_id="arimass",
            source=Source.NASA_POWER,
            api_code="AIRMASS",
            display_name="Airmass",
            unit="unitless",
            native_unit="unitless",
            description=(
                "Atmospheric airmass. Variable id retains the legacy "
                "'airmass → arimass' typo for warehouse compatibility."
            ),
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

    @classmethod
    def all_variables(cls) -> tuple[VariableSpec, ...]:
        return cls.NASA_POWER_VARIABLES + cls.CAMS_VARIABLES

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
