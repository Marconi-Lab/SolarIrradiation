"""MERRA-2 data access — Earthdata-authenticated OPeNDAP fetcher."""

from .merra_config import Merra2Config
from .merra_daily_fetcher import MerraAuthError, MerraDailyFetcher, cos_zenith_aggregate
from .merra_product import MerraProducts

__all__ = [
    "Merra2Config",
    "MerraAuthError",
    "MerraDailyFetcher",
    "MerraProducts",
    "cos_zenith_aggregate",
]
