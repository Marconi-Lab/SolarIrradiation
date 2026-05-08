"""Warehouse access layer: typed configs + thin BigQuery client + repositories."""

from .bq import BigQueryClient, BigQueryOptions
from .config import (
    MatchStrategy,
    TableRefs,
    TableSchema,
    TableSchemas,
    WarehouseConfig,
    WarehouseOptions,
)
from .feature_service import FeatureService
from .repositories import GroundRepository, SatelliteRepository

__all__ = [
    "BigQueryClient",
    "BigQueryOptions",
    "FeatureService",
    "GroundRepository",
    "MatchStrategy",
    "SatelliteRepository",
    "TableRefs",
    "TableSchema",
    "TableSchemas",
    "WarehouseConfig",
    "WarehouseOptions",
]
