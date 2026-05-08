from __future__ import annotations
from dataclasses import dataclass
import logging
from datetime import date
import pandas as pd
from ...io.bq import BigQueryClient
from ...io.config import TABLES
from ...io.repositories import GroundRepository, SatelliteRepository
from ..types import LocationSpec, VariableSpec
from ..validators import validate_long_schema
from ..loaders import load_long_with_merge
from ...api_clients.protocols import TimeSeriesFetcher
logger = logging.getLogger(__name__)
@dataclass
class CamsJob:
    bq: BigQueryClient
    fetcher: TimeSeriesFetcher
    variables: list[VariableSpec]
    start: date | None = None
    end: date | None = None
    locations: list[LocationSpec] | None = None
    def run(self) -> None:
        ground = GroundRepository(self.bq)
        sat = SatelliteRepository(self.bq)
        if self.locations is None:
            locs = [LocationSpec(float(r.lat), float(r.lon)) for r in ground.distinct_locations().itertuples(index=False)]
        else:
            locs = self.locations
        latest = sat.latest_date(TABLES.cams_daily_vars_long)
        start = (latest.to_pydatetime().date() if latest else date(2000,1,1)) if self.start is None else self.start
        end = (date.today() if self.end is None else self.end)
        frames = []
        varnames = [v.code or v.name for v in self.variables]
        for loc in locs:
            df = self.fetcher.fetch(start=start, end=end, lat=loc.lat, lon=loc.lon, variables=varnames)
            if "variable" in df.columns and "value" in df.columns:
                tidy = df.copy()
            else:
                id_cols = [c for c in df.columns if c not in varnames]
                tidy = df.melt(id_vars=id_cols, value_vars=varnames, var_name="variable", value_name="value")
            tidy["source"] = "CAMS"
            if "geohash5" not in tidy.columns:
                tidy["geohash5"] = None
            tidy = tidy.rename(columns={"variable":"variable_id"})
            frames.append(tidy)
        if not frames:
            logger.warning("No data fetched for CAMS.")
            return
        out = pd.concat(frames, ignore_index=True)
        for c in ["date","geohash5","variable_id","value","source"]:
            if c not in out.columns:
                out[c] = None
        validate_long_schema(out)
        load_long_with_merge(self.bq, out, TABLES.cams_daily_vars_long)
        logger.info("CAMS load completed: %d rows", len(out))
