"""Satellite ingest jobs for NASA POWER and CAMS.

A single base class implements the per-pattern lifecycle (resolve
locations → coverage check → fetch → reshape → split into irradiance
vs auxiliary → MERGE-load). Each concrete subclass plugs in a
source-specific fetch + reshape step.

Both ``NamedLocationsPlan`` and ``GridPlan`` flow through the same path
because they only differ in how the location list is resolved — the
inner loop is identical.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import date, datetime, timezone
from typing import ClassVar

import pandas as pd
import pygeohash

from ....api_clients.cams import CAMSClient
from ....api_clients.cams.cams_client import CamsApiError
from ....api_clients.NASA_Power import (
    NASAPowerFetchData,
    NASAPowerProduct,
    TemporalResolution,
)
from ...io.bq import BigQueryClient
from ...io.config import TableRefs, TableSchemas
from ..base_job import BaseJob, JobResult
from ..coverage import CoverageRepository
from ..dim_variable import VariableCatalog
from ..types import (
    DateRange,
    FetchPlan,
    GridPlan,
    LocationSpec,
    NamedLocationsPlan,
    Source,
    VariableSpec,
)
from .satellite_loading import irradiance_long_to_wide, load_irradiance, load_long

try:
    from geopy import Point
except (
    ImportError
):  # pragma: no cover - geopy is a hard dep but the import path is checked at use site
    Point = None  # type: ignore[assignment]


def _location_spec_to_geopy(loc: LocationSpec):
    """Adapter for the NASA POWER fetcher.

    The fetcher only reads ``.latitude`` / ``.longitude`` from the object it
    is passed, both of which :class:`geopy.Point` exposes directly. We used
    to wrap the ``Point`` in a :class:`geopy.location.Location`, but
    ``Location.__init__`` requires ``address``/``raw`` as of geopy 2.4 —
    and we have neither, nor do we need them.
    """
    if Point is None:
        raise RuntimeError(
            "geopy is required for NASA POWER ingest but could not be imported. "
            "Add geopy to requirements.txt and reinstall."
        )
    return Point(loc.lat, loc.lon)


_logger = logging.getLogger(__name__)


class BaseSatelliteJob(BaseJob):
    """Base for satellite-source ingest jobs.

    Subclasses provide the source identity and the per-location fetch+reshape
    that produces a long-format DataFrame (one row per (date, variable, value)).
    """

    def __init__(
        self,
        bq: BigQueryClient,
        *,
        refs: TableRefs | None = None,
        geohash_precision: int = 5,
    ) -> None:
        super().__init__(bq)
        self._refs = refs or TableRefs(config=bq.config)
        self._geohash_precision = geohash_precision

    @property
    @abstractmethod
    def source(self) -> Source:
        """Source tag for this satellite ingest (``Source.NASA_POWER`` etc.)."""

    @property
    @abstractmethod
    def long_table_fqn(self) -> str:
        """FQN of the long-format auxiliary table for this source."""

    @abstractmethod
    def _fetch_long_for_location(
        self,
        location: LocationSpec,
        date_range: DateRange,
        variables: tuple[VariableSpec, ...],
    ) -> pd.DataFrame:
        """Return a long-format DataFrame for one location.

        Required columns: ``date, variable_id, value``. The base class
        adds ``latitude, longitude, geohash5, source`` and routes the
        irradiance subset to ``irradiance_daily``.
        """

    @property
    def name(self) -> str:
        return f"satellite_ingest:{self.source.value}"

    def run(self, plan: FetchPlan) -> JobResult:
        if not isinstance(plan, (NamedLocationsPlan, GridPlan)):
            raise TypeError(
                f"{self.name} accepts NamedLocationsPlan or GridPlan, "
                f"got {type(plan).__name__}."
            )
        if plan.source is not self.source:
            raise ValueError(
                f"{self.name} requires plan.source == {self.source}, "
                f"got {plan.source}."
            )
        started = self._start_time()
        locations = self._resolve_locations(plan)
        date_range = plan.date_range
        all_vars = tuple(plan.variables)
        long_vars = tuple(
            v
            for v in all_vars
            if v.variable_id not in VariableCatalog.IRRADIANCE_VARIABLE_IDS
        )
        irradiance_vars = tuple(
            v
            for v in all_vars
            if v.variable_id in VariableCatalog.IRRADIANCE_VARIABLE_IDS
        )

        coverage = CoverageRepository(self._bq)
        # Pre-compute the geohashes for this plan's locations so coverage
        # queries can be scoped to them. Without this scope, named-location
        # plans whose date range overlaps the existing warehouse footprint
        # pull back millions of irrelevant rows and grind for minutes.
        plan_geohashes = tuple(
            pygeohash.encode(loc.lat, loc.lon, precision=self._geohash_precision)
            for loc in locations
        )
        _logger.info(
            "%s: %d location(s), date %s..%s, %d long-vars, %d irradiance-vars",
            self.name,
            len(locations),
            date_range.start,
            date_range.end,
            len(long_vars),
            len(irradiance_vars),
        )
        existing_long = (
            coverage.existing_long_keys(
                self.long_table_fqn,
                date_range=date_range,
                source=self.source,
                variable_ids=tuple(v.variable_id for v in long_vars),
                geohash5s=plan_geohashes,
            )
            if long_vars
            else set()
        )
        existing_irr = (
            coverage.existing_irradiance_keys(
                self._refs.irradiance_daily,
                date_range=date_range,
                source=self.source,
                geohash5s=plan_geohashes,
            )
            if irradiance_vars
            else set()
        )

        rows_added_long = 0
        rows_added_irr = 0
        skipped_locations = 0
        api_calls = 0

        for loc in locations:
            geohash5 = pygeohash.encode(
                loc.lat, loc.lon, precision=self._geohash_precision
            )
            if self._location_fully_cached(
                geohash5,
                date_range=date_range,
                long_vars=long_vars,
                irradiance_vars=irradiance_vars,
                existing_long=existing_long,
                existing_irr=existing_irr,
            ):
                skipped_locations += 1
                continue

            long_df = self._fetch_long_for_location(loc, date_range, all_vars)
            api_calls += 1
            long_df = self._enrich_long(long_df, loc, geohash5)

            if irradiance_vars:
                irr_df = self._extract_irradiance(long_df, irradiance_vars)
                if not irr_df.empty:
                    rows_added_irr += self._load_irradiance(irr_df)

            if long_vars:
                aux_df = long_df[
                    long_df["variable_id"].isin([v.variable_id for v in long_vars])
                ].copy()
                if not aux_df.empty:
                    rows_added_long += self._load_long(aux_df)

        finished = datetime.now(timezone.utc)
        result = JobResult(
            job_name=self.name,
            plan_summary=plan.describe(),
            started_at=started,
            finished_at=finished,
            rows_added=rows_added_long + rows_added_irr,
            api_calls_made=api_calls,
            extra={
                "rows_added_long": rows_added_long,
                "rows_added_irradiance": rows_added_irr,
                "skipped_locations": skipped_locations,
                "processed_locations": len(locations) - skipped_locations,
            },
        )
        _logger.info("%s | %s", self.name, result.summary())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_locations(
        self, plan: NamedLocationsPlan | GridPlan
    ) -> list[LocationSpec]:
        if isinstance(plan, NamedLocationsPlan):
            return list(plan.locations)
        return list(plan.grid.iter_locations())

    def _location_fully_cached(
        self,
        geohash5: str,
        *,
        date_range: DateRange,
        long_vars: tuple[VariableSpec, ...],
        irradiance_vars: tuple[VariableSpec, ...],
        existing_long: set[tuple],
        existing_irr: set[tuple],
    ) -> bool:
        """True iff every (date, variable) we'd need is already cached."""
        all_dates = pd.date_range(date_range.start, date_range.end, freq="D").date
        for d in all_dates:
            if irradiance_vars and (d, geohash5) not in existing_irr:
                return False
            for v in long_vars:
                if (d, geohash5, v.variable_id) not in existing_long:
                    return False
        return True

    def _enrich_long(
        self, df: pd.DataFrame, loc: LocationSpec, geohash5: str
    ) -> pd.DataFrame:
        out = df.copy()
        out["latitude"] = loc.lat
        out["longitude"] = loc.lon
        out["geohash5"] = geohash5
        out["source"] = self.source.value
        return out

    def _extract_irradiance(
        self, long_df: pd.DataFrame, irradiance_vars: tuple[VariableSpec, ...]
    ) -> pd.DataFrame:
        """Pivot the requested irradiance rows from long → wide."""
        wanted_ids = [v.variable_id for v in irradiance_vars]
        sub = long_df[long_df["variable_id"].isin(wanted_ids)].copy()
        if sub.empty:
            return sub
        return irradiance_long_to_wide(sub)

    def _load_long(self, df: pd.DataFrame) -> int:
        return load_long(
            self._bq,
            table_fqn=self.long_table_fqn,
            schema=self._long_schema,
            df=df,
            context=f"{self.name} long",
        )

    def _load_irradiance(self, df: pd.DataFrame) -> int:
        return load_irradiance(
            self._bq, table_fqn=self._refs.irradiance_daily, df=df
        )

    @property
    def _long_schema(self):
        if self.source is Source.NASA_POWER:
            return TableSchemas.NASA_DAILY_VARS_LONG
        if self.source is Source.CAMS:
            return TableSchemas.CAMS_DAILY_VARS_LONG
        raise ValueError(
            f"No long-format schema registered for source={self.source}. "
            f"Add a branch here when adding a new satellite source."
        )


# ---------------------------------------------------------------------------
# NASA POWER
# ---------------------------------------------------------------------------


class NasaPowerSatelliteJob(BaseSatelliteJob):
    """Ingest job for NASA POWER (daily resolution)."""

    def __init__(
        self,
        bq: BigQueryClient,
        *,
        fetcher: NASAPowerFetchData | None = None,
        refs: TableRefs | None = None,
        geohash_precision: int = 5,
    ) -> None:
        super().__init__(bq, refs=refs, geohash_precision=geohash_precision)
        self._fetcher = fetcher or NASAPowerFetchData()

    @property
    def source(self) -> Source:
        return Source.NASA_POWER

    @property
    def long_table_fqn(self) -> str:
        return self._refs.nasa_daily_vars_long

    def _fetch_long_for_location(
        self,
        location: LocationSpec,
        date_range: DateRange,
        variables: tuple[VariableSpec, ...],
    ) -> pd.DataFrame:
        products = self._variables_to_products(variables)
        result = self._fetcher.fetch_multiple_parameters(
            start_date=datetime.combine(date_range.start, datetime.min.time()),
            end_date=datetime.combine(date_range.end, datetime.min.time()),
            location=_location_spec_to_geopy(location),
            products=products,
            temporal_resolution=TemporalResolution.DAILY,
        )
        return self._result_to_long(result, variables)

    def _variables_to_products(
        self, variables: tuple[VariableSpec, ...]
    ) -> list[NASAPowerProduct]:
        out: list[NASAPowerProduct] = []
        for v in variables:
            try:
                out.append(self._product_for_api_code(v.api_code))
            except KeyError as exc:
                raise ValueError(
                    f"Variable '{v.variable_id}' has api_code '{v.api_code}' "
                    f"but no NASAPowerProduct enum entry uses that code. "
                    f"Update NASAPowerProduct or VariableCatalog."
                ) from exc
        return out

    @staticmethod
    def _product_for_api_code(api_code: str) -> NASAPowerProduct:
        for product in NASAPowerProduct:
            if product.value == api_code:
                return product
        raise KeyError(api_code)

    def _result_to_long(
        self, result, variables: tuple[VariableSpec, ...]
    ) -> pd.DataFrame:
        api_to_var = {v.api_code: v.variable_id for v in variables}
        rows: list[dict] = []
        for product in result.products:
            data = result.get_parameter_data(product)
            if data is None:
                continue
            variable_id = api_to_var.get(product.value)
            if variable_id is None:
                continue
            for ts_str, value in data.items():
                if value is None:
                    continue
                # NASA POWER fill-value is -999 for missing data.
                if value == -999 or value == -999.0:
                    continue
                row_date = datetime.strptime(ts_str, "%Y%m%d").date()
                rows.append(
                    {
                        "date": row_date,
                        "variable_id": variable_id,
                        "value": float(value),
                    }
                )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CAMS
# ---------------------------------------------------------------------------


class CamsSatelliteJob(BaseSatelliteJob):
    """Ingest job for CAMS radiation (daily resolution)."""

    def __init__(
        self,
        bq: BigQueryClient,
        *,
        fetcher: CAMSClient | None = None,
        refs: TableRefs | None = None,
        geohash_precision: int = 5,
    ) -> None:
        super().__init__(bq, refs=refs, geohash_precision=geohash_precision)
        self._fetcher = fetcher or CAMSClient()

    @property
    def source(self) -> Source:
        return Source.CAMS

    @property
    def long_table_fqn(self) -> str:
        return self._refs.cams_daily_vars_long

    # CAMS via pvlib's get_cams returns daily irradiance values as the
    # mean power in W/m² over a 24-hour observation period. The warehouse
    # column convention is total energy in kWh/m²/day, so we multiply by
    # (24 hours / 1000 W/kW) = 0.024. Validated empirically against the
    # legacy CSV ingest path in migration A4.
    _W_M2_TO_KWH_M2_DAY: ClassVar[float] = 0.024

    def _fetch_long_for_location(
        self,
        location: LocationSpec,
        date_range: DateRange,
        variables: tuple[VariableSpec, ...],
    ) -> pd.DataFrame:
        try:
            df, _metadata = self._fetcher.fetch_data(
                latitude=location.lat,
                longitude=location.lon,
                start=datetime.combine(date_range.start, datetime.min.time()),
                end=datetime.combine(date_range.end, datetime.min.time()),
                time_step="1d",
            )
        except CamsApiError:
            raise
        return self._cams_dataframe_to_long(df, variables)

    @classmethod
    def _cams_dataframe_to_long(
        cls, df: pd.DataFrame, variables: tuple[VariableSpec, ...]
    ) -> pd.DataFrame:
        api_to_var = {v.api_code: v.variable_id for v in variables}
        # The pvlib output's first column is timestamp (already ISO string after
        # CAMSClient processing); rename to a known name for melt convenience.
        ts_col = df.columns[0]
        present = [c for c in df.columns if c in api_to_var]
        if not present:
            return pd.DataFrame(columns=["date", "variable_id", "value"])
        long = df[[ts_col] + present].melt(
            id_vars=[ts_col],
            value_vars=present,
            var_name="api_code",
            value_name="value",
        )
        long["date"] = pd.to_datetime(long[ts_col]).dt.date
        long["variable_id"] = long["api_code"].map(api_to_var)
        long = long[["date", "variable_id", "value"]].dropna(subset=["value"])
        long["value"] = long["value"].astype(float) * cls._W_M2_TO_KWH_M2_DAY
        return long.reset_index(drop=True)
