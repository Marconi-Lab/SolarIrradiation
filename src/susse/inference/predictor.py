"""Predictor — portal-facing inference for a TrainedBundle.

Composes the existing primitives (:class:`FeatureService`,
:class:`Preprocessor`, :class:`TrainedBundle`) into the single
deployment-friendly API the portal needs: given a list of coordinates
plus a date range, return a long-format DataFrame of bias-corrected
daily GHI predictions.

The implementation pulls features from the warehouse via
:meth:`FeatureService.build_satellite_features`, applies the bundle's
``FeatureSpec`` through :meth:`Preprocessor.apply_to_dataframe`
(inference shape — no cleaners, no NaN-drop), and runs the regressor
in a single batched call.

Cache misses (geohash5 cells absent from the warehouse for some of
the requested dates) are surfaced loudly by default. With
``on_cache_miss="fetch"``, the predictor instead reuses
:class:`NasaPowerSatelliteJob` and :class:`CamsSatelliteJob` to
synchronously fetch + persist the missing cells before re-querying
and predicting. The CAMS API has a daily quota; the predictor caps
its on-demand fetches at :data:`MAX_CAMS_CALLS_PER_PREDICT` per
invocation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Sequence

import pandas as pd
import pygeohash

from .. import schema
from ..datasets import FeatureSelection, FeatureService
from ..preprocessing import Preprocessor
from ..training import TrainedBundle, load_bundle
from ..warehouse_ops.io.bq import BigQueryClient
from ..warehouse_ops.io.config import WarehouseConfig
from ..warehouse_ops.population.dim_variable import VariableCatalog
from ..warehouse_ops.population.jobs.satellite_job import (
    CamsSatelliteJob,
    NasaPowerSatelliteJob,
)
from ..warehouse_ops.population.types import (
    DateRange,
    LocationSpec,
    NamedLocationsPlan,
    Source,
    VariableSpec,
)

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ..preprocessing.elevation import ElevationProvider

_PREDICTION_COLUMN: str = schema.PREDICTION

MAX_CAMS_CALLS_PER_PREDICT: int = 30
"""Hard cap on per-cell CAMS fetches in a single :meth:`Predictor.predict`
invocation. CAMS free-tier quota is ~40 requests/day; the cap leaves
headroom for retries and for other CAMS work in the same calendar day.
Hit this cap by widening the warehouse via a pre-cache migration."""

CacheMissPolicy = Literal["raise", "fetch"]


@dataclass(frozen=True)
class PredictionRequest:
    """Coordinates + date range to predict over, validated at construction.

    Attributes:
        coords: One ``(lat, lon)`` per location to predict. Duplicates
            are permitted (e.g. multiple named locations sharing one
            geohash5 cell) — the predictor expands them back per coord
            after the warehouse query.
        start_date: Inclusive start of the prediction window.
        end_date: Inclusive end of the prediction window.
    """

    coords: tuple[tuple[float, float], ...]
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if not self.coords:
            raise ValueError(
                "PredictionRequest.coords is empty. Pass at least one "
                "(lat, lon) tuple."
            )
        for lat, lon in self.coords:
            if not -90.0 <= lat <= 90.0:
                raise ValueError(
                    f"Latitude {lat} is outside [-90, 90]. Check the "
                    f"coordinate order — Predictor.predict expects "
                    f"(lat, lon) tuples."
                )
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"Longitude {lon} is outside [-180, 180].")
        if self.end_date < self.start_date:
            raise ValueError(
                f"PredictionRequest: end_date {self.end_date} precedes "
                f"start_date {self.start_date}. Swap them or correct "
                f"the input."
            )


class Predictor:
    """Bias-corrected daily GHI predictions for a trained bundle.

    The bundle carries its source manifest (which records the
    :class:`FeatureSelection` used at training time); the predictor
    consults that manifest to know which warehouse sources / variables
    to assemble, so inference and training share an identical feature
    contract by construction.

    Attributes:
        bundle: The trained-model artifact loaded into memory.
        on_cache_miss: ``"raise"`` (default) fails loudly when a
            requested ``(geohash5, date)`` cell is missing from the
            warehouse. ``"fetch"`` is reserved for Phase B (on-demand
            NASA POWER + CAMS fetch) and raises ``NotImplementedError``
            today.
    """

    def __init__(
        self,
        *,
        bundle: TrainedBundle,
        bq: BigQueryClient,
        service: Optional[FeatureService] = None,
        on_cache_miss: CacheMissPolicy = "raise",
    ) -> None:
        self._bundle = bundle
        self._bq = bq
        self._service = service if service is not None else FeatureService(bq)
        self._on_cache_miss: CacheMissPolicy = on_cache_miss
        self._selection: FeatureSelection = bundle.source_manifest.feature_selection
        if on_cache_miss == "fetch":
            self._validate_fetch_mode_credentials()

    @staticmethod
    def _validate_fetch_mode_credentials() -> None:
        """Fail loudly at __init__ when fetch mode lacks the CAMS email."""
        if not os.environ.get("CAMS_EMAIL"):
            raise ValueError(
                "Predictor was constructed with on_cache_miss='fetch' "
                "but CAMS_EMAIL is not set in the environment. CAMS "
                "(SoDa-Pro) requires a registered email; set "
                "`export CAMS_EMAIL=you@example.com` or add it to your "
                ".env file before constructing the predictor. NASA "
                "POWER is public and needs no credentials. BQ write "
                "permission is also required (failures surface at "
                "first write)."
            )

    @classmethod
    def from_bundle_dir(
        cls,
        bundle_dir: Path,
        *,
        config: Optional[WarehouseConfig] = None,
        providers: Optional[Mapping[str, "ElevationProvider"]] = None,
        on_cache_miss: CacheMissPolicy = "raise",
    ) -> "Predictor":
        """Convenience constructor: load the bundle + open a BigQuery client.

        Args:
            bundle_dir: Path to a directory written by
                :meth:`TrainedBundle.save`.
            config: Warehouse config; defaults to the project-default
                :class:`WarehouseConfig()`.
            providers: Mapping of derived-feature kind → provider, passed
                through to :func:`susse.training.load_bundle`.
                Typically ``{"altitude": PvlibElevationProvider()}``.
            on_cache_miss: See class attribute.
        """
        cfg = config if config is not None else WarehouseConfig()
        bq = BigQueryClient(config=cfg)
        bundle = load_bundle(bundle_dir, providers=dict(providers or {}))
        return cls(bundle=bundle, bq=bq, on_cache_miss=on_cache_miss)

    @property
    def bundle(self) -> TrainedBundle:
        return self._bundle

    @property
    def on_cache_miss(self) -> CacheMissPolicy:
        return self._on_cache_miss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        *,
        coords: Sequence[tuple[float, float]],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Daily bias-corrected GHI predictions over ``coords × [start, end]``.

        Args:
            coords: One ``(lat, lon)`` tuple per location to predict.
            start_date, end_date: Inclusive date bounds.

        Returns:
            Long-format DataFrame with columns ``lat``, ``lon``,
            ``geohash5``, ``date``, the bundle's expected feature
            columns (so callers can overlay raw satellite baselines
            without re-querying), and ``y_pred_kwh_m2_day`` — one row
            per ``(coord, date)`` pair.

        Raises:
            ValueError: For invalid inputs (empty coords, swapped
                date order, out-of-range lat/lon).
            RuntimeError: When ``on_cache_miss="raise"`` and the
                warehouse is missing any requested ``(geohash5, date)``
                cell. The error names the missing cells.
            NotImplementedError: When ``on_cache_miss="fetch"`` is
                requested — Phase B will add the on-demand fetch path.
        """
        request = PredictionRequest(
            coords=tuple(coords),
            start_date=start_date,
            end_date=end_date,
        )
        coords_df = self._coords_to_dataframe(request.coords)
        unique_geohashes = tuple(coords_df[schema.GEOHASH5].unique())
        warehouse_df = self._fetch_warehouse_features(
            geohash5s=unique_geohashes,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        missing = self._detect_cache_misses(
            warehouse_df=warehouse_df,
            geohash5s=unique_geohashes,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        if missing:
            if self._on_cache_miss == "raise":
                self._raise_cache_miss(missing)
            else:  # "fetch"
                missing_geohashes = {gh for gh, _ in missing}
                self._fetch_and_persist(
                    coords_df=coords_df,
                    missing_geohash5s=missing_geohashes,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )
                # Re-query the now-warm warehouse for the fresh rows.
                warehouse_df = self._fetch_warehouse_features(
                    geohash5s=unique_geohashes,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )
                still_missing = self._detect_cache_misses(
                    warehouse_df=warehouse_df,
                    geohash5s=unique_geohashes,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )
                if still_missing:
                    self._raise_cache_miss(
                        still_missing,
                        prefix=(
                            "Even after on-demand fetch, the following "
                            "(geohash5, date) pairs are still missing. "
                            "The upstream API probably returned no data "
                            "for them (e.g. dates outside CAMS's coverage)."
                        ),
                    )
        inference_df = self._merge_coords_with_warehouse(coords_df, warehouse_df)
        spec = self._bundle.feature_spec
        processed = Preprocessor(spec).apply_to_dataframe(inference_df)
        processed[_PREDICTION_COLUMN] = self._bundle.regressor.predict(
            processed[list(spec.output_feature_names)]
        )
        return self._format_output(processed)

    # ------------------------------------------------------------------
    # Internal — coord prep
    # ------------------------------------------------------------------

    def _coords_to_dataframe(
        self,
        coords: Sequence[tuple[float, float]],
    ) -> pd.DataFrame:
        """Compute geohash5 + synthetic location name per coord."""
        precision = self._service.geohash_precision
        rows: list[dict[str, object]] = []
        for i, (lat, lon) in enumerate(coords):
            rows.append(
                {
                    "coord_idx": i,
                    schema.LOCATION: f"point_{i:04d}",
                    schema.LAT: float(lat),
                    schema.LON: float(lon),
                    schema.GEOHASH5: pygeohash.encode(
                        lat, lon, precision=precision
                    ),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal — warehouse query
    # ------------------------------------------------------------------

    def _fetch_warehouse_features(
        self,
        *,
        geohash5s: tuple[str, ...],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Pull irradiance + per-source aux for the requested cells in one go."""
        return self._service.build_satellite_features(
            selection=self._selection,
            date_start=start_date,
            date_end=end_date,
            geohash5s=geohash5s,
        )

    def _detect_cache_misses(
        self,
        *,
        warehouse_df: pd.DataFrame,
        geohash5s: tuple[str, ...],
        start_date: date,
        end_date: date,
    ) -> list[tuple[str, date]]:
        """Return ``(geohash5, date)`` pairs absent from the warehouse frame."""
        if not geohash5s:
            return []
        expected_dates = pd.date_range(start_date, end_date, freq="D").date
        expected = pd.MultiIndex.from_product(
            [geohash5s, expected_dates], names=[schema.GEOHASH5, schema.DATE]
        )
        if warehouse_df.empty:
            present_dates: pd.Series = pd.Series([], dtype="object")
            present_gh: pd.Series = pd.Series([], dtype=str)
        else:
            present_dates = pd.to_datetime(warehouse_df[schema.DATE]).dt.date
            present_gh = warehouse_df[schema.GEOHASH5]
        present = pd.MultiIndex.from_arrays(
            [present_gh, present_dates],
            names=[schema.GEOHASH5, schema.DATE],
        )
        missing = expected.difference(present)
        return list(missing)

    def _raise_cache_miss(
        self,
        missing: list[tuple[str, date]],
        *,
        prefix: str | None = None,
    ) -> None:
        """Build the canonical cache-miss error message."""
        sample = missing[:5]
        head = (
            prefix
            if prefix is not None
            else (
                f"Predictor: {len(missing)} warehouse cell-day pairs are "
                f"missing. Apply the corresponding warehouse-ingest "
                f"migration before calling predict, or construct the "
                f"Predictor with on_cache_miss='fetch' to fetch on demand "
                f"(requires CAMS_EMAIL + BQ write perms)."
            )
        )
        raise RuntimeError(f"{head} First few missing (geohash5, date): {sample}.")

    def _fetch_and_persist(
        self,
        *,
        coords_df: pd.DataFrame,
        missing_geohash5s: set[str],
        start_date: date,
        end_date: date,
    ) -> None:
        """Run the existing ingest jobs over the missing cells.

        Reuses :class:`NasaPowerSatelliteJob` and :class:`CamsSatelliteJob`
        directly so the on-demand fetch path is bit-for-bit identical to
        the batch-ingest path: same coverage check, same variable-id
        mapping, same warehouse schema, same idempotency guarantees.
        """
        if len(missing_geohash5s) > MAX_CAMS_CALLS_PER_PREDICT:
            raise RuntimeError(
                f"Predictor: {len(missing_geohash5s)} cells need an "
                f"on-demand CAMS fetch, exceeding the per-invocation "
                f"cap of {MAX_CAMS_CALLS_PER_PREDICT} (CAMS free-tier "
                f"daily quota). Pre-cache this region via a warehouse "
                f"ingest migration before calling predict on it."
            )
        # Recover (lat, lon, name) for each missing geohash5 from the
        # coords_df. Multiple coords sharing one geohash5 contribute one
        # LocationSpec.
        gh_to_coord = coords_df.drop_duplicates(subset=schema.GEOHASH5).set_index(
            schema.GEOHASH5
        )[[schema.LAT, schema.LON, schema.LOCATION]]
        locations = tuple(
            LocationSpec(
                name=str(gh_to_coord.loc[gh, schema.LOCATION]),
                lat=float(gh_to_coord.loc[gh, schema.LAT]),
                lon=float(gh_to_coord.loc[gh, schema.LON]),
            )
            for gh in sorted(missing_geohash5s)
        )
        date_range = DateRange(start=start_date, end=end_date)
        nasa_vars = self._variables_for_source(Source.NASA_POWER)
        if nasa_vars:
            NasaPowerSatelliteJob(self._bq).run(
                NamedLocationsPlan(
                    source=Source.NASA_POWER,
                    date_range=date_range,
                    locations=locations,
                    variables=nasa_vars,
                ),
            )
        cams_vars = self._variables_for_source(Source.CAMS)
        if cams_vars:
            CamsSatelliteJob(self._bq).run(
                NamedLocationsPlan(
                    source=Source.CAMS,
                    date_range=date_range,
                    locations=locations,
                    variables=cams_vars,
                ),
            )

    def _variables_for_source(self, source: Source) -> tuple[VariableSpec, ...]:
        """Resolve the full :class:`VariableSpec` for each requested ID + band."""
        selection = self._selection
        ids: tuple[str, ...] = ()
        if source is Source.NASA_POWER:
            ids = selection.nasa_variable_ids
        elif source is Source.CAMS:
            ids = selection.cams_variable_ids
        elif source is Source.MERRA_2:
            ids = selection.merra_variable_ids
        aux = tuple(VariableCatalog.get(variable_id=vid, source=source) for vid in ids)
        irradiance: tuple[VariableSpec, ...] = ()
        if source in selection.include_satellite_irradiance:
            band_specs = []
            for band in selection.include_satellite_bands:
                try:
                    band_specs.append(
                        VariableCatalog.get(variable_id=band.value, source=source)
                    )
                except KeyError:
                    # Source / band combination not in catalogue — skip.
                    continue
            irradiance = tuple(band_specs)
        return aux + irradiance

    # ------------------------------------------------------------------
    # Internal — assembly + inference
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_coords_with_warehouse(
        coords_df: pd.DataFrame, warehouse_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Cross-join coords × warehouse rows for that geohash5."""
        if warehouse_df.empty:
            return coords_df.assign(date=pd.NaT).iloc[0:0]
        return coords_df.merge(warehouse_df, on=schema.GEOHASH5, how="inner")

    def _format_output(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Order columns + drop internal scaffolding from the result."""
        front = [
            c
            for c in (schema.LAT, schema.LON, schema.GEOHASH5, schema.DATE, schema.LOCATION)
            if c in processed_df.columns
        ]
        rest = [
            c for c in processed_df.columns if c not in front + [_PREDICTION_COLUMN]
        ]
        out = processed_df[front + rest + [_PREDICTION_COLUMN]].copy()
        return out.reset_index(drop=True)
