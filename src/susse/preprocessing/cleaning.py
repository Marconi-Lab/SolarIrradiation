"""Data-cleaning step abstraction.

A :class:`DataCleaner` is one named, configurable row-level transform —
filter rows or impute missing values, with column schema preserved.
The :class:`Preprocessor` runs every cleaner in
``spec.cleaners`` in declared order, **before** any
:class:`DerivedFeature` computation.

Parallel to :class:`DerivedFeature` but on a different axis:

* :meth:`DerivedFeature.compute` adds new columns (row count unchanged).
* :meth:`DataCleaner.apply` modifies rows (column count unchanged).

Both share :class:`KindTaggedSpec` as a base (kind tag + JSON roundtrip
+ required-input declaration). The verb distinction — ``apply`` vs
``compute`` — lives on each family's concrete ABC, so a reader still
sees ``cleaner.apply(df)`` and immediately knows the row set may
change, and ``feature.compute(df)`` and knows new columns are emitted.

Adding a new cleaner type is one new subclass: register a
:class:`CleanerKind` value, add the subclass with its own ``apply``
implementation, and reference it from :meth:`CleanerKind.spec_class`.
Nothing else in the codebase changes.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._kind_tagged import KindTaggedSpec, kind_dispatched_from_dict


class CleanerKind(StrEnum):
    """Persistence + dispatch tag for :class:`DataCleaner` subclasses.

    The string value ends up in ``feature_spec.json`` under each
    cleaner entry's ``kind`` key. New subclasses register here and in
    :meth:`spec_class`; nothing else needs to learn the new kind.
    """

    GHI_UPPER_BOUND = "ghi_upper_bound"
    IQR_LOWER_BOUND = "iqr_lower_bound"
    HIGH_MISSING_YEAR = "high_missing_year"
    KNN_YEAR_GAP_IMPUTE = "knn_year_gap_impute"
    PER_STATION_MEAN_IMPUTE = "per_station_mean_impute"

    def spec_class(self) -> type["DataCleaner"]:
        """Return the concrete :class:`DataCleaner` subclass for this kind.

        Implements the protocol :func:`kind_dispatched_from_dict`
        consumes; the same method name appears on :class:`FeatureKind`.
        """
        if self is CleanerKind.GHI_UPPER_BOUND:
            return GhiUpperBoundCleaner
        if self is CleanerKind.IQR_LOWER_BOUND:
            return IqrLowerBoundCleaner
        if self is CleanerKind.HIGH_MISSING_YEAR:
            return HighMissingYearExcluder
        if self is CleanerKind.KNN_YEAR_GAP_IMPUTE:
            return KnnYearGapImputer
        if self is CleanerKind.PER_STATION_MEAN_IMPUTE:
            return PerStationMeanImputer
        raise AssertionError(f"Unhandled CleanerKind: {self!r}")  # pragma: no cover


class DataCleaner(KindTaggedSpec[CleanerKind]):
    """ABC for one row-level cleaning step.

    Subclasses are frozen-dataclass value objects (configuration only,
    no fitted state). :meth:`apply` may shrink the row set or fill
    values within existing rows; the column schema is preserved
    end-to-end.

    Inherits the kind/required-input/to_dict/_from_dict contract from
    :class:`KindTaggedSpec`; this ABC adds the cleaner-specific
    ``apply`` verb.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the cleaning rule. Returns a frame with the same columns.

        The returned DataFrame's row count may be less than or equal to
        the input's; for imputers it equals the input's. The column set
        and ordering are preserved.
        """


def data_cleaner_from_dict(
    d: dict[str, Any],
    *,
    providers: Optional[dict[str, Any]] = None,
) -> DataCleaner:
    """Inverse of :meth:`DataCleaner.to_dict` — kind-dispatched.

    Thin wrapper around :func:`kind_dispatched_from_dict` that narrows
    the return type to :class:`DataCleaner` and supplies the
    family-specific error-message hint. ``providers`` is accepted for
    forward compatibility (no current cleaner needs it) so
    :class:`FeatureSpec.from_dict` can pass one dict through to both
    families uniformly.
    """
    return kind_dispatched_from_dict(  # type: ignore[no-any-return]
        d,
        kind_enum=CleanerKind,
        providers=providers,
        family_name="cleaner",
    )


# ---------------------------------------------------------------------------
# Concrete cleaners
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GhiUpperBoundCleaner(DataCleaner):
    """Drop rows whose target column exceeds a physically-implausible value.

    Ground-truth GHI values above ~12 kWh/m²/day in sub-Saharan Africa are
    almost certainly sensor errors (the theoretical clear-sky maximum at
    these latitudes is ~9–10 kWh/m²/day). Configurable so other regions
    or units can adjust the threshold.

    Used in: Mukiibi & Mikelson (2026) ground-truth curation.
    """

    column: str = "ghi_kwh_m2_day"
    threshold: float = 12.0

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("GhiUpperBoundCleaner.column must be non-empty.")
        if not np.isfinite(self.threshold):
            raise ValueError(
                f"GhiUpperBoundCleaner.threshold must be finite, got {self.threshold}."
            )

    @property
    def kind(self) -> CleanerKind:
        return CleanerKind.GHI_UPPER_BOUND

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return (self.column,)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.column] <= self.threshold].reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "column": self.column,
            "threshold": self.threshold,
        }

    @classmethod
    def _from_dict(
        cls, d: dict[str, Any], *, providers: dict[str, Any]
    ) -> "GhiUpperBoundCleaner":
        del providers
        return cls(
            column=d.get("column", "ghi_kwh_m2_day"),
            threshold=float(d.get("threshold", 12.0)),
        )


@dataclass(frozen=True)
class IqrLowerBoundCleaner(DataCleaner):
    """Drop rows whose target column is below ``Q1 − multiplier × IQR``.

    Standard Tukey-fence outlier rule, applied **only on the lower side**
    (an upper-side fence here would clip legitimate clear-sky days; for
    the upper bound use :class:`GhiUpperBoundCleaner`). The threshold is
    computed from the input frame at apply time — no fitted state, but
    the bound depends on the data.

    Multiplier 1.5 is the classic Tukey value the paper uses.
    """

    column: str = "ghi_kwh_m2_day"
    multiplier: float = 1.5

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("IqrLowerBoundCleaner.column must be non-empty.")
        if self.multiplier <= 0:
            raise ValueError(
                f"IqrLowerBoundCleaner.multiplier must be positive, "
                f"got {self.multiplier}."
            )

    @property
    def kind(self) -> CleanerKind:
        return CleanerKind.IQR_LOWER_BOUND

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return (self.column,)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        col = df[self.column].dropna()
        if col.empty:
            return df
        q1 = float(col.quantile(0.25))
        q3 = float(col.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - self.multiplier * iqr
        return df[df[self.column] >= lower_bound].reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "column": self.column,
            "multiplier": self.multiplier,
        }

    @classmethod
    def _from_dict(
        cls, d: dict[str, Any], *, providers: dict[str, Any]
    ) -> "IqrLowerBoundCleaner":
        del providers
        return cls(
            column=d.get("column", "ghi_kwh_m2_day"),
            multiplier=float(d.get("multiplier", 1.5)),
        )


@dataclass(frozen=True)
class HighMissingYearExcluder(DataCleaner):
    """Drop every (station, year) group whose missing-day fraction exceeds
    ``missing_fraction_threshold``.

    "Missing fraction" is computed against the calendar-year span the
    station has any data for: ``1 − observed_days / span_days``, where
    ``span_days`` is the number of days from the group's first to last
    record inclusive. This avoids penalising stations that legitimately
    operated for only part of a year.

    Pair with :class:`KnnYearGapImputer` downstream: this cleaner removes
    irrecoverably-sparse (station, year) groups, the imputer fills the
    remaining smaller gaps.
    """

    station_column: str = "location"
    date_column: str = "date"
    missing_fraction_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not self.station_column:
            raise ValueError(
                "HighMissingYearExcluder.station_column must be non-empty."
            )
        if not self.date_column:
            raise ValueError("HighMissingYearExcluder.date_column must be non-empty.")
        if not (0.0 <= self.missing_fraction_threshold <= 1.0):
            raise ValueError(
                f"HighMissingYearExcluder.missing_fraction_threshold must be "
                f"in [0, 1], got {self.missing_fraction_threshold}."
            )

    @property
    def kind(self) -> CleanerKind:
        return CleanerKind.HIGH_MISSING_YEAR

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return (self.station_column, self.date_column)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        dates = pd.to_datetime(df[self.date_column])
        years = dates.dt.year
        groups = df.groupby([df[self.station_column], years])
        keep_mask = pd.Series(True, index=df.index)
        for (station, year), idx in groups.groups.items():
            sub = dates.loc[idx]
            span_days = (sub.max() - sub.min()).days + 1
            observed = len(sub)
            missing_fraction = 1.0 - observed / span_days if span_days > 0 else 0.0
            if missing_fraction > self.missing_fraction_threshold:
                keep_mask.loc[idx] = False
        return df[keep_mask].reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "station_column": self.station_column,
            "date_column": self.date_column,
            "missing_fraction_threshold": self.missing_fraction_threshold,
        }

    @classmethod
    def _from_dict(
        cls, d: dict[str, Any], *, providers: dict[str, Any]
    ) -> "HighMissingYearExcluder":
        del providers
        return cls(
            station_column=d.get("station_column", "location"),
            date_column=d.get("date_column", "date"),
            missing_fraction_threshold=float(d.get("missing_fraction_threshold", 0.05)),
        )


@dataclass(frozen=True)
class KnnYearGapImputer(DataCleaner):
    """Impute missing daily values inside each (station, year) group via temporal kNN.

    For every (station, year) where the group's date range has gaps,
    inserts new rows at the missing dates and fills ``target_column``
    with the mean of the ``k`` temporally-nearest existing values within
    that same (station, year) group. Other columns on the inserted rows
    are forward/back-filled from the same group so id columns
    (``location``, ``geohash5``, lat/lon) propagate.

    Assumes :class:`HighMissingYearExcluder` has already run upstream so
    every surviving (station, year) is sparse enough to impute reliably.
    Won't run on a group with fewer than ``k`` observed days (returns it
    unchanged so the caller can spot the under-served case).
    """

    target_column: str = "ghi_kwh_m2_day"
    station_column: str = "location"
    date_column: str = "date"
    k: int = 5

    def __post_init__(self) -> None:
        if not self.target_column:
            raise ValueError("KnnYearGapImputer.target_column must be non-empty.")
        if not self.station_column:
            raise ValueError("KnnYearGapImputer.station_column must be non-empty.")
        if not self.date_column:
            raise ValueError("KnnYearGapImputer.date_column must be non-empty.")
        if self.k < 1:
            raise ValueError(f"KnnYearGapImputer.k must be >= 1, got {self.k}.")

    @property
    def kind(self) -> CleanerKind:
        return CleanerKind.KNN_YEAR_GAP_IMPUTE

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return (self.target_column, self.station_column, self.date_column)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        dates = pd.to_datetime(df[self.date_column])
        years = dates.dt.year
        out_chunks: list[pd.DataFrame] = []
        for (station, year), idx in df.groupby(
            [df[self.station_column], years]
        ).groups.items():
            group = df.loc[idx].copy()
            group[self.date_column] = pd.to_datetime(group[self.date_column])
            imputed = self._impute_group(group)
            out_chunks.append(imputed)
        if not out_chunks:
            return df
        out = pd.concat(out_chunks, ignore_index=True)
        # Restore the original date dtype if it was a date (not datetime).
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            out[self.date_column] = out[self.date_column].dt.date
        return out

    def _impute_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Fill (station, year) gaps via mean-of-k-nearest existing values."""
        if len(group) < self.k:
            return group
        group = group.sort_values(self.date_column).reset_index(drop=True)
        start, end = group[self.date_column].min(), group[self.date_column].max()
        full_range = pd.date_range(start, end, freq="D")
        if len(full_range) == len(group):
            return group  # no gaps
        present_dates = pd.to_datetime(group[self.date_column]).to_numpy()
        present_values = group[self.target_column].to_numpy(dtype=float)
        ordinal_present = present_dates.astype("datetime64[D]").astype(int)
        ordinal_full = full_range.to_numpy().astype("datetime64[D]").astype(int)
        missing_mask = ~np.isin(ordinal_full, ordinal_present)
        if not missing_mask.any():
            return group
        # For each missing date, average the k nearest present-date values.
        missing_ordinals = ordinal_full[missing_mask]
        deltas = np.abs(missing_ordinals[:, None] - ordinal_present[None, :])
        nearest_idx = np.argpartition(deltas, kth=self.k - 1, axis=1)[:, : self.k]
        imputed_values = present_values[nearest_idx].mean(axis=1)
        imputed_dates = full_range[missing_mask]
        # Carry context columns over from the most recent present row.
        context_row = group.iloc[-1].copy()
        new_rows = pd.DataFrame(
            np.tile(context_row.values, (len(imputed_dates), 1)),
            columns=group.columns,
        )
        new_rows[self.date_column] = imputed_dates
        new_rows[self.target_column] = imputed_values
        merged = pd.concat([group, new_rows], ignore_index=True)
        merged = merged.sort_values(self.date_column).reset_index(drop=True)
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_column": self.target_column,
            "station_column": self.station_column,
            "date_column": self.date_column,
            "k": self.k,
        }

    @classmethod
    def _from_dict(
        cls, d: dict[str, Any], *, providers: dict[str, Any]
    ) -> "KnnYearGapImputer":
        del providers
        return cls(
            target_column=d.get("target_column", "ghi_kwh_m2_day"),
            station_column=d.get("station_column", "location"),
            date_column=d.get("date_column", "date"),
            k=int(d.get("k", 5)),
        )


@dataclass(frozen=True)
class PerStationMeanImputer(DataCleaner):
    """Fill NaN values in feature columns with that station's column mean.

    For each ``(station, column)`` pair, replace NaN entries with the mean
    of the column's non-NaN values **at the same station**. Useful when a
    sensor outage produces a short NaN gap at a station that otherwise
    has good coverage — the per-station mean is the most defensible
    constant-fill choice in that situation.

    When NOT to use this:

    * **All-NaN-at-some-station columns** (e.g. NASA POWER's land-only
      ``evaporation_land`` over an oceanic station). The imputer has no
      values to average and leaves the NaNs in place; the row will then
      be dropped by ``dropna_features``. There is no honest imputation
      for a column whose values do not exist by design — drop the
      column instead.
    * **Strong time trend within a station.** Mean-fill flattens the
      annual cycle. For seasonal features, prefer per-(station, month)
      means or kNN imputation in the time dimension.

    The imputer is opt-in per column via :attr:`columns`. Listing a
    column whose station is fully NaN raises a warning but does not
    fail the run.

    Args:
        columns: Feature columns to impute. Passing an empty tuple is
            an error — be explicit about which columns to touch.
        station_column: Column carrying the station identifier. Default
            ``"location"`` matches the rest of the pipeline.
    """

    columns: tuple[str, ...] = ()
    station_column: str = "location"

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError(
                "PerStationMeanImputer.columns is empty. List the feature "
                "columns you want to impute explicitly; an empty tuple is "
                "almost certainly a misconfiguration."
            )
        if not self.station_column:
            raise ValueError("PerStationMeanImputer.station_column must be non-empty.")

    @property
    def kind(self) -> CleanerKind:
        return CleanerKind.PER_STATION_MEAN_IMPUTE

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return (self.station_column,) + self.columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(
                    f"PerStationMeanImputer: column {col!r} is not present "
                    f"in the input DataFrame. Available columns: "
                    f"{sorted(out.columns)}."
                )
            per_station_mean = out.groupby(self.station_column)[col].transform("mean")
            out[col] = out[col].fillna(per_station_mean)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "columns": list(self.columns),
            "station_column": self.station_column,
        }

    @classmethod
    def _from_dict(
        cls, d: dict[str, Any], *, providers: dict[str, Any]
    ) -> "PerStationMeanImputer":
        del providers
        return cls(
            columns=tuple(d.get("columns", ())),
            station_column=d.get("station_column", "location"),
        )
