"""Derived-feature abstraction.

A :class:`DerivedFeature` is one named, computable model input. Concrete
subclasses know what they produce, what input columns they need, and how
to compute themselves from a DataFrame. The :class:`Preprocessor`
iterates over a tuple of these, in order — there is no per-feature
dispatch in :class:`Preprocessor`, no flags on :class:`FeatureSpec`.
Adding a new derived feature is one new subclass; consumers do not
change.

Design points worth preserving across edits:

* Each subclass is a frozen-dataclass-style value object, JSON
  roundtrippable via :meth:`to_dict` / :meth:`_from_dict`. Fields that
  are not JSON-serialisable (e.g. an :class:`ElevationProvider`) are
  re-injected at deserialisation time via the ``providers`` kwarg.
  The roundtrip contract is inherited from :class:`KindTaggedSpec`,
  shared with :class:`DataCleaner`.
* Multi-column outputs are first-class:
  :class:`CyclicalDayOfYearFeature` produces both ``doy_sin`` and
  ``doy_cos`` from a single ``compute()``.
* Validation lives on the feature, not on :class:`FeatureSpec`. The
  spec only checks that the *combined* output column names don't
  collide with each other or with pass-through ``feature_columns``.
* :class:`FeatureKind` is the persistence + dispatch tag, mirroring the
  :class:`susse.models.ModelKind` pattern used by the model layer.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from ._kind_tagged import KindTaggedSpec, kind_dispatched_from_dict
from .derived_features import clear_sky_index, cyclical_day_of_year

if TYPE_CHECKING:  # pragma: no cover
    from .elevation import ElevationProvider


class FeatureKind(StrEnum):
    """Persistence + dispatch tag for :class:`DerivedFeature` subclasses.

    The string value ends up in ``feature_spec.json`` under the ``kind``
    key on each derived-feature entry. New subclasses register here and
    in the lookup method; nothing else in the codebase needs to learn
    about the new feature type.
    """

    CLEAR_SKY_INDEX = "clear_sky_index"
    CYCLICAL_DOY = "cyclical_doy"
    ALTITUDE = "altitude"
    LONGITUDE = "longitude"

    def spec_class(self) -> type["DerivedFeature"]:
        """Return the concrete :class:`DerivedFeature` subclass for this kind.

        Implements the protocol :func:`kind_dispatched_from_dict`
        consumes; the same method name appears on :class:`CleanerKind`.
        """
        # Local references avoid the import cycle: each subclass below
        # references FeatureKind for its own ``kind`` property.
        if self is FeatureKind.CLEAR_SKY_INDEX:
            return ClearSkyIndexFeature
        if self is FeatureKind.CYCLICAL_DOY:
            return CyclicalDayOfYearFeature
        if self is FeatureKind.ALTITUDE:
            return AltitudeFeature
        if self is FeatureKind.LONGITUDE:
            return LongitudeFeature
        raise AssertionError(f"Unhandled FeatureKind: {self!r}")  # pragma: no cover


class DerivedFeature(KindTaggedSpec[FeatureKind]):
    """ABC for one computed model input.

    Subclasses are frozen value objects (carry only configuration, no
    fitted state). Provider-style runtime dependencies — e.g. a network
    elevation lookup — are constructor-captured fields, not fitted
    state, so the same subclass can be both serialised (provider
    dropped) and re-hydrated at load time.

    Inherits the kind/required-input/to_dict/_from_dict contract from
    :class:`KindTaggedSpec`; this ABC adds the feature-specific
    ``output_columns`` and ``compute`` contracts.
    """

    @property
    @abstractmethod
    def output_columns(self) -> tuple[str, ...]:
        """Column names this feature produces, in deterministic order."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute this feature from ``df``; result index matches ``df.index``.

        Returns a DataFrame whose columns are exactly
        :attr:`output_columns`. Raises if required state (e.g. an
        injected provider) is missing.
        """


def derived_feature_from_dict(
    d: dict[str, Any],
    *,
    providers: Optional[dict[str, Any]] = None,
) -> DerivedFeature:
    """Inverse of :meth:`DerivedFeature.to_dict` — kind-dispatched.

    Thin wrapper around :func:`kind_dispatched_from_dict` that narrows
    the return type to :class:`DerivedFeature` and supplies the
    family-specific error-message hint.

    Args:
        d: Output of :meth:`DerivedFeature.to_dict`. Must include
            ``"kind"``.
        providers: Map of provider-key → provider, for features that
            need re-injection (e.g. ``{"altitude": PvlibElevationProvider()}``).
            Pass-through for features that don't.

    Raises:
        ValueError: If ``"kind"`` is missing or names an unknown kind.
            Concrete ``_from_dict`` impls also raise if a required
            provider is absent — see e.g. :meth:`AltitudeFeature._from_dict`.
    """
    return kind_dispatched_from_dict(  # type: ignore[no-any-return]
        d,
        kind_enum=FeatureKind,
        providers=providers,
        family_name="derived feature",
    )


# ---------------------------------------------------------------------------
# Concrete features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClearSkyIndexFeature(DerivedFeature):
    """Compute ``kt = ghi / ghi_clear`` for a single satellite-source pair.

    Replaces the previous ``ClearSkyIndexSpec``. Construction-time
    validation is field-local (non-empty names); collision checks
    against other features live on :class:`FeatureSpec`.
    """

    ghi_column: str
    ghi_clear_column: str
    output_column: str

    def __post_init__(self) -> None:
        for name, value in (
            ("ghi_column", self.ghi_column),
            ("ghi_clear_column", self.ghi_clear_column),
            ("output_column", self.output_column),
        ):
            if not value:
                raise ValueError(f"ClearSkyIndexFeature.{name} must be non-empty.")

    @property
    def kind(self) -> FeatureKind:
        return FeatureKind.CLEAR_SKY_INDEX

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (self.output_column,)

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return (self.ghi_column, self.ghi_clear_column)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        kt = clear_sky_index(df[self.ghi_column], df[self.ghi_clear_column])
        return kt.rename(self.output_column).to_frame()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "ghi_column": self.ghi_column,
            "ghi_clear_column": self.ghi_clear_column,
            "output_column": self.output_column,
        }

    @classmethod
    def _from_dict(
        cls,
        d: dict[str, Any],
        *,
        providers: dict[str, Any],
    ) -> "ClearSkyIndexFeature":
        # `providers` is unused for this feature; signature matches the
        # generic dispatcher.
        del providers
        return cls(
            ghi_column=d["ghi_column"],
            ghi_clear_column=d["ghi_clear_column"],
            output_column=d["output_column"],
        )


@dataclass(frozen=True)
class CyclicalDayOfYearFeature(DerivedFeature):
    """Sin/cos encoding of day-of-year, derived from a ``date`` column.

    Multi-column output: produces both ``doy_sin`` and ``doy_cos`` from
    a single :meth:`compute` call. No fields — output column names are
    fixed to keep cross-bundle comparisons honest.
    """

    @property
    def kind(self) -> FeatureKind:
        return FeatureKind.CYCLICAL_DOY

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("doy_sin", "doy_cos")

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return ("date",)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return cyclical_day_of_year(df["date"])

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value}

    @classmethod
    def _from_dict(
        cls,
        d: dict[str, Any],
        *,
        providers: dict[str, Any],
    ) -> "CyclicalDayOfYearFeature":
        del providers, d
        return cls()


# Provider key in the ``providers`` dict for AltitudeFeature.
ALTITUDE_PROVIDER_KEY: str = "altitude"


@dataclass(frozen=True)
class AltitudeFeature(DerivedFeature):
    """Look up altitude in metres for every row's ``(lat, lon)``.

    The :class:`~susse.preprocessing.ElevationProvider` is injected at
    construction. The provider is dropped from :meth:`to_dict` (not JSON
    -serialisable) and must be re-supplied at load time via
    ``providers={"altitude": …}`` to :func:`derived_feature_from_dict`
    or :func:`susse.training.load_bundle`.
    """

    provider: "ElevationProvider"
    output_column: str = "altitude_m"

    def __post_init__(self) -> None:
        if self.provider is None:
            raise ValueError(
                "AltitudeFeature.provider must not be None. Construct "
                "with `AltitudeFeature(provider=PvlibElevationProvider())` "
                "(import from `susse.preprocessing`). For load-time "
                're-injection, pass providers={"altitude": ...} to '
                "load_bundle()."
            )
        if not self.output_column:
            raise ValueError("AltitudeFeature.output_column must be non-empty.")

    @property
    def kind(self) -> FeatureKind:
        return FeatureKind.ALTITUDE

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (self.output_column,)

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return ("lat", "lon")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dedupe on (lat, lon) so the provider is invoked once per
        # unique station / grid point, not once per row.
        unique = df[["lat", "lon"]].drop_duplicates()
        elev_map: dict[tuple[float, float], float] = {
            (lat, lon): self.provider(lat, lon)
            for lat, lon in zip(unique["lat"], unique["lon"])
        }
        values = np.array(
            [elev_map[(lat, lon)] for lat, lon in zip(df["lat"], df["lon"])],
            dtype=float,
        )
        return pd.DataFrame(
            {self.output_column: values},
            index=df.index,
        )

    def to_dict(self) -> dict[str, Any]:
        # Provider deliberately excluded — re-injected at load time.
        return {
            "kind": self.kind.value,
            "output_column": self.output_column,
        }

    @classmethod
    def _from_dict(
        cls,
        d: dict[str, Any],
        *,
        providers: dict[str, Any],
    ) -> "AltitudeFeature":
        if ALTITUDE_PROVIDER_KEY not in providers:
            raise ValueError(
                f"AltitudeFeature requires "
                f"providers[{ALTITUDE_PROVIDER_KEY!r}] at load time. Pass "
                f'`load_bundle(dir, providers={{"{ALTITUDE_PROVIDER_KEY}": '
                f"PvlibElevationProvider()}})` (import from `susse.preprocessing`)."
            )
        return cls(
            provider=providers[ALTITUDE_PROVIDER_KEY],
            output_column=d.get("output_column", "altitude_m"),
        )


@dataclass(frozen=True)
class LongitudeFeature(DerivedFeature):
    """Pass-through of the input frame's ``lon`` column as a model feature.

    KNOWN ANTI-PATTERN — included only for reproducing published baselines
    that used longitude as a predictor. New training pipelines should NOT
    use it.

    Why this is bad: with ~28 ground-truth stations spread across SSA, a
    Random Forest can memorise coordinate-to-target associations, which
    destroys generalisation to held-out stations and to off-station grid
    points. Geographical signal should enter the model via continuous
    physical properties (altitude, distance-to-coast, terrain ruggedness,
    …), not raw coordinates.

    Why it nonetheless exists: Mukiibi & Mikelson (2026) used longitude as
    a predictor (their Table II). Faithful recomputation of those headline
    metrics requires the same predictor set. The recomputation notebook
    under ``notebooks/papers/mukiibi_mikelson_2026/`` is the only
    legitimate consumer; do not import this class from new code.
    """

    output_column: str = "longitude"

    def __post_init__(self) -> None:
        if not self.output_column:
            raise ValueError("LongitudeFeature.output_column must be non-empty.")

    @property
    def kind(self) -> FeatureKind:
        return FeatureKind.LONGITUDE

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (self.output_column,)

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return ("lon",)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {self.output_column: df["lon"].astype(float).to_numpy()},
            index=df.index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "output_column": self.output_column,
        }

    @classmethod
    def _from_dict(
        cls,
        d: dict[str, Any],
        *,
        providers: dict[str, Any],
    ) -> "LongitudeFeature":
        del providers
        return cls(output_column=d.get("output_column", "longitude"))
