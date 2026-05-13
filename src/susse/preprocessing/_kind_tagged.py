"""Shared base for kind-tagged, JSON-roundtrippable value-object families.

:class:`DerivedFeature` and :class:`DataCleaner` are two such families:
each is a frozen-dataclass-style value object carrying configuration,
tagged with a ``kind`` enum for persistence + dispatch, and
roundtrippable to JSON via :meth:`to_dict` / :meth:`_from_dict`.

The families differ in their verb — :meth:`DerivedFeature.compute`
adds columns, :meth:`DataCleaner.apply` modifies rows — and in their
specific extras (``output_columns`` exists only on the feature side).
Those differences live on the concrete sub-ABCs. Everything else —
the kind contract, the required-input declaration, and the JSON
roundtrip — is shared here.

Each family's enum (:class:`FeatureKind`, :class:`CleanerKind`) carries
a ``spec_class()`` method returning the concrete leaf class for that
kind. :func:`kind_dispatched_from_dict` is the generic
dict-to-instance dispatcher that consumes the enum and returns an
instance of the resolved leaf class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Generic, Optional, TypeVar

KindT = TypeVar("KindT", bound=StrEnum)


class KindTaggedSpec(Generic[KindT], ABC):
    """Frozen-dataclass-style value object with a kind tag + JSON roundtrip.

    Subclasses come in two layers:

    * Concrete sub-ABCs (e.g. :class:`DerivedFeature`,
      :class:`DataCleaner`) that add a family-specific verb and any
      family-specific extras.
    * Leaf classes (e.g. :class:`ClearSkyIndexFeature`,
      :class:`GhiUpperBoundCleaner`) — frozen dataclasses implementing
      the abstract contracts here plus the family's verb.
    """

    @property
    @abstractmethod
    def kind(self) -> KindT:
        """The kind tag identifying this concrete leaf class."""

    @property
    @abstractmethod
    def required_input_columns(self) -> tuple[str, ...]:
        """Source columns this spec needs from the input DataFrame."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict including a ``"kind"`` tag.

        Non-serialisable fields (providers, etc.) are dropped. The
        :meth:`_from_dict` inverse re-injects them from the
        caller-supplied providers dict.
        """

    @classmethod
    @abstractmethod
    def _from_dict(
        cls, d: dict[str, Any], *, providers: dict[str, Any]
    ) -> "KindTaggedSpec[KindT]":
        """Inverse of :meth:`to_dict` for this concrete leaf class.

        Called by :func:`kind_dispatched_from_dict` after kind-dispatch.
        ``providers`` re-injects any non-serialisable runtime
        dependencies (e.g. an :class:`ElevationProvider`) captured at
        construction time.
        """


def kind_dispatched_from_dict(
    d: dict[str, Any],
    *,
    kind_enum: type[StrEnum],
    providers: Optional[dict[str, Any]] = None,
    family_name: str = "spec",
) -> Any:
    """Generic dispatcher: dict → concrete :class:`KindTaggedSpec` leaf.

    Each family's enum must expose a ``spec_class()`` method that
    returns the concrete leaf class for that kind. This contract is
    enforced by convention rather than by a Protocol — :class:`StrEnum`
    + :class:`Protocol` interact awkwardly in mypy, and the contract is
    used in exactly two places.

    Args:
        d: Output of :meth:`KindTaggedSpec.to_dict`. Must include
            ``"kind"``.
        kind_enum: The family's enum (e.g. ``FeatureKind``,
            ``CleanerKind``).
        providers: Map of provider-key → provider for non-serialisable
            re-injection. Pass-through for kinds that don't need it.
        family_name: Used in the error message ("derived feature",
            "cleaner") so callers see a family-specific hint.

    Returns:
        The reconstructed leaf-class instance. Typed as :data:`Any`
        because the helper is family-agnostic; thin family-specific
        wrappers (e.g. :func:`derived_feature_from_dict`) narrow the
        return type via their own annotation.

    Raises:
        ValueError: If ``"kind"`` is missing or names a value not in
            ``kind_enum``.
    """
    if "kind" not in d:
        raise ValueError(
            f"kind_dispatched_from_dict expects a 'kind' tag identifying "
            f"the {family_name} type. Pass a dict produced by to_dict()."
        )
    kind = kind_enum(d["kind"])
    cls = kind.spec_class()  # type: ignore[attr-defined]
    return cls._from_dict(d, providers=providers or {})
