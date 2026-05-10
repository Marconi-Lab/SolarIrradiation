"""Preprocessing — turn raw warehouse rows into model-ready features.

Public surface:

* :class:`FeatureSpec` — typed config bundling pass-through columns +
  a tuple of :class:`DerivedFeature` instances.
* :class:`DerivedFeature` — ABC for one named, computed model input.
  Concrete subclasses today: :class:`ClearSkyIndexFeature`,
  :class:`CyclicalDayOfYearFeature`, :class:`AltitudeFeature`.
* :class:`FeatureKind` — persistence + dispatch tag.
* :func:`derived_feature_from_dict` — JSON → :class:`DerivedFeature`,
  with provider re-injection.
* :class:`Preprocessor` — applies a spec to a
  :class:`~susse.datasets.TrainingDataset`.
* :class:`PreprocessedDataset` — value object bundling the transformed
  DataFrame with its feature column list, target column, and provenance
  back to the source dataset.
* :class:`ElevationProvider`, :class:`PvlibElevationProvider` — runtime
  altitude lookup (used by :class:`AltitudeFeature`).

Stateless: there's no ``fit()``. All transforms in v1 are deterministic
given the inputs (kt division, sin/cos of day-of-year, NaN dropping,
DEM lookup). Scaling lives in the model wrapper since it's
model-specific and only the model knows which fold to fit on.
"""

from .derived import (
    AltitudeFeature,
    ClearSkyIndexFeature,
    CyclicalDayOfYearFeature,
    DerivedFeature,
    FeatureKind,
    derived_feature_from_dict,
)
from .derived_features import clear_sky_index, cyclical_day_of_year
from .elevation import ElevationProvider, PvlibElevationProvider
from .feature_spec import FeatureSpec
from .preprocessor import PreprocessedDataset, Preprocessor

__all__ = [
    "AltitudeFeature",
    "ClearSkyIndexFeature",
    "CyclicalDayOfYearFeature",
    "DerivedFeature",
    "ElevationProvider",
    "FeatureKind",
    "FeatureSpec",
    "PreprocessedDataset",
    "Preprocessor",
    "PvlibElevationProvider",
    "clear_sky_index",
    "cyclical_day_of_year",
    "derived_feature_from_dict",
]
