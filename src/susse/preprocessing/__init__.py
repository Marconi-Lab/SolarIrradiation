"""Preprocessing — turn raw warehouse rows into model-ready features.

Public surface:

* :class:`FeatureSpec` — typed config bundling pass-through columns,
  row-level cleaners, and derived features.
* :class:`DataCleaner` — ABC for one row-level transform (filter or
  impute). Concrete subclasses: :class:`GhiUpperBoundCleaner`,
  :class:`IqrLowerBoundCleaner`, :class:`HighMissingYearExcluder`,
  :class:`KnnYearGapImputer`. Run *before* derived features.
* :class:`DerivedFeature` — ABC for one named, computed model input.
  Concrete subclasses: :class:`ClearSkyIndexFeature`,
  :class:`CyclicalDayOfYearFeature`, :class:`AltitudeFeature`, and
  :class:`LongitudeFeature` (paper-faithful only — see its docstring).
* :class:`DerivedColumnMetadata` — human-facing label / unit /
  description for one column a :class:`DerivedFeature` emits, exposed
  via :attr:`DerivedFeature.output_metadata`.
* :class:`CleanerKind`, :class:`FeatureKind` — persistence + dispatch
  tags for the two ABC families.
* :func:`data_cleaner_from_dict`, :func:`derived_feature_from_dict` —
  JSON → instance, with provider re-injection.
* :class:`Preprocessor` — applies a spec to a
  :class:`~susse.datasets.TrainingDataset`. Order: cleaners →
  pass-through + derived features → dropna policy.
* :class:`PreprocessedDataset` — value object bundling the transformed
  DataFrame with its feature column list, target column, and provenance
  back to the source dataset.
* :class:`ElevationProvider`, :class:`PvlibElevationProvider` — runtime
  altitude lookup (used by :class:`AltitudeFeature`).

Stateless: there's no ``fit()``. All transforms are deterministic given
the inputs. Scaling lives in the model wrapper since it's
model-specific and only the model knows which fold to fit on.
"""

from .cleaning import (
    CleanerKind,
    DataCleaner,
    GhiUpperBoundCleaner,
    HighMissingYearExcluder,
    IqrLowerBoundCleaner,
    KnnYearGapImputer,
    PerStationMeanImputer,
    data_cleaner_from_dict,
)
from .derived import (
    AltitudeFeature,
    ClearSkyIndexFeature,
    CyclicalDayOfYearFeature,
    DerivedColumnMetadata,
    DerivedFeature,
    FeatureKind,
    LongitudeFeature,
    derived_feature_from_dict,
)
from .derived_features import clear_sky_index, cyclical_day_of_year
from .elevation import ElevationProvider, PvlibElevationProvider
from .feature_spec import FeatureSpec
from .preprocessor import PreprocessedDataset, Preprocessor

__all__ = [
    "AltitudeFeature",
    "CleanerKind",
    "ClearSkyIndexFeature",
    "CyclicalDayOfYearFeature",
    "DataCleaner",
    "DerivedColumnMetadata",
    "DerivedFeature",
    "ElevationProvider",
    "FeatureKind",
    "FeatureSpec",
    "GhiUpperBoundCleaner",
    "HighMissingYearExcluder",
    "IqrLowerBoundCleaner",
    "KnnYearGapImputer",
    "LongitudeFeature",
    "PerStationMeanImputer",
    "PreprocessedDataset",
    "Preprocessor",
    "PvlibElevationProvider",
    "clear_sky_index",
    "cyclical_day_of_year",
    "data_cleaner_from_dict",
    "derived_feature_from_dict",
]
