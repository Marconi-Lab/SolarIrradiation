"""Preprocessing — turn raw warehouse rows into model-ready features.

Public surface:

* :class:`FeatureSpec`, :class:`ClearSkyIndexSpec` — typed configs.
* :class:`Preprocessor` — applies a spec to a
  :class:`~susse.datasets.TrainingDataset`.
* :class:`PreprocessedDataset` — value object bundling the transformed
  DataFrame with its feature column list, target column, and provenance
  back to the source dataset.

Stateless: there's no ``fit()``. All transforms in v1 are deterministic
given the inputs (kt division, sin/cos of day-of-year, NaN dropping).
Scaling lives in the model wrapper since it's model-specific and only
the model knows which fold to fit on.
"""

from .derived_features import clear_sky_index, cyclical_day_of_year
from .feature_spec import ClearSkyIndexSpec, FeatureSpec
from .preprocessor import PreprocessedDataset, Preprocessor

__all__ = [
    "ClearSkyIndexSpec",
    "FeatureSpec",
    "PreprocessedDataset",
    "Preprocessor",
    "clear_sky_index",
    "cyclical_day_of_year",
]
