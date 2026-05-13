"""Inference — portal-facing wrapper around a TrainedBundle.

Public entry points:

* :class:`Predictor` — coords + date range → long-format DataFrame
  of bias-corrected daily GHI predictions. Reads features from the
  warehouse and applies the bundle's :class:`FeatureSpec`.
* :class:`PredictionRequest` — validated request object exposed for
  callers that want to construct requests programmatically.

The current implementation requires every requested ``(geohash5,
date)`` cell to be present in the warehouse. Arbitrary-lat/lon
queries against uncached regions need the on-demand fetch path
(Phase B), which is reserved as a constructor option but not yet
implemented.
"""

from .predictor import PredictionRequest, Predictor

__all__ = ["PredictionRequest", "Predictor"]
