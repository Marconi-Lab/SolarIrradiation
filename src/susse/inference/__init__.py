"""Inference — portal-facing wrapper around a TrainedBundle.

Public entry points:

* :class:`Predictor` — coords + date range → long-format DataFrame
  of bias-corrected daily GHI predictions. Reads features from the
  warehouse and applies the bundle's :class:`FeatureSpec`.
* :class:`PredictionRequest` — validated request object exposed for
  callers that want to construct requests programmatically.

Cache misses (geohash5 cells absent from the warehouse for some of
the requested dates) are surfaced loudly by default
(``on_cache_miss="raise"``). With ``on_cache_miss="fetch"``, the
predictor synchronously reuses :class:`NasaPowerSatelliteJob` and
:class:`CamsSatelliteJob` to populate the missing cells, then
re-queries the warehouse before predicting. Per-invocation CAMS
fetches are capped to keep within the daily quota.
"""

from .predictor import PredictionRequest, Predictor

__all__ = ["PredictionRequest", "Predictor"]
