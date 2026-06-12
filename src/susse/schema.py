"""Column-name vocabulary for SuSSE DataFrames.

Single source of truth for every column name the library treats as
semantically meaningful. Consumers — :class:`FeatureService`,
:class:`Preprocessor`, :class:`Trainer`, :class:`Predictor`,
:class:`DerivedFeature` subclasses, :class:`DataCleaner` subclasses
— import from here rather than carrying string literals.

What lives here:

* **Identity columns** (:data:`DATE`, :data:`LOCATION`,
  :data:`GEOHASH5`, :data:`LAT`, :data:`LON`, :data:`QC_LEVEL`) — the
  vocabulary every FeatureService output row carries.
* **Ground-truth and target column names** (:data:`GROUND_GHI`,
  :data:`TRAINING_TARGET`, :data:`PREDICTION`).
* :data:`ID_COLUMNS` — convenience tuple bundling the standard
  identity columns for :attr:`FeatureSpec.id_columns`.

What does *not* live here:

* **Satellite-irradiance column names**
  (``sat_<band>_<source>_kwh_m2_day``) — parametric over
  :class:`Source` × :class:`IrradianceBand`. Built by
  :func:`susse.warehouse_ops.population.types.satellite_irradiance_column`
  (kept next to the enums it depends on to avoid a cycle).
* **Auxiliary aux-column names** (``<source_prefix>_<variable_id>``,
  e.g. ``nasa_aod_550``) — parametric over the prefix in
  ``_LONG_AUX_TABLES`` and the catalog ``variable_id``. Constructed at
  pivot time inside the satellite repository.
* **Derived-feature output column names** — those are per-feature
  user-chosen configuration (e.g. ``kt_nasa`` from a
  :class:`ClearSkyIndexFeature` instance), not a library-wide vocabulary.

Relationship to the warehouse-side vocabulary
---------------------------------------------

There are three places where column-name vocabulary lives in this
repository. They are parallel and *deliberately separate*:

1. **This module** (:mod:`susse.schema`) — names for Python DataFrame
   columns the library produces and consumes.
2. **BigQuery DDL** in ``warehouse/sql/00_schema/*.sql`` — the actual
   column names in the warehouse tables.
3. **:class:`TableSchemas`** in
   :mod:`susse.warehouse_ops.io.config` — Python-side
   ``table_id`` + MERGE-key registry; merge keys reference BigQuery
   column names.

Several strings overlap across these layers (notably ``"date"`` and
``"geohash5"``). That is **not** duplication to deduplicate — it is the
"translate at boundaries" pattern from ``CLAUDE.md``:

* Each layer is correct in its own domain (Python DataFrame contract,
  BigQuery storage, or merge-key contract).
* The :class:`SatelliteRepository` and :class:`GroundRepository` SQL
  queries are the *translators* — they SELECT warehouse columns into
  the DataFrame names this module declares, aliasing where the two
  diverge (e.g. ``latitude`` → ``lat`` for tables that store the long
  form).
* Today the SQL projects align with this module's names by
  *convention*, so most queries don't need aliases. If a future
  warehouse rename diverges from a name here, the fix is to alias in
  the repository query — **not** to couple :class:`TableSchemas` merge
  keys to these constants. The two contracts are independent on
  purpose.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity columns — present on every FeatureService output row.
# ---------------------------------------------------------------------------

DATE: str = "date"
"""ISO date of the observation. Index column for most downstream joins."""

LOCATION: str = "location"
"""Free-text station name (e.g. ``"egypt_location1"``). Present on
training-pair rows; absent on inference-grid rows."""

GEOHASH5: str = "geohash5"
"""5-character geohash of the row's ``(lat, lon)``. Warehouse joins
key on this. Precision is controlled by
:attr:`WarehouseOptions.geohash_precision` (default 5)."""

LAT: str = "lat"
"""Latitude in decimal degrees."""

LON: str = "lon"
"""Longitude in decimal degrees."""

QC_LEVEL: str = "qc_level"
"""Quality-control tag on a ground measurement (e.g. ``"pass"``,
``"fail_range"``). Kept through training-pair assembly for
traceability; not a model input."""

# ---------------------------------------------------------------------------
# Bundle for FeatureSpec.id_columns and callers that want the canonical set.
# ---------------------------------------------------------------------------

ID_COLUMNS: tuple[str, ...] = (DATE, LOCATION, GEOHASH5)
"""The identity-column bundle :class:`FeatureSpec` carries through
preprocessing by default. Read-only — callers that want a custom
subset construct their own tuple."""

# ---------------------------------------------------------------------------
# Ground-truth + target + prediction column names.
# ---------------------------------------------------------------------------

GROUND_GHI: str = "ghi_kwh_m2_day"
"""Ground-truth GHI column name as stored in the warehouse
``ground_measurements`` table. :meth:`FeatureService.build_training_pairs`
renames this to :data:`TRAINING_TARGET` on the way out."""

TRAINING_TARGET: str = "y_ghi_kwh_m2_day"
"""Post-rename target column on the training-pairs frame, the value
:class:`Preprocessor` treats as ``spec.target_column`` by default."""

PREDICTION: str = "y_pred_kwh_m2_day"
"""Model output column added to the long-format result by
:meth:`Predictor.predict`. Mirrors :data:`TRAINING_TARGET`'s naming
convention so plots and metrics can swap series cleanly."""
