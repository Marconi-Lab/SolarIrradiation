"""Generator for ``tutorial/07_inference.ipynb``.

Edit the cell strings here, then regenerate with::

    python notebooks/tutorial/_build_07_inference.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/tutorial/07_inference.ipynb").resolve()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _split_lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_lines(text),
    }


def _split_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return lines


COLAB_BOOTSTRAP = """\
# ============================================================
# Colab bootstrap (no-op when run locally).
# ------------------------------------------------------------
# First-time setup on Colab:
#   1. Create a GitHub Personal Access Token (PAT) at
#      https://github.com/settings/tokens with `repo` scope.
#      The repository is private, so the clone needs this token
#      (or an SSH key Colab knows about, which is more fiddly).
#   2. Add the token under Tools → Secrets in Colab with name
#      `GITHUB_PAT` and toggle "Notebook access" on.
#   3. Sign in with a Google account that has BigQuery read access
#      to `solar-irradiation-estimation` when prompted.
# ============================================================
import os
import sys

if "google.colab" in sys.modules:
    REPO = "Marconi-Lab/Solar_irradiation"
    BRANCH = "jm/add_model"

    if not os.path.exists("/content/Solar_irradiation/.git"):
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_PAT")
            clone_url = f"https://{token}@github.com/{REPO}.git"
            print("Cloning with Colab secret 'GITHUB_PAT'.")
        except Exception:
            clone_url = f"git@github.com:{REPO}.git"
            print(
                "Colab secret 'GITHUB_PAT' not set — trying SSH. If the "
                "clone fails, follow the PAT setup steps above and re-run."
            )
        !git clone -q -b {BRANCH} {clone_url} /content/Solar_irradiation

    %cd /content/Solar_irradiation
    !pip install -q -e . 2>&1 | tail -3

    from google.colab import auth
    auth.authenticate_user()
    !gcloud config set project solar-irradiation-estimation 2>/dev/null
    print("Colab setup complete.")
"""


CELLS: list[dict] = [
    md(
        """# 07 — Inference: from bundle on disk to deployment-ready predictions

The end of the tutorial chain. Earlier notebooks built the training
dataset, trained models, packaged them into bundles, and evaluated
them. This notebook shows the **deployment-facing API**:

```python
predictor = Predictor.from_bundle_dir(bundle_dir)
df = predictor.predict(
    coords=[(0.333542, 32.56863), (3.04679, 30.91350), ...],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
)
# → long DataFrame: lat, lon, geohash5, date, ..., y_pred_kwh_m2_day
```

This is the call site the companion `Irradiation_Portal` Flask app
mirrors. If you change the bundle, you don't need to update the
portal — re-running this notebook is the smoke test.

## What's new in this notebook

```text
Predictor                                     (susse.inference)
    bundle: TrainedBundle
    service: FeatureService
    on_cache_miss: "raise" | "fetch"
    predict(coords, start_date, end_date) → DataFrame

PredictionRequest                             (dataclass — validated input)
    coords, start_date, end_date
```

## Notebook structure

| Step | What you'll see |
|---|---|
| 0  | Setup + load the paper bundle from `data/bundles/mukiibi_mikelson_2026/` |
| 1  | Single-point prediction: Kampala over 2024, with NASA / CAMS overlaid |
| 2  | Grid prediction: the Uganda 2024 inference cache (~1961 cells × 365 days) |
| 3  | Annual aggregation per cell |
| 4  | Folium choropleth — what the portal renders |
| 5  | Cache-miss diagnostics + the Phase B (on-demand fetch) follow-up |

## Two design choices worth knowing

* **The bundle is the source of truth.** `Predictor` reads
  `bundle.source_manifest.feature_selection` to know which NASA / CAMS
  variable IDs to assemble. Training and inference can never drift
  out of sync.
* **Cache miss is loud by default.** If you ask for a `(geohash5,
  date)` pair that isn't in the warehouse, you get a
  `RuntimeError` naming the missing cells — not a NaN, not a silent
  empty row. Phase B will add the optional on-demand NASA POWER +
  CAMS fetch path; for now, queries must hit cached cells."""
    ),
    md(
        """### Inputs, outputs, and prerequisites

| | |
|---|---|
| **Inputs**  | The paper bundle at `data/bundles/mukiibi_mikelson_2026/`. The Uganda 2024 inference grid in the warehouse (the green-rectangle layer in `notebooks/tutorial/01_data_overview.ipynb`). |
| **Outputs** | None on disk — a long DataFrame in memory and a folium map. |
| **Prereqs** | BigQuery read access (`gcloud auth application-default login` locally; the Colab bootstrap handles it). The bundle must exist; run `notebooks/papers/mukiibi_mikelson_2026/01_recomputation.ipynb` once if not. |
| **Wall time** | ~30 seconds for §1 (one cell × 365 days), ~1 minute for §2 (whole Uganda grid). |"""
    ),
    code(COLAB_BOOTSTRAP),
    md("## 0 — Setup"),
    code(
        """%load_ext autoreload
%autoreload 2

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path.cwd().resolve()
_PROJECT_ROOT = _HERE
while _PROJECT_ROOT != _PROJECT_ROOT.parent and not (_PROJECT_ROOT / "src" / "susse").exists():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Load .env so CAMS_EMAIL (used by the fetch-mode demo in §5) is
# picked up automatically. One helper across all notebooks.
from susse.env import load_project_env
load_project_env()

print(f"project root: {_PROJECT_ROOT}")"""
    ),
    md(
        """### Load the bundle

`Predictor.from_bundle_dir` is the one-liner the portal uses too. It
loads the trained model, the feature spec, the source manifest, and
opens a BigQuery client against the project default warehouse.

The `providers` argument supplies an elevation provider for
`AltitudeFeature` — pvlib's `lookup_altitude()` queries Open-Elevation
on the fly, so no warehouse trip is needed for that derived
feature."""
    ),
    code(
        """from susse.inference import Predictor
from susse.preprocessing import PvlibElevationProvider

BUNDLE_DIR = (_PROJECT_ROOT / "data" / "bundles" / "mukiibi_mikelson_2026").resolve()
assert BUNDLE_DIR.exists(), (
    f"Bundle missing at {BUNDLE_DIR}. Run "
    f"notebooks/papers/mukiibi_mikelson_2026/01_recomputation.ipynb "
    f"end-to-end first."
)

predictor = Predictor.from_bundle_dir(
    BUNDLE_DIR,
    providers={"altitude": PvlibElevationProvider()},
    on_cache_miss="raise",
)
print(f"Bundle:        {BUNDLE_DIR.name}")
print(f"Trained at:    {predictor.bundle.metadata.created_at_utc}")
print(f"Splitter:      {predictor.bundle.metadata.splitter_name}")
print(f"Val RMSE:      {predictor.bundle.metadata.val_metrics.rmse:.3f}  "
      f"(val MAE {predictor.bundle.metadata.val_metrics.mae:.3f})")
print(f"Features:      {len(predictor.bundle.feature_spec.feature_columns)} raw + "
      f"{len(predictor.bundle.feature_spec.derived_features)} derived")"""
    ),
    md(
        """## 1 — Single-point prediction: Kampala over 2024

Kampala is one of the 28 training stations, so its geohash5 is
warehouse-cached for the full 2024 calendar year. The Predictor:

1. Computes geohash5 from `(lat, lon)`
2. Issues one batched query per source (irradiance + NASA aux + CAMS aux)
3. Assembles the inference frame and applies the bundle's `FeatureSpec` (cleaners disabled, no NaN-drop)
4. Runs the regressor in a single batched call

Result: 365 daily bias-corrected GHI predictions plus the raw NASA /
CAMS satellite columns so you can overlay them in plots without
re-querying."""
    ),
    code(
        """KAMPALA_LAT = 0.333542
KAMPALA_LON = 32.56863

kampala = predictor.predict(
    coords=[(KAMPALA_LAT, KAMPALA_LON)],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
)
print(f"Predictions: {len(kampala):,} rows × {len(kampala.columns)} columns")
print(f"Columns of interest:")
for c in ("lat", "lon", "geohash5", "date", "y_pred_kwh_m2_day",
          "sat_ghi_nasa_kwh_m2_day", "sat_ghi_cams_kwh_m2_day"):
    if c in kampala.columns:
        print(f"  {c}")
kampala.head()"""
    ),
    md(
        """### Time-series visualisation

The same convention as NB 06: the model's prediction is rendered
prominently with markers; the raw NASA / CAMS satellite estimates are
faint background lines so the visual contrast surfaces which series
the model *produced* vs the satellite *inputs* it bias-corrected."""
    ),
    code(
        """from susse.evaluation.plots import plot_training_fit_timeseries

# Single-station inference view — no observed series (we don't have
# ground GHI for arbitrary lat/lon at deploy time). The prediction is
# rendered prominently; NASA / CAMS sit faint in the background.
ts = kampala.copy()
ts["date"] = pd.to_datetime(ts["date"])
ts["location"] = "Kampala"

plot_training_fit_timeseries(
    ts,
    observed_column=None,
    prediction_series={"RF (bias-corrected)": "y_pred_kwh_m2_day"},
    reference_series={
        "NASA": "sat_ghi_nasa_kwh_m2_day",
        "CAMS": "sat_ghi_cams_kwh_m2_day",
    },
    location_column="location",
    n_stations=1,
)"""
    ),
    md(
        """## 2 — Grid prediction: the Uganda 2024 inference cache

The warehouse caches NASA + CAMS feature rows for every 0.05° geohash
cell in a Uganda bounding box across 2024. `01_data_overview.ipynb`
renders these as a green rectangle on its coverage map; here we
query the cell centres and predict for the whole grid.

The query below pulls the cell centroids straight from BigQuery so
we never duplicate the grid definition. ~2k cells × 365 days = ~700k
predictions; the bundle's RF is batched, so this takes about a
minute even on a workstation."""
    ),
    code(
        """from susse.warehouse_ops.io import BigQueryClient, WarehouseConfig
from susse.warehouse_ops.io.config import TableRefs

bq = BigQueryClient(config=WarehouseConfig())
tables = TableRefs(config=bq.config)

# Cell centres of the Uganda 2024 grid (the 2024-only NASA cells the
# warehouse caches for portal inference, excluding ground stations).
grid_cells = bq.query(f\"\"\"
SELECT DISTINCT geohash5,
                ANY_VALUE(latitude)  AS lat,
                ANY_VALUE(longitude) AS lon
FROM `{tables.irradiance_daily}`
WHERE date BETWEEN DATE('2024-01-01') AND DATE('2024-12-31')
  AND source = 'NASA'
  AND geohash5 NOT IN (SELECT DISTINCT geohash5 FROM `{tables.ground_measurements}`)
GROUP BY geohash5
ORDER BY lat, lon
\"\"\")
print(f"Uganda 2024 grid: {len(grid_cells):,} cells")
grid_cells.head()"""
    ),
    code(
        """uganda_2024 = predictor.predict(
    coords=list(zip(grid_cells["lat"], grid_cells["lon"])),
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
)
print(f"Daily predictions: {len(uganda_2024):,} rows  "
      f"({uganda_2024[['lat', 'lon']].drop_duplicates().shape[0]:,} unique cells × "
      f"{uganda_2024['date'].nunique()} days)")"""
    ),
    md(
        """## 3 — Annual aggregation per cell

The portal renders an **annual-mean** GHI choropleth. Reduce the
daily long-format frame to one row per cell carrying its 2024 mean
predicted GHI plus the two satellite baselines for comparison."""
    ),
    code(
        """annual = (
    uganda_2024
    .groupby(["lat", "lon", "geohash5"], as_index=False)
    .agg(
        rf_mean=("y_pred_kwh_m2_day", "mean"),
        nasa_mean=("sat_ghi_nasa_kwh_m2_day", "mean"),
        cams_mean=("sat_ghi_cams_kwh_m2_day", "mean"),
    )
)
print(f"Annual mean grid: {len(annual):,} cells")
print(f"  RF   range: {annual['rf_mean'].min():.2f} – {annual['rf_mean'].max():.2f} kWh/m²/day")
print(f"  NASA range: {annual['nasa_mean'].min():.2f} – {annual['nasa_mean'].max():.2f}")
print(f"  CAMS range: {annual['cams_mean'].min():.2f} – {annual['cams_mean'].max():.2f}")
annual.head()"""
    ),
    md(
        """## 4 — Folium choropleth — what the portal renders

A small circle marker per cell, coloured by predicted annual GHI.
This is the static analogue of the portal's interactive heatmap; the
portal uses the same DataFrame with `folium.raster_layers.ImageOverlay`
for a rasterised version."""
    ),
    code(
        """import folium
import branca.colormap as cm

cmap = cm.LinearColormap(
    colors=["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    vmin=float(annual["rf_mean"].min()),
    vmax=float(annual["rf_mean"].max()),
    caption="RF-corrected annual mean GHI (kWh/m²/day)",
)

centre_lat = float(annual["lat"].mean())
centre_lon = float(annual["lon"].mean())
m = folium.Map(location=[centre_lat, centre_lon], zoom_start=7, control_scale=True)
for _, row in annual.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=4,
        color=None, fill=True,
        fillColor=cmap(row["rf_mean"]), fillOpacity=0.85,
        popup=folium.Popup(
            f"<b>geohash5</b>: {row['geohash5']}<br>"
            f"<b>RF</b>:   {row['rf_mean']:.2f} kWh/m²/day<br>"
            f"<b>NASA</b>: {row['nasa_mean']:.2f}<br>"
            f"<b>CAMS</b>: {row['cams_mean']:.2f}",
            max_width=240,
        ),
    ).add_to(m)
cmap.add_to(m)
m"""
    ),
    md(
        """## 5 — Cache misses + on-demand fetch (Phase B)

The Predictor's default behaviour is loud-by-default: any requested
`(geohash5, date)` pair that isn't in the warehouse raises a
`RuntimeError` naming the missing cells. That's what the
`on_cache_miss="raise"` constructor mode you've been using does."""
    ),
    code(
        """# A point in the Indian Ocean — definitely not in any cache.
try:
    predictor.predict(
        coords=[(-5.0, 70.0)],
        start_date=date(2024, 1, 1), end_date=date(2024, 1, 7),
    )
except RuntimeError as e:
    print("Cache miss surfaced as expected:")
    print(str(e)[:400], "...")"""
    ),
    md(
        """### Switching to `on_cache_miss="fetch"`

With `on_cache_miss="fetch"`, the predictor instead invokes the
existing `NasaPowerSatelliteJob` and `CamsSatelliteJob` to fetch +
persist the missing cells before re-querying and predicting. The
same code path the batch-ingest migrations use — same schema, same
idempotency, same hardening — wrapped one layer up.

**Prerequisites for fetch mode:**

* `CAMS_EMAIL` env var set to a registered [SoDa-Pro](https://www.soda-pro.com/web-services/radiation/cams-radiation-service)
  account. The constructor raises `ValueError` upfront if missing.
* BigQuery write access on the warehouse dataset. Failure surfaces at
  the first write.
* Implicit cap of **30 cells per `predict()`** to respect CAMS's free-tier
  daily quota — exceeding it raises rather than silently throttling.

A typical interactive single-coord query for an uncached cell takes
~25-40 seconds: NASA POWER fetch (~5s) + CAMS fetch (~10s) + warehouse
write (~5-10s) + warehouse re-query (~3s) + predict (~1s). Subsequent
queries for the same cell hit the warehouse and run in ~3 seconds."""
    ),
    code(
        """import os

# `load_dotenv()` was already called in §0, so a CAMS_EMAIL set in
# `.env` is already in os.environ here. Either path (shell export or
# .env entry) makes this cell light up.
if not os.environ.get("CAMS_EMAIL"):
    print("CAMS_EMAIL is not set; skipping the fetch-mode demo.\\n"
          "Either `export CAMS_EMAIL=you@example.com` in the shell, or\\n"
          "add `CAMS_EMAIL=you@example.com` to a `.env` at the project\\n"
          "root, then restart the kernel.")
else:
    fetch_predictor = Predictor.from_bundle_dir(
        BUNDLE_DIR,
        providers={"altitude": PvlibElevationProvider()},
        on_cache_miss="fetch",
    )
    # A single uncached coord — Lake Albert, on the Uganda/DRC border.
    # If the Uganda 2024 grid covered it already, this hits the cache;
    # otherwise the fetch path fires.
    lake_albert = fetch_predictor.predict(
        coords=[(1.6, 30.85)],
        start_date=date(2024, 6, 1),
        end_date=date(2024, 6, 7),
    )
    print(f"Predictions: {len(lake_albert)} rows")
    print(lake_albert[["lat", "lon", "date", "y_pred_kwh_m2_day"]].to_string(index=False))"""
    ),
    md(
        """## What's next

* **`Irradiation_Portal`** — the Flask portal already loads a bundle
  via `load_bundle(...)`. The next portal release should switch to
  `Predictor.from_bundle_dir(...)` so the portal and the tutorial
  evolve together. The portal also gets free warehouse caching: any
  user query for an uncached cell, while `on_cache_miss="fetch"`,
  populates the warehouse for the next user.
* **Inspection notebooks** — `notebooks/inspection/` is the right
  place for ad-hoc deployment-readiness checks (per-region bias,
  out-of-distribution probe coordinates, model-vs-model overlays).

The discipline going forward:

| Change | What ripples |
|---|---|
| New model kind | New `BaseRegressor` subclass + params. Predictor signature unchanged. |
| New satellite source | New `Source` enum value + ingest job + entry in `_LONG_AUX_TABLES`. Predictor signature unchanged. |
| New derived feature | New `DerivedFeature` subclass; FeatureSpec gains a tuple entry. Predictor signature unchanged. |
| Cache-miss policy change | `Predictor.on_cache_miss` is a knob; portal flips it. |

The `Predictor` is the unit of deployment. The portal owns a single
instance and reuses it across requests. If something here feels
missing — a request-level metric you want logged, an artifact
provenance field that should travel into every response — extend the
`Predictor` (not the call site)."""
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
