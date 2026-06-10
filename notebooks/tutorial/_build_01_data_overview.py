"""Generator for ``tutorial/01_data_overview.ipynb``.

Edit the cell strings here, then regenerate with::

    python notebooks/tutorial/_build_01_data_overview.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/tutorial/01_data_overview.ipynb").resolve()


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
        """# 01 — What's in the warehouse

A first look at the SuSSE warehouse — what tables exist, what each one
holds, and where on the map the data lives. Read this before any other
tutorial notebook. Everything downstream queries the warehouse this
notebook describes.

The notebook is **read-only**: every cell issues SELECTs only. Safe to
re-run as often as you like.

## Three things by the end

1. A mental model of the warehouse — six tables, three storage shapes,
   one catalog.
2. A map showing exactly where SuSSE has ground measurements,
   validation sites, and the inference grid.
3. Per-source coverage tables, so the next notebook
   (`02_query_the_warehouse.ipynb`) doesn't surprise you with empty
   results."""
    ),
    code(COLAB_BOOTSTRAP),
    md("## 0 — Setup"),
    code(
        """%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

import folium
import pandas as pd
import pygeohash
from folium import CircleMarker, FeatureGroup, LayerControl, Rectangle

# Locate project root by walking up until we see src/susse/.
_HERE = Path.cwd().resolve()
_PROJECT_ROOT = _HERE
while _PROJECT_ROOT != _PROJECT_ROOT.parent and not (_PROJECT_ROOT / "src" / "susse").exists():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from susse.warehouse_ops.io import BigQueryClient, WarehouseConfig
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.population import (
    PhysicalStorage, Source, VariableCatalog, variables_to_dataframe,
)

bq = BigQueryClient(config=WarehouseConfig())
tables = TableRefs(config=bq.config)
print(f"warehouse: {bq.config.project_id}.{bq.config.dataset}")"""
    ),
    md(
        """## 1 — The tables, at a glance

| Table | Shape | What it stores |
|---|---|---|
| `dim_variable` | catalog | One row per `(variable_id, source)` — unit, valid range, spatial resolution, which physical table holds the value |
| `irradiance_daily` | wide | Per-`(date, geohash5, source)` daily GHI / DHI / DNI for NASA POWER and CAMS |
| `nasa_daily_vars_long` | long | NASA POWER auxiliary variables (`temperature`, `aod_550`, `cloud_amount`, …) keyed by `(date, geohash5, variable_id)` |
| `cams_daily_vars_long` | long | CAMS auxiliary variables (`ghi_clear`, `bhi`, `dni_clear`, …) |
| `merra_daily_vars_long` | long | MERRA-2 reanalysis variables (aerosol decomposition, precipitable water) |
| `modis_observations` | observation-keyed | MODIS composites keyed by `(date, geohash5, product_id, band_id)` |
| `ground_measurements` | curated long | Daily ground GHI per station with QC level |
| `ground_measurements_raw` | uncurated long | Raw partner CSVs before curation |

**Wide vs long** is chosen per source: GHI / DHI / DNI go in the wide
`irradiance_daily` because they're a tightly coupled trio always
queried together; everything else lives in a long-format companion so
adding a new variable doesn't require a schema migration.

`dim_variable.physical_storage` tags every catalog row with which of
the three storage shapes it lives in (`long_format` / `irradiance_wide`
/ `modis_observations`). `FeatureSelection` (NB 02) uses this tag to
reject typos at construction with a clear error.

For the full data lineage (which job writes which table, what's derived
from what), see [`warehouse/README.md`](../../warehouse/README.md)."""
    ),
    md("### Browsing the variable catalog"),
    code(
        """# The catalog is hand-maintained in code, then upserted into dim_variable.
# Browse it without hitting BigQuery:
catalog_df = variables_to_dataframe(VariableCatalog.all_variables())
print(f"Catalog: {len(catalog_df)} variables across {catalog_df['source'].nunique()} sources.")
print()
print("Counts by source × storage:")
print(
    catalog_df.groupby(["source", "physical_storage"]).size().unstack(fill_value=0)
)"""
    ),
    code(
        """# Drill into one source — list variable_ids and their storage.
# Repeat with Source.CAMS, Source.MERRA_2, Source.MODIS as needed.
nasa = catalog_df[catalog_df["source"] == Source.NASA_POWER.value]
print(f"NASA POWER ({len(nasa)} variables):")
nasa[["variable_id", "display_name", "unit", "physical_storage"]]"""
    ),
    md(
        """## 2 — Coverage on the map

Three layers — toggle from the layer control in the top right.

| Marker | What it is | Source |
|---|---|---|
| **Blue dots** (sized by coverage) | Ground-measurement stations (training labels) | `ground_measurements` |
| **Red / orange / green dots** | Katongole 2023 validation stations, colour by warehouse coverage status | `data/external_references/katongole_2023_monthly.csv` + `irradiance_daily` |
| **Green rectangle** | Uganda 2024 inference grid (portal-coverage cache) | `irradiance_daily` filtered to 2024-only cells |

Click any marker for details."""
    ),
    code(
        """# Ground stations + their per-source satellite coverage.
ground = bq.query(f\"\"\"
WITH ground_summary AS (
  SELECT location,
         ANY_VALUE(lat) AS lat,
         ANY_VALUE(lon) AS lon,
         ANY_VALUE(geohash5) AS geohash5,
         MIN(date) AS min_date,
         MAX(date) AS max_date,
         COUNT(*) AS n_rows,
         COUNTIF(qc_level = 'pass') AS n_qc_passed
  FROM `{tables.ground_measurements}`
  GROUP BY location
),
sat_coverage AS (
  SELECT geohash5,
         COUNT(DISTINCT IF(source='NASA', date, NULL)) AS n_nasa_days,
         COUNT(DISTINCT IF(source='CAMS', date, NULL)) AS n_cams_days
  FROM `{tables.irradiance_daily}`
  GROUP BY geohash5
)
SELECT g.*, COALESCE(s.n_nasa_days, 0) AS n_nasa_days,
            COALESCE(s.n_cams_days, 0) AS n_cams_days
FROM ground_summary g
LEFT JOIN sat_coverage s USING (geohash5)
ORDER BY n_rows DESC
\"\"\")
print(f"Ground stations: {len(ground)}")
ground.head()"""
    ),
    code(
        """# Katongole validation stations + 2017-2022 coverage status.
KATONGOLE_CSV = (
    _PROJECT_ROOT / "data" / "external_references" / "katongole_2023_monthly.csv"
)
katongole = pd.read_csv(KATONGOLE_CSV)
katongole["geohash5"] = [
    pygeohash.encode(lat, lon, precision=5)
    for lat, lon in zip(katongole["latitude"], katongole["longitude"])
]
_gh_quoted = ", ".join(f"'{g}'" for g in katongole["geohash5"].unique())
cov_2017_22 = bq.query(f\"\"\"
SELECT geohash5,
       COUNT(DISTINCT IF(source='NASA', date, NULL)) AS n_nasa_days,
       COUNT(DISTINCT IF(source='CAMS', date, NULL)) AS n_cams_days
FROM `{tables.irradiance_daily}`
WHERE date BETWEEN DATE('2017-01-01') AND DATE('2022-12-31')
  AND geohash5 IN ({_gh_quoted})
GROUP BY geohash5
\"\"\")
katongole = katongole.merge(cov_2017_22, on="geohash5", how="left").fillna(
    {"n_nasa_days": 0, "n_cams_days": 0}
).astype({"n_nasa_days": int, "n_cams_days": int})
_N_EXPECTED = (pd.Timestamp("2022-12-31") - pd.Timestamp("2017-01-01")).days + 1
_FULL = int(0.95 * _N_EXPECTED)
katongole["coverage_status"] = [
    "full" if (n_nasa >= _FULL and n_cams >= _FULL)
    else "missing" if (n_nasa == 0 and n_cams == 0)
    else "partial"
    for n_nasa, n_cams in zip(katongole["n_nasa_days"], katongole["n_cams_days"])
]
print(
    f"Katongole stations: {len(katongole)} — "
    + ", ".join(
        f"{k}: {v}" for k, v in katongole['coverage_status'].value_counts().items()
    )
)"""
    ),
    code(
        """# Uganda 2024 inference-grid bbox.
uganda_grid = bq.query(f\"\"\"
SELECT MIN(latitude)  AS min_lat,
       MAX(latitude)  AS max_lat,
       MIN(longitude) AS min_lon,
       MAX(longitude) AS max_lon,
       COUNT(DISTINCT geohash5) AS n_cells,
       MIN(date) AS min_date,
       MAX(date) AS max_date
FROM `{tables.irradiance_daily}`
WHERE date BETWEEN DATE('2024-01-01') AND DATE('2024-12-31')
  AND source = 'NASA'
  AND geohash5 NOT IN (SELECT DISTINCT geohash5 FROM `{tables.ground_measurements}`)
\"\"\")
uganda_grid"""
    ),
    code(
        """def _ground_popup(row: pd.Series) -> str:
    return (
        f\"<b>{row.location}</b><br>\"
        f\"lat={row.lat:.3f}, lon={row.lon:.3f} (geohash5 {row.geohash5})<br>\"
        f\"<b>Ground</b>: {row.n_qc_passed:,} QC-passed days \"
        f\"({row.min_date} → {row.max_date})<br>\"
        f\"<b>NASA</b>:   {row.n_nasa_days:,} days<br>\"
        f\"<b>CAMS</b>:   {row.n_cams_days:,} days\"
    )


def _katongole_popup(row: pd.Series) -> str:
    if row.coverage_status == "full":
        cov = "<span style='color:green'>full 2017-2022 coverage</span>"
    elif row.coverage_status == "missing":
        cov = (
            \"<span style='color:red'>no warehouse data yet</span> — \"
            \"run migration A12\"
        )
    else:
        cov = (
            f\"<span style='color:orange'>partial</span>: \"
            f\"NASA {row.n_nasa_days} days, CAMS {row.n_cams_days} days\"
        )
    return (
        f\"<b>{row.location}</b><br>\"
        f\"lat={row.latitude:.3f}, lon={row.longitude:.3f}<br>\"
        f\"{cov}\"
    )


m = folium.Map(location=[5.0, 25.0], zoom_start=4, control_scale=True)

ground_layer = FeatureGroup(name=f"Ground stations ({len(ground)})", show=True)
for _, row in ground.iterrows():
    radius = 4 + int(0.001 * row.n_qc_passed)
    CircleMarker(
        location=[row.lat, row.lon],
        radius=min(radius, 14),
        popup=folium.Popup(_ground_popup(row), max_width=320),
        tooltip=row.location,
        color="#1f77b4", weight=1, fill=True,
        fillColor="#1f77b4", fillOpacity=0.8,
    ).add_to(ground_layer)
ground_layer.add_to(m)

katongole_layer = FeatureGroup(
    name=f"Katongole validation ({len(katongole)})", show=True,
)
_status_color = {"full": "#2ca02c", "partial": "#ff7f0e", "missing": "#d62728"}
for _, row in katongole.iterrows():
    color = _status_color[row.coverage_status]
    CircleMarker(
        location=[row.latitude, row.longitude], radius=4,
        popup=folium.Popup(_katongole_popup(row), max_width=320),
        tooltip=f\"{row.location} — {row.coverage_status}\",
        color=color, weight=1, fill=True,
        fillColor=color, fillOpacity=0.85,
    ).add_to(katongole_layer)
katongole_layer.add_to(m)

ug_layer = FeatureGroup(name="Uganda 2024 inference grid", show=True)
ug_row = uganda_grid.iloc[0]
Rectangle(
    bounds=[
        [float(ug_row.min_lat), float(ug_row.min_lon)],
        [float(ug_row.max_lat), float(ug_row.max_lon)],
    ],
    color="#2ca02c", weight=2, fill=True,
    fillColor="#2ca02c", fillOpacity=0.08,
    popup=folium.Popup(
        f\"<b>Uganda 2024 inference grid</b><br>\"
        f\"{int(ug_row.n_cells):,} cells × 365 days<br>\"
        f\"{ug_row.min_date} → {ug_row.max_date}\",
        max_width=320,
    ),
    tooltip=\"Uganda 2024 inference grid\",
).add_to(ug_layer)
ug_layer.add_to(m)

LayerControl(collapsed=False).add_to(m)
m"""
    ),
    md(
        """## 3 — Where to go next

* [`02_query_the_warehouse.ipynb`](02_query_the_warehouse.ipynb) — build
  a `TrainingDataset` from the warehouse using `FeatureSelection` /
  `FeatureService`, plus an example of dropping down to raw SQL when
  needed.
* [`03_preprocessing.ipynb`](03_preprocessing.ipynb) → 04 → 05 — the
  rest of the pipeline: preprocessing, model factory, trainer.
* [`../../warehouse/extending_the_warehouse.ipynb`](../../warehouse/extending_the_warehouse.ipynb)
  — for contributors who need to add new variables, sources, or
  stations to the warehouse.
* [`../papers/mukiibi_mikelson_2026/01_recomputation.ipynb`](../papers/mukiibi_mikelson_2026/01_recomputation.ipynb)
  — a complete worked example, paper-faithful Random Forest GHI bias
  correction trained against the data shown above and validated against
  the Katongole network."""
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
