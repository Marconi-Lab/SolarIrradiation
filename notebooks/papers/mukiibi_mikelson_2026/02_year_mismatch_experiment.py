"""02 — Year-mismatch experiment.

Tests whether the apparent Katongole gap (RF visibly under-corrects vs
the climatology) is driven by the year mismatch in §6 — daily 2024
predictions averaged to monthly, compared against Katongole's
2017–2022 climatology.

Procedure
---------
1. Pick a small set of Katongole stations (the worst-bias central-Uganda
   ones, by default) where the gap is most visible.
2. Ingest **fresh** NASA POWER + CAMS data at those stations' exact
   (lat, lon) for the 2017-2022 climatology window. Idempotent — skips
   what's already cached.
3. Load the existing trained bundle from
   ``data/bundles/mukiibi_mikelson_2026``.
4. Build inference frames at those stations for both windows:
   the original 2024 single-year, and the 2017–2022 6-year window.
5. Predict daily, aggregate to monthly **climatology** (mean across
   years per calendar month), and compare against the Katongole CSV
   alongside the raw NASA / CAMS climatologies.
6. Print and plot per-station RMSE for both windows.

Interpretation
--------------
* If 2017-2022 RMSE is **substantially better** than 2024 → the gap is
  largely a year-mismatch artefact; we should extend this to all 54
  stations.
* If RMSE is **about the same** → the model has a real generalisation
  issue (training set composition, feature gaps, or both); the year
  mismatch isn't the load-bearing problem.

Run as: ``python notebooks/papers/mukiibi_mikelson_2026/02_year_mismatch_experiment.py``
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pygeohash
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
# Load .env so CAMS_EMAIL etc. are visible to os.getenv.
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("year_mismatch_experiment")

# Resolve the project root from this script's path so the imports work
# regardless of where Python was launched.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from susse.datasets import (
    DatasetManifest, FeatureSelection, FeatureService, TrainingDataset,
)
from susse.preprocessing import (
    FeatureSpec, Preprocessor, PvlibElevationProvider,
)
from susse.training import load_bundle
from susse.warehouse_ops.io import BigQueryClient, WarehouseConfig
from susse.warehouse_ops.io.config import TableRefs
from susse.warehouse_ops.io.repositories import SatelliteRepository
from susse.warehouse_ops.population.dim_variable import VariableCatalog
from susse.warehouse_ops.population.jobs import (
    CamsSatelliteJob, NasaPowerSatelliteJob,
)
from susse.warehouse_ops.population.types import (
    DateRange, LocationSpec, NamedLocationsPlan, Source,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Stations chosen from the worst-fit panel of the v1 recomputation. Lake
# Victoria region where NASA + CAMS systematically over-estimate by
# ~1.5 kWh/m²/day. Edit this list to extend the experiment.
EXPERIMENT_STATIONS: tuple[str, ...] = (
    "Nakivubo", "Bulindi", "Makerere S", "Buvuma HQ", "Kasese S",
)

# Target climatology window — same as the Katongole CSV's 2017-2022 mean.
EXPERIMENT_DATE_START = date(2017, 1, 1)
EXPERIMENT_DATE_END = date(2022, 12, 31)

# Comparison window — what the §6 recomputation currently uses.
BASELINE_YEAR = 2024

# Variables — must match the recomputation notebook's FeatureSelection
# so the loaded bundle's spec aligns with the inference frame columns.
NASA_AUX_VARIABLE_IDS: tuple[str, ...] = (
    "longwave_downward_irr", "aod_550_adj", "cloud_amount", "precipitable_water",
    "airmass", "zero_plane_displacement", "surface_albedo", "clearness_index",
    "temperature", "relative_humidity", "surface_pressure", "wind_speed",
    "temperature_range", "precipitation_corrected", "surface_roughness",
    "northern_wind", "evapotranspiration_energy", "planetary_boundary",
    "total_column_ozone", "surface_air_density", "evaporation_land",
    "surface_soil_wetness",
)
CAMS_VARIABLE_IDS: tuple[str, ...] = ("ghi_clear", "dhi_clear", "dni_clear")
NASA_IRRADIANCE_IDS = ("ghi", "dhi", "dni")
CAMS_IRRADIANCE_IDS = ("ghi", "dhi", "dni")

BUNDLE_DIR = (_PROJECT_ROOT / "data" / "bundles" / "mukiibi_mikelson_2026").resolve()
KATONGOLE_CSV = (
    _PROJECT_ROOT / "notebooks" / "papers" / "mukiibi_mikelson_2026"
    / "reference_data" / "katongole_2023_monthly.csv"
)
OUTPUT_DIR = KATONGOLE_CSV.parent.parent  # alongside the recomputation notebook
PLOT_PATH = OUTPUT_DIR / "02_year_mismatch_comparison.png"
CSV_PATH = OUTPUT_DIR / "02_year_mismatch_comparison.csv"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_catalog_variables(source: Source, ids: set[str]):
    """Return the VariableSpec entries from the catalog matching ``ids``."""
    return tuple(
        v for v in VariableCatalog.for_source(source) if v.variable_id in ids
    )


def _named_locations_plan(
    source: Source, locations: tuple[LocationSpec, ...],
    variable_ids: set[str],
) -> NamedLocationsPlan:
    return NamedLocationsPlan(
        source=source,
        date_range=DateRange(
            start=EXPERIMENT_DATE_START, end=EXPERIMENT_DATE_END,
        ),
        locations=locations,
        variables=_select_catalog_variables(source, variable_ids),
    )


def _check_cams_email_present() -> bool:
    if os.getenv("CAMS_EMAIL"):
        return True
    log.warning(
        "CAMS_EMAIL is not set — CAMS ingest will be skipped. "
        "Set it in your .env or shell environment to enable."
    )
    return False


# ---------------------------------------------------------------------------
# Step 1 — load station coords from the Katongole CSV
# ---------------------------------------------------------------------------


def load_experiment_stations() -> tuple[LocationSpec, ...]:
    katongole = pd.read_csv(KATONGOLE_CSV)
    sub = katongole[katongole["location"].isin(EXPERIMENT_STATIONS)]
    missing = set(EXPERIMENT_STATIONS) - set(sub["location"])
    if missing:
        raise SystemExit(
            f"Stations missing from {KATONGOLE_CSV.name}: {sorted(missing)}"
        )
    locations = tuple(
        LocationSpec(
            name=row.location, lat=float(row.latitude), lon=float(row.longitude),
        )
        for row in sub.itertuples(index=False)
    )
    log.info(
        "Experiment stations: %s",
        ", ".join(f"{loc.name}({loc.lat:.3f}, {loc.lon:.3f})" for loc in locations),
    )
    return locations


# ---------------------------------------------------------------------------
# Step 2 — fresh ingest of NASA + CAMS at the experiment stations
# ---------------------------------------------------------------------------


def run_ingest(locations: tuple[LocationSpec, ...]) -> None:
    bq = BigQueryClient(config=WarehouseConfig())
    refs = TableRefs(config=bq.config)

    nasa_ids = set(NASA_AUX_VARIABLE_IDS) | set(NASA_IRRADIANCE_IDS)
    cams_ids = set(CAMS_VARIABLE_IDS) | set(CAMS_IRRADIANCE_IDS)

    log.info(
        "NASA POWER ingest: %d stations × %d days, %d variables",
        len(locations),
        (EXPERIMENT_DATE_END - EXPERIMENT_DATE_START).days + 1,
        len(nasa_ids),
    )
    nasa_job = NasaPowerSatelliteJob(bq=bq, refs=refs)
    nasa_result = nasa_job.run(
        _named_locations_plan(Source.NASA_POWER, locations, nasa_ids)
    )
    log.info("NASA POWER result: %s", nasa_result.summary())

    if _check_cams_email_present():
        log.info(
            "CAMS ingest: %d stations × %d days, %d variables",
            len(locations),
            (EXPERIMENT_DATE_END - EXPERIMENT_DATE_START).days + 1,
            len(cams_ids),
        )
        cams_job = CamsSatelliteJob(bq=bq, refs=refs)
        cams_result = cams_job.run(
            _named_locations_plan(Source.CAMS, locations, cams_ids)
        )
        log.info("CAMS result: %s", cams_result.summary())


# ---------------------------------------------------------------------------
# Step 3-4 — build inference frame for an arbitrary date range
# ---------------------------------------------------------------------------


def build_inference_frame(
    locations: tuple[LocationSpec, ...],
    date_start: date, date_end: date,
) -> pd.DataFrame:
    """Pull warehouse features for a list of stations × a date range and
    shape them into one row per (station, date)."""
    bq = BigQueryClient(config=WarehouseConfig())
    refs = TableRefs(config=bq.config)
    sat = SatelliteRepository(bq=bq, tables=refs)

    geohashes = tuple(
        pygeohash.encode(loc.lat, loc.lon, precision=5) for loc in locations
    )
    irr = sat.daily_irradiance_by_geohash(
        date_start, date_end,
        sources=("NASA", "CAMS"),
        geohash5s=geohashes,
    )
    nasa_aux = sat.long_aux_pivoted(
        table_fqn=refs.nasa_daily_vars_long, column_prefix="nasa",
        start=date_start, end=date_end,
        variable_ids=NASA_AUX_VARIABLE_IDS,
        geohash5s=geohashes,
    )
    cams_aux = sat.long_aux_pivoted(
        table_fqn=refs.cams_daily_vars_long, column_prefix="cams",
        start=date_start, end=date_end,
        variable_ids=CAMS_VARIABLE_IDS,
        geohash5s=geohashes,
    )
    feats = (
        irr
        .merge(nasa_aux, on=["date", "geohash5"], how="left")
        .merge(cams_aux, on=["date", "geohash5"], how="left")
    )
    station_lookup = pd.DataFrame([
        {"location": loc.name, "lat": loc.lat, "lon": loc.lon,
         "geohash5": pygeohash.encode(loc.lat, loc.lon, 5)}
        for loc in locations
    ])
    out = feats.merge(station_lookup, on="geohash5", how="inner")
    log.info(
        "Inference frame for %s..%s: %d rows × %d cols across %d stations",
        date_start, date_end, len(out), len(out.columns),
        out["location"].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# Step 5 — predict + climatology
# ---------------------------------------------------------------------------


def predict_and_climatology(
    inference_df: pd.DataFrame, bundle, label: str,
) -> pd.DataFrame:
    """Apply the bundle's preprocessor + regressor; aggregate to monthly
    climatology (mean over years per calendar month)."""
    # Inference: drop the cleaners (target-curation only); keep the
    # feature column list and derived features so X aligns with X_train.
    spec = bundle.feature_spec
    inference_spec = FeatureSpec(
        target_column=spec.target_column,
        feature_columns=spec.feature_columns,
        cleaners=(),  # inference has no target to clean
        derived_features=spec.derived_features,
        id_columns=spec.id_columns,
        dropna_target=False,
        dropna_features=False,
    )
    # Wrap for Preprocessor.apply.
    df = inference_df.assign(y_ghi_kwh_m2_day=np.nan).copy()
    manifest = DatasetManifest(
        name=f"experiment_{label}", version="v0",
        created_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        susse_version="experiment", git_sha=None,
        feature_selection=FeatureSelection(),
        date_start=df["date"].min(), date_end=df["date"].max(),
        location_filter=None,
        warehouse_project="exp", warehouse_dataset="exp",
        warehouse_table_mods={},
        n_rows=len(df), n_cols=len(df.columns),
        column_names=tuple(df.columns), content_hash="exp",
    )
    processed = Preprocessor(inference_spec).apply(
        TrainingDataset(df=df, manifest=manifest)
    )
    preds = bundle.regressor.predict(processed.X())
    out = processed.df.assign(y_pred=preds.values)

    # Aggregate to per-station per-month climatology (mean across years).
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year
    daily = out.set_index(["location", "year", "month"])[
        ["y_pred", "sat_ghi_nasa_kwh_m2_day", "sat_ghi_cams_kwh_m2_day"]
    ]
    monthly = daily.groupby(level=[0, 1, 2]).mean()
    climatology = monthly.groupby(level=[0, 2]).mean().reset_index()
    climatology["window"] = label
    return climatology


# ---------------------------------------------------------------------------
# Step 6 — compare against the Katongole CSV + plot
# ---------------------------------------------------------------------------


def compare_and_plot(
    locations: tuple[LocationSpec, ...],
    climatology_2024: pd.DataFrame,
    climatology_window: pd.DataFrame,
) -> None:
    katongole = pd.read_csv(KATONGOLE_CSV)
    katongole_long = katongole[katongole["location"].isin(EXPERIMENT_STATIONS)].melt(
        id_vars=["location"],
        value_vars=MONTH_NAMES,
        var_name="month_name", value_name="monthly_obs",
    )
    katongole_long["month"] = katongole_long["month_name"].apply(
        lambda m: MONTH_NAMES.index(m) + 1
    )
    katongole_long = katongole_long.drop(columns="month_name")

    def merge_and_score(climo: pd.DataFrame, label: str) -> pd.DataFrame:
        merged = katongole_long.merge(climo, on=["location", "month"], how="inner")
        rows = []
        for loc, sub in merged.groupby("location"):
            err = sub["y_pred"] - sub["monthly_obs"]
            rows.append({
                "location": loc, "window": label,
                "rf_rmse": float(np.sqrt((err ** 2).mean())),
                "rf_mbe":  float(err.mean()),
                "n_months": int(len(sub)),
            })
        return pd.DataFrame(rows)

    table = pd.concat([
        merge_and_score(climatology_2024, "2024"),
        merge_and_score(climatology_window,
                        f"{EXPERIMENT_DATE_START.year}-{EXPERIMENT_DATE_END.year}"),
    ])
    pivot = table.pivot(index="location", columns="window",
                        values=["rf_rmse", "rf_mbe"]).round(3)
    print("\n=== Year-mismatch experiment ===")
    print(pivot.to_string())
    pivot.to_csv(CSV_PATH)
    log.info("Saved comparison table → %s", CSV_PATH)

    # Plot: 4 stations per row × 2 stations per col-group, RF in two
    # windows + Katongole + raw NASA / CAMS climatologies overlaid.
    import matplotlib.pyplot as plt
    n_stations = len(EXPERIMENT_STATIONS)
    ncols = min(3, n_stations)
    nrows = (n_stations + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True)
    flat_axes = list(np.array(axes).flat) if hasattr(axes, "flat") else [axes]

    for ax, station_name in zip(flat_axes, EXPERIMENT_STATIONS):
        obs = (
            katongole_long[katongole_long["location"] == station_name]
            .sort_values("month")
        )
        c2024 = (
            climatology_2024[climatology_2024["location"] == station_name]
            .sort_values("month")
        )
        cwin = (
            climatology_window[climatology_window["location"] == station_name]
            .sort_values("month")
        )
        ax.plot(obs["month"], obs["monthly_obs"], "-o", ms=3,
                color="C0", label="Observed (Katongole 2017-2022)")
        ax.plot(c2024["month"], c2024["y_pred"], "-o", ms=3,
                color="C3", label="RF (2024)")
        ax.plot(cwin["month"], cwin["y_pred"], "-s", ms=3,
                color="C4", label=f"RF ({EXPERIMENT_DATE_START.year}-{EXPERIMENT_DATE_END.year})")
        ax.plot(cwin["month"], cwin["sat_ghi_nasa_kwh_m2_day"], "-",
                lw=0.8, color="C2", alpha=0.7,
                label=f"NASA ({EXPERIMENT_DATE_START.year}-{EXPERIMENT_DATE_END.year})")
        ax.plot(cwin["month"], cwin["sat_ghi_cams_kwh_m2_day"], "-",
                lw=0.8, color="C1", alpha=0.7,
                label=f"CAMS ({EXPERIMENT_DATE_START.year}-{EXPERIMENT_DATE_END.year})")
        ax.set_title(station_name, fontsize=10)
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3)
    flat_axes[0].legend(fontsize=7, loc="lower left")
    fig.supxlabel("Month")
    fig.supylabel("GHI (kWh/m²/day)")
    fig.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    log.info("Saved plot → %s", PLOT_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    locations = load_experiment_stations()
    log.info("Step 2 — ingesting NASA + CAMS for %s", locations)
    run_ingest(locations)

    log.info("Step 3 — loading trained bundle from %s", BUNDLE_DIR)
    bundle = load_bundle(
        BUNDLE_DIR, providers={"altitude": PvlibElevationProvider()},
    )

    log.info("Step 4a — building inference frame for the climatology window")
    inf_window = build_inference_frame(
        locations, EXPERIMENT_DATE_START, EXPERIMENT_DATE_END,
    )
    log.info("Step 4b — building inference frame for the 2024 baseline window")
    inf_2024 = build_inference_frame(
        locations, date(BASELINE_YEAR, 1, 1), date(BASELINE_YEAR, 12, 31),
    )

    log.info("Step 5 — predicting + aggregating climatology")
    climo_window = predict_and_climatology(
        inf_window, bundle,
        label=f"{EXPERIMENT_DATE_START.year}-{EXPERIMENT_DATE_END.year}",
    )
    climo_2024 = predict_and_climatology(
        inf_2024, bundle, label=str(BASELINE_YEAR),
    )

    log.info("Step 6 — comparing against Katongole CSV")
    compare_and_plot(locations, climo_2024, climo_window)


if __name__ == "__main__":
    main()
