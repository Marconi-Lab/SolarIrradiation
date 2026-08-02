"""Paper-specific helpers for the Mukiibi & Mikelson (2026) recomputation.

Everything here is bound to *this* paper: the 56-station Katongole
benchmark, migration A12's warehouse layout, the paper's two custom
figure layouts (the calibration-discrepancy plot and the 4 × 4
Figure 2 panel grid), and the deterministic station-selection rule
that picks the paper's 24 training + 2 holdout split.

Generic evaluation plots — scatter panels, per-station bars,
covariate-shift KDEs, training-fit time series, feature importances —
were moved to :mod:`susse.evaluation.plots` once the second
consumer appeared. Import those directly:

    from susse.evaluation.plots import (
        plot_target_distribution_per_station,
        plot_predictor_vs_observed_scatter,
        plot_feature_correlation_heatmap,
        plot_pca_by_station,
        plot_training_fit_timeseries,
        plot_training_fit_scatter,
        plot_covariate_shift_kde,
        plot_predicted_vs_observed_panels,
        plot_per_station_metric,
        plot_feature_importances_top_n,
    )

If a function in this module starts looking generic, move it to
``src/susse/evaluation/plots.py`` and import it back here.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from susse.datasets import FeatureSelection
    from susse.warehouse_ops.io.bq import BigQueryClient
    from susse.warehouse_ops.io.config import TableRefs
    from susse.warehouse_ops.io.repositories import SatelliteRepository


# ---------------------------------------------------------------------------
# Sanity checks — silent on success, raise on failure.
# ---------------------------------------------------------------------------


def assert_katongole_coverage(
    bq: "BigQueryClient",
    tables: "TableRefs",
    katongole: pd.DataFrame,
    val_start: date,
    val_end: date,
    *,
    min_days_fraction: float = 0.95,
    allow_partial: bool = False,
) -> None:
    """Raise if any Katongole station lacks NASA + CAMS coverage over the window.

    Args:
        bq: Live BigQuery client.
        tables: Resolved warehouse table references.
        katongole: DataFrame with a ``geohash5`` column for every station.
        val_start, val_end: Inclusive bounds of the validation window.
        min_days_fraction: A station is "full" when both NASA and CAMS have
            at least this fraction of the expected ``val_end - val_start``
            days. Default 0.95 tolerates ~1 missing day per month.
        allow_partial: When True, raise only when *no* station has any data
            (development mode). Production runs leave this False so missing
            stations fail loud.

    Raises:
        RuntimeError: When any station is below the coverage threshold.
            The message names the migration that ingests the missing data.
    """
    n_expected = (val_end - val_start).days + 1
    min_days = int(min_days_fraction * n_expected)
    geohashes = ", ".join(f"'{g}'" for g in katongole["geohash5"].unique())
    coverage = bq.query(f"""
        SELECT geohash5,
               COUNT(DISTINCT IF(source='NASA', date, NULL)) AS n_nasa_days,
               COUNT(DISTINCT IF(source='CAMS', date, NULL)) AS n_cams_days
        FROM `{tables.irradiance_daily}`
        WHERE date BETWEEN DATE('{val_start}') AND DATE('{val_end}')
          AND geohash5 IN ({geohashes})
        GROUP BY geohash5
    """)
    per_station = katongole[["location", "geohash5"]].merge(
        coverage, on="geohash5", how="left",
    ).fillna({"n_nasa_days": 0, "n_cams_days": 0}).astype(
        {"n_nasa_days": int, "n_cams_days": int}
    )
    is_full = (per_station["n_nasa_days"] >= min_days) & (
        per_station["n_cams_days"] >= min_days
    )
    is_missing = (per_station["n_nasa_days"] == 0) & (
        per_station["n_cams_days"] == 0
    )
    if allow_partial:
        if is_missing.all():
            raise RuntimeError(
                "No Katongole station has any warehouse coverage. "
                "Apply migration A12: warehouse/migrations/"
                "2026-05-11_a12_ingest_katongole_2017_2022.py --apply."
            )
        return
    if not is_full.all():
        incomplete = per_station[~is_full]
        raise RuntimeError(
            f"{len(incomplete)} of {len(per_station)} Katongole stations "
            f"have incomplete warehouse coverage for {val_start}..{val_end}.\n"
            f"To fix: .venv/bin/python warehouse/migrations/"
            f"2026-05-11_a12_ingest_katongole_2017_2022.py --apply\n"
            f"Pass allow_partial=True to bypass this assertion during "
            f"development.\n"
            f"Incomplete stations:\n"
            f"{incomplete[['location', 'n_nasa_days', 'n_cams_days']].to_string(index=False)}"
        )


def assert_inference_complete(
    katongole: pd.DataFrame, inference_df: pd.DataFrame,
) -> None:
    """Raise if any Katongole station is absent from the inference frame.

    Defence-in-depth: the pre-flight should have caught this, but a merge
    glitch downstream could still drop rows. A second check at the call
    site is cheap.
    """
    expected = set(katongole["location"])
    present = set(inference_df["location"])
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(
            f"{len(missing)} stations are absent from the inference frame "
            f"despite the pre-flight passing. Investigate before trusting "
            f"downstream metrics. Missing: {missing}"
        )


# ---------------------------------------------------------------------------
# Deterministic station selection.
#
# Operates on ``station_meta`` (a row-per-station summary frame) to pick the
# paper's 24 training + 2 holdout split *before* loading per-row data. This
# is distinct from :class:`susse.training.SpatialSpreadHoldoutSplitter`,
# which selects rows of an already-loaded :class:`PreprocessedDataset`. Both
# implement the same max-pairwise-haversine idea at different pipeline
# stages.
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    r_earth_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(d_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    )
    return float(2 * r_earth_km * np.arcsin(np.sqrt(a)))

def pick_training_and_holdout_stations(
    station_meta: pd.DataFrame,
    *,
    n_training: int,
    n_holdout: int,
    holdout_stations: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select training + spatial-holdout stations by a deterministic rule.

    The rule (paper Table I + III equivalent):

    1. Sort the available stations by total QC-passed row count and keep the
       top ``n_training`` — drops the shortest-coverage stations first.
    2. From those, pick the ``n_holdout``-station subset with the largest
       pairwise haversine spread; greedy pairwise maximum for ``n_holdout=2``.
       Skipped entirely if ``holdout_stations`` is given.

    Args:
        holdout_stations: Optional explicit holdout names (e.g. ``("Site A",
            "Site B")`` to pin the paper's known split) to use instead of
            the greedy haversine-max search. Must contain exactly
            ``n_holdout`` names, each present among the top ``n_training``
            candidates by row count.

    Returns:
        ``(training_stations, holdout_stations)`` as tuples of location
        names. ``training_stations`` excludes the holdout set.

    Raises:
        ValueError: ``n_training`` exceeds the rows in ``station_meta``,
            ``n_holdout`` doesn't satisfy ``1 <= n_holdout < n_training``,
            or ``holdout_stations`` is malformed / not a subset of the
            candidate pool.
    """
    if len(station_meta) < n_training:
        raise ValueError(
            f"Only {len(station_meta)} stations available but n_training="
            f"{n_training}. Lower n_training or ingest more stations."
        )
    if n_holdout >= n_training or n_holdout < 1:
        raise ValueError(
            f"n_holdout={n_holdout} must satisfy 1 <= n_holdout < n_training "
            f"(n_training={n_training})."
        )

    candidates = station_meta.nlargest(n_training, "n_rows").reset_index(drop=True)
    candidate_names = set(candidates["location"])

    if holdout_stations is not None:
        holdout_stations = tuple(holdout_stations)
        if len(holdout_stations) != n_holdout:
            raise ValueError(
                f"holdout_stations has {len(holdout_stations)} name(s) but "
                f"n_holdout={n_holdout}."
            )
        unknown = [s for s in holdout_stations if s not in set(station_meta["location"])]
        if unknown:
            raise ValueError(f"holdout_stations not found in station_meta: {unknown}")
        not_in_candidates = [s for s in holdout_stations if s not in candidate_names]
        if not_in_candidates:
            raise ValueError(
                f"holdout_stations {not_in_candidates} fall outside the top "
                f"{n_training} stations by n_rows; raise n_training or pick "
                "different holdout stations."
            )
        training_stations = tuple(
            s for s in candidates["location"] if s not in holdout_stations
        )
        return training_stations, holdout_stations

    if n_holdout != 2:
        raise NotImplementedError(
            f"Only n_holdout=2 is implemented (matching paper Table III); "
            f"got {n_holdout}. Extend pick_training_and_holdout_stations to "
            f"select a larger holdout set by k-medoids or similar, or pass "
            "holdout_stations explicitly."
        )
    best_distance = -1.0
    holdout_pair: tuple[str, str] = ("", "")
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            d_km = _haversine_km(
                candidates.loc[i, "lat"], candidates.loc[i, "lon"],
                candidates.loc[j, "lat"], candidates.loc[j, "lon"],
            )
            if d_km > best_distance:
                best_distance = d_km
                holdout_pair = (
                    candidates.loc[i, "location"],
                    candidates.loc[j, "location"],
                )
    training_stations = tuple(
        s for s in candidates["location"] if s not in holdout_pair
    )
    return training_stations, holdout_pair

# ---------------------------------------------------------------------------
# Katongole inference frame builder.
# ---------------------------------------------------------------------------


def build_katongole_inference_frame(
    *,
    bq: "BigQueryClient",
    tables: "TableRefs",
    sat_repo: "SatelliteRepository",
    katongole: pd.DataFrame,
    selection: "FeatureSelection",
    nasa_aux_ids: tuple[str, ...],
    cams_aux_ids: tuple[str, ...],
    val_start: date,
    val_end: date,
) -> pd.DataFrame:
    """Assemble the daily inference frame for every Katongole station.

    After migration A12 each Katongole station has its own warehouse cell,
    so the merge is on the station's own ``geohash5`` — no
    snap-to-nearest-cell logic is needed.

    Returns:
        DataFrame with one row per (station, date) over ``val_start..val_end``,
        carrying ``location``, ``lat``, ``lon``, ``geohash5``, ``date``,
        every ``sat_<band>_<source>_kwh_m2_day`` column, every NASA aux
        column (prefix ``nasa_``), and every CAMS aux column (prefix
        ``cams_``). Suitable to pass through the same ``Preprocessor`` as
        the training frame.
    """
    katongole_geohashes = tuple(katongole["geohash5"].unique())
    irr_wide = sat_repo.daily_irradiance_by_geohash(
        val_start, val_end,
        sources=("NASA", "CAMS"),
        bands=selection.include_satellite_bands,
        geohash5s=katongole_geohashes,
    )
    nasa_aux = sat_repo.long_aux_pivoted(
        table_fqn=tables.nasa_daily_vars_long,
        column_prefix="nasa",
        start=val_start, end=val_end,
        variable_ids=nasa_aux_ids,
        geohash5s=katongole_geohashes,
    )
    cams_aux = sat_repo.long_aux_pivoted(
        table_fqn=tables.cams_daily_vars_long,
        column_prefix="cams",
        start=val_start, end=val_end,
        variable_ids=cams_aux_ids,
        geohash5s=katongole_geohashes,
    )
    geohash_features = (
        irr_wide
        .merge(nasa_aux, on=["date", "geohash5"], how="left")
        .merge(cams_aux, on=["date", "geohash5"], how="left")
    )
    inference_df = katongole[
        ["location", "latitude", "longitude", "geohash5"]
    ].merge(geohash_features, on="geohash5", how="inner").rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )
    return inference_df


# ---------------------------------------------------------------------------
# Paper-specific figure layouts.
#
# Generic versions of every other plot moved to susse.evaluation.plots; only
# layouts that hardcode paper-specific series (Makerere co-location, the
# 4 × 4 Figure 2 grid) live here.
# ---------------------------------------------------------------------------


def plot_calibration_discrepancy(calibration_df: pd.DataFrame) -> None:
    """Co-located comparison at Makerere — two ground sensors + 2 satellites.

    Paper-specific: the four series and their labels are pinned to the
    Mukiibi-Mikelson §6.3 narrative.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(calibration_df["month"], calibration_df["kampala_mean"], "-o",
            ms=4, color="C0", label="Our kampala (MEMD/CB pyranometer)")
    ax.plot(calibration_df["month"], calibration_df["makerere_mean"], "-s",
            ms=4, color="C4", label="Katongole Makerere S (TAHMO ATMOS 41)")
    ax.plot(calibration_df["month"], calibration_df["NASA"], "-", lw=0.8,
            color="C2", alpha=0.7, label="NASA CERES (2017-2022)")
    ax.plot(calibration_df["month"], calibration_df["CAMS"], "-", lw=0.8,
            color="C1", alpha=0.7, label="CAMS (2017-2022)")
    ax.set_xlabel("Month")
    ax.set_ylabel("GHI (kWh/m²/day)")
    ax.set_xticks(range(1, 13))
    ax.set_title(
        "Co-located comparison at Makerere University, 2017-2022\n"
        "Same lat/lon — different pyranometer networks differ by ~14% (kampala higher)"
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_figure_2_grid(
    comparison_calibrated: pd.DataFrame,
    *,
    stations: Iterable[str] | None = None,
    validation_label: str = "2017-2022",
    n_panels: int = 16,
    observed: bool = True,
) -> list[str]:
    """Paper-style 4 × 4 monthly GHI grid for 16 stations.

    Paper-specific: the five series per panel (measured raw,
    measured calibrated, RF predicted, NASA, CAMS), the panel-code
    overlays (a1..d4), and the German-paper-style axis labels are
    pinned to Mukiibi-Mikelson Figure 2.

    When ``stations`` is None, picks 4 stations per RMSE quartile so the
    figure spans the model's full performance range. Returns the selected
    station list so the caller can re-use it for tables or a custom legend.
    """
    errs = (
        comparison_calibrated["monthly_pred"]
        - comparison_calibrated["monthly_obs_calibrated"]
    )
    per_station_rmse = (
        (errs ** 2)
        .groupby(comparison_calibrated["location"])
        .mean()
        .pow(0.5)
        .sort_values()
    )
    if stations is None:
        n = len(per_station_rmse)
        picked: list[str] = []
        for q in range(4):
            start = (q * n) // 4
            picked.extend(per_station_rmse.index[start: start + 4].tolist())
        stations = picked[:n_panels]
    stations = list(stations)
    if len(stations) != n_panels:
        raise ValueError(
            f"plot_figure_2_grid expects exactly {n_panels} stations, "
            f"got {len(stations)}."
        )

    panel_codes = [f"{r}{c}" for r in ("a", "b") for c in (1, 2)]
    panel_codes = panel_codes + [f"{r}{c}" for r in ("a", "b") for c in (3, 4)]
    panel_codes = panel_codes + [f"{r}{c}" for r in ("c", "d") for c in (1, 2)]
    panel_codes = panel_codes + [f"{r}{c}" for r in ("c", "d") for c in (3, 4)]
    month_ticks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    for ax, station, code in zip(axes.flat, stations, panel_codes):
        sub = comparison_calibrated[
            comparison_calibrated["location"] == station
        ].sort_values("month")
        if observed:
            ax.plot(sub["month"], sub["monthly_obs"], "-",
                    lw=0.7, color="C0", alpha=0.35,
                    label="Measured (TAHMO, raw)")
        ax.plot(sub["month"], sub["monthly_obs_calibrated"], "-o",
                lw=1.4, ms=4, color="C0",
                label="Measured (calibrated)")
        ax.plot(sub["month"], sub["monthly_pred"], "-o",
                lw=1.4, ms=5, color="C3", label="RF Predicted")
        ax.plot(sub["month"], sub["sat_ghi_nasa_kwh_m2_day"], "-",
                lw=0.8, alpha=0.7, color="C2", label="NASA GHI")
        ax.plot(sub["month"], sub["sat_ghi_cams_kwh_m2_day"], "-",
                lw=0.8, alpha=0.7, color="C1", label="CAMS GHI")
        ax.set_title(
            f"GHI comparison for {station} ({validation_label})",
            fontsize=9,
        )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_ticks, rotation=45, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)
        ax.text(
            0.965, 0.955, code, transform=ax.transAxes,
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="green", lw=1.0),
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Solar irradiation (kWh m⁻² day⁻¹)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Month of the Year", fontsize=8)
    axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    plt.show()
    return stations


# ---------------------------------------------------------------------------
# Scoring table.
# ---------------------------------------------------------------------------


def score_table_iv(
    comparison: pd.DataFrame, *,
    label: str | None = None,
    obs_col: str = "monthly_obs",
    rf_label: str = "RF",
) -> pd.DataFrame:
    """Cross-station mean of paper-Table-IV style metrics.

    Thin paper-specific wrapper over :class:`susse.evaluation.Evaluator`
    that hardcodes the RF / NASA CERES / CAMS comparison series and
    relabels the Evaluator's ``prediction`` column to ``model`` to
    match the paper's table layout.

    Returns a DataFrame with rows for RF, NASA CERES and CAMS evaluated
    against ``obs_col``. Pass ``obs_col="monthly_obs_calibrated"`` for the
    cross-network-calibrated comparison.
    """
    from susse.evaluation import Evaluator

    if label is None:
        label = f"{comparison['location'].nunique()} stations"
    table = Evaluator().score(
        observed=comparison[obs_col],
        predictions={
            rf_label:     comparison["monthly_pred"],
            "NASA CERES": comparison["sat_ghi_nasa_kwh_m2_day"],
            "CAMS":       comparison["sat_ghi_cams_kwh_m2_day"],
        },
        splits={label: comparison.index},
    ).rename(columns={"prediction": "model"})
    return table.round({"RMSE": 3, "nRMSE_%": 2, "MAE": 3, "nMAE_%": 2,
                        "MBE": 3, "R²": 3, "IOA": 3})

# ---------------------------------------------------------------------------
# Model comparison — paper Table III.
# ---------------------------------------------------------------------------

def run_model_comparison(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    train_stations,
    sat_val_nasa=None,
    sat_val_cams=None,
    # Random Forest Params
    rf_n_estimators: int = 300,
    rf_min_samples_leaf: int = 2,
    rf_max_features: float = 0.5,
    rf_max_depth = None,
    # XGBoost Params
    xgb_n_estimators: int = 200,
    xgb_learning_rate: float = 0.1,
    xgb_max_depth: int = 6,
    xgb_subsample: float = 1.0,
    xgb_colsample_bytree: float = 1.0,
    # SVR Params
    svr_C: float = 10.0,
    svr_epsilon: float = 0.1,
    svr_kernel: str = "rbf",
    # Ridge Params
    ridge_alpha: float = 0.01,
    # Global Settings
    random_state: int = 42,
    run_grid_search: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Train and evaluate all paper candidate models on the held-out fold."""
    import time
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import (
        make_scorer,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        GroupKFold,
        RandomizedSearchCV,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    # ── Metric helpers ────────────────────────────────────────────────────────
    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _mbe(y_true, y_pred):
        return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))

    def _metrics(y_true, y_pred, label=""):
        r = _rmse(y_true, y_pred)
        m = float(mean_absolute_error(y_true, y_pred))
        return {
            "label": label,
            "RMSE": r,
            "MAE": m,
            "MBE": _mbe(y_true, y_pred),
            "R2": float(r2_score(y_true, y_pred)),
        }

    # ── Candidate configs ─────────────────────────────────────────────────────
    search_configs = {
        "Random Forest": dict(
            paper_params=dict(
                n_estimators=rf_n_estimators,
                min_samples_leaf=rf_min_samples_leaf,
                max_features=rf_max_features,
                max_depth=rf_max_depth,
                random_state=random_state,
                n_jobs=-1,
            ),
            estimator=RandomForestRegressor(
                random_state=random_state, n_jobs=-1
            ),
            param_grid={
                "n_estimators": [100, 200, 300],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", 0.3, 0.5, 0.7],
                "max_depth": [None, 20, 30],
            },
        ),
        "XGBoost": dict(
            paper_params=dict(
                n_estimators=xgb_n_estimators,
                learning_rate=xgb_learning_rate,
                max_depth=xgb_max_depth,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            ),
            estimator=XGBRegressor(
                random_state=random_state, n_jobs=-1, verbosity=0
            ),
            param_grid={
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 6, 9],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            },
            use_random_search=True,
            n_iter=60,
        ),
        "SVR": dict(
            # SVR grid search is prohibitively slow — always use paper params.
            paper_params=dict(
                kernel=svr_kernel, 
                C=svr_C, 
                epsilon=svr_epsilon
            ),
            estimator=Pipeline([("scaler", StandardScaler()), ("svr", SVR())]),
            param_grid={},
            skip_search=True,
        ),
        "Linear Regression": dict(
            paper_params=dict(n_jobs=-1),
            estimator=LinearRegression(n_jobs=-1),
            param_grid={},
            skip_search=True,
        ),
        "Ridge Regression": dict(
            paper_params=dict(
                alpha=ridge_alpha, 
                random_state=random_state
            ),
            estimator=Ridge(random_state=random_state),
            param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        ),
    }

    cv = GroupKFold(n_splits=5)
    rmse_scorer = make_scorer(
        _rmse, greater_is_better=False
    )

    # ── Header ────────────────────────────────────────────────────────────────
    if run_grid_search:
        print("=" * 65)
        print("Hyperparameter search — Spatial CV on training fold only")
        print("(set RUN_GRID_SEARCH=False to skip and use paper params)")
        print("=" * 65)
    else:
        print("=" * 65)
        print("Using paper best-known hyperparameters (RUN_GRID_SEARCH=False)")
        print("Set RUN_GRID_SEARCH=True to re-run the full search (~2-6 hours on consumer laptop)")
        print("=" * 65)

    # ── Fit models ────────────────────────────────────────────────────────────
    final_models: dict = {}
    for name, cfg in search_configs.items():
        t0 = time.time()
        skip = cfg.get("skip_search") or not run_grid_search

        if skip:
            if isinstance(cfg["estimator"], Pipeline):
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svr", SVR(**cfg["paper_params"])),
                ])
            else:
                model = cfg["estimator"].__class__(**cfg["paper_params"])
            model.fit(X_train, y_train)
            final_models[name] = model
            src = "paper params" if not cfg.get("skip_search") else "fixed"
            print(f"  {name:<22}  fitted with {src}  ({time.time()-t0:.1f}s)")

        elif cfg.get("use_random_search"):
            search = RandomizedSearchCV(
                cfg["estimator"], cfg["param_grid"],
                n_iter=cfg.get("n_iter", 50), scoring=rmse_scorer,
                cv=cv, n_jobs=-1, refit=True,
                random_state=random_state, verbose=0,
            )
            search.fit(X_train, y_train, groups=train_stations)
            final_models[name] = search.best_estimator_
            print(
                f"  {name:<22}  CV RMSE={-search.best_score_:.4f}  "
                f"({(time.time()-t0)/60:.1f} min)  best={search.best_params_}"
            )
        else:
            search = GridSearchCV(
                cfg["estimator"], cfg["param_grid"],
                scoring=rmse_scorer, cv=cv, n_jobs=-1, refit=True, verbose=0,
            )
            search.fit(X_train, y_train, groups=train_stations)
            final_models[name] = search.best_estimator_
            print(
                f"  {name:<22}  CV RMSE={-search.best_score_:.4f}  "
                f"({(time.time()-t0)/60:.1f} min)  best={search.best_params_}"
            )

    # ── Score on held-out fold ────────────────────────────────────────────────
    rows = [
        _metrics(y_val, model.predict(X_val), label=name)
        for name, model in final_models.items()
    ]
    if sat_val_nasa is not None:
        rows.append(_metrics(y_val, sat_val_nasa, label="NASA CERES"))
    if sat_val_cams is not None:
        rows.append(_metrics(y_val, sat_val_cams, label="CAMS"))

    table3_df = pd.DataFrame(rows).set_index("label")
    return final_models, table3_df