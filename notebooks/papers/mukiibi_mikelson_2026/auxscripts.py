"""Auxiliary helpers for the Mukiibi & Mikelson (2026) recomputation notebook.

These are functions that would otherwise clutter the notebook's flow:
sanity-check assertions, the longer query orchestration, plot builders,
and the scoring tables. Imported once at the top of
``01_recomputation.ipynb`` as ``auxscripts`` so the notebook reads as a
methodology overview, not a wall of plt styling.

Anything in this file is paper-recomputation-specific. Move it into the
library (`src/susse/`) if a second use-case appears.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from susse.datasets import FeatureSelection
    from susse.training import TrainedBundle
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
    station_meta: pd.DataFrame, *, n_training: int, n_holdout: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select training + spatial-holdout stations by a deterministic rule.

    The rule (paper Table I + III equivalent):

    1. Sort the available stations by total QC-passed row count and keep the
       top ``n_training`` — drops the shortest-coverage stations first.
    2. From those, pick the ``n_holdout``-station subset with the largest
       pairwise haversine spread; greedy pairwise maximum for ``n_holdout=2``.

    Returns:
        ``(training_stations, holdout_stations)`` as tuples of location
        names. ``training_stations`` excludes the holdout set.

    Raises:
        ValueError: ``n_training`` exceeds the rows in ``station_meta``,
            or ``n_holdout`` >= ``n_training``.
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
    if n_holdout != 2:
        raise NotImplementedError(
            f"Only n_holdout=2 is implemented (matching paper Table III); "
            f"got {n_holdout}. Extend pick_training_and_holdout_stations to "
            f"select a larger holdout set by k-medoids or similar."
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
# Plots — one function per figure, no shared state.
# ---------------------------------------------------------------------------


def plot_target_distribution_per_station(processed_df: pd.DataFrame) -> None:
    """Boxplot + per-station KDE of the absolute GHI target."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    processed_df.boxplot(
        column="y_ghi_kwh_m2_day", by="location",
        rot=90, ax=axes[0], grid=False,
    )
    axes[0].set_title("Target GHI per training station")
    axes[0].set_ylabel("y_ghi (kWh/m²/day)")
    axes[0].set_xlabel("")
    plt.suptitle("")

    ax = axes[1]
    locs = sorted(processed_df["location"].unique())
    for loc in locs:
        sub = processed_df.loc[processed_df["location"] == loc, "y_ghi_kwh_m2_day"]
        if len(sub) > 1:
            sub.plot.kde(ax=ax, alpha=0.5, lw=0.7)
    ax.set_xlabel("y_ghi (kWh/m²/day)")
    ax.set_ylabel("density")
    ax.set_title(f"KDE per station ({len(locs)} stations)")
    fig.tight_layout()
    plt.show()


def plot_satellite_vs_observed_scatter(processed_df: pd.DataFrame) -> None:
    """NASA / CAMS GHI vs ground truth — the bias the model has to correct."""
    y_obs = processed_df["y_ghi_kwh_m2_day"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, sat_col, label in zip(
        axes,
        ("sat_ghi_nasa_kwh_m2_day", "sat_ghi_cams_kwh_m2_day"),
        ("NASA CERES", "CAMS"),
    ):
        ax.scatter(y_obs, processed_df[sat_col], s=3, alpha=0.2, color="C0")
        lim = (0, 10)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Observed GHI (kWh/m²/day)")
        ax.set_ylabel(f"{label} GHI (kWh/m²/day)")
        err = processed_df[sat_col] - y_obs
        ax.set_title(
            f"{label} vs observed  "
            f"(MBE={err.mean():.2f}, RMSE={np.sqrt((err ** 2).mean()):.2f})"
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_feature_correlation_heatmap(
    processed_df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    *,
    top_n: int = 10,
) -> None:
    """Pairwise feature × target correlation heatmap + top-N table."""
    cols = list(feature_cols) + [target_col]
    corr = processed_df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title("Feature correlation matrix (training data)")
    fig.tight_layout()
    plt.show()

    target_corr = (
        corr[target_col].drop(target_col).abs().sort_values(ascending=False)
    )
    print(f"Top {top_n} features by |correlation| with {target_col}:")
    print(target_corr.head(top_n).round(3).to_string())


def plot_pca_by_station(
    processed_df: pd.DataFrame, feature_cols: Iterable[str],
) -> None:
    """PCA(2) of standardised features, points coloured by station."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = processed_df[list(feature_cols)].to_numpy()
    pcs = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(x))
    pca = PCA(n_components=2).fit(StandardScaler().fit_transform(x))
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    loc_arr = processed_df["location"].to_numpy()
    unique_locs = sorted(set(loc_arr))
    cmap = plt.get_cmap("tab20", max(20, len(unique_locs)))
    for i, loc in enumerate(unique_locs):
        mask = loc_arr == loc
        ax.scatter(
            pcs[mask, 0], pcs[mask, 1], s=4, alpha=0.5,
            color=cmap(i % cmap.N), label=loc,
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} var)")
    ax.set_title("PCA of training features — coloured by station")
    ax.legend(fontsize=6, ncol=3, loc="best", framealpha=0.7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_training_fit_timeseries(
    diag_df: pd.DataFrame, *, n_stations: int = 6,
) -> None:
    """Weekly-resampled time series of observed vs RF vs NASA vs CAMS.

    Picks the ``n_stations`` stations with the most rows, restricts to
    each station's most recent calendar year for legibility.
    """
    plot_stations = (
        diag_df.groupby("location").size()
        .sort_values(ascending=False).head(n_stations).index.tolist()
    )
    fig, axes = plt.subplots(
        (n_stations + 1) // 2, 2, figsize=(16, 3 * ((n_stations + 1) // 2)),
        sharey=True,
    )
    for ax, loc in zip(np.atleast_1d(axes).flat, plot_stations):
        sub = diag_df[diag_df["location"] == loc].sort_values("date").copy()
        last_year = sub["date"].dt.year.max()
        sub = sub[sub["date"].dt.year == last_year]
        weekly = (
            sub.set_index("date")[
                ["y_ghi_kwh_m2_day", "y_pred",
                 "sat_ghi_nasa_kwh_m2_day", "sat_ghi_cams_kwh_m2_day"]
            ].resample("W").mean()
        )
        ax.plot(weekly.index, weekly["y_ghi_kwh_m2_day"], "-o", ms=3,
                label="Observed", color="C0")
        ax.plot(weekly.index, weekly["y_pred"], "-o", ms=3,
                label="RF (in-sample)", color="C3")
        ax.plot(weekly.index, weekly["sat_ghi_nasa_kwh_m2_day"], "-",
                lw=0.8, label="NASA", color="C2", alpha=0.7)
        ax.plot(weekly.index, weekly["sat_ghi_cams_kwh_m2_day"], "-",
                lw=0.8, label="CAMS", color="C1", alpha=0.7)
        ax.set_title(f"{loc} ({last_year}, n={len(sub):,})", fontsize=10)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes).flat[0].legend(fontsize=8, loc="lower left")
    fig.supylabel("GHI (kWh/m²/day)")
    fig.tight_layout()
    plt.show()


def plot_training_fit_scatter(diag_df: pd.DataFrame) -> None:
    """In-sample predicted vs observed; tight diagonal = healthy training."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(diag_df["y_ghi_kwh_m2_day"], diag_df["y_pred"],
               s=2, alpha=0.25, color="C3")
    hi = max(diag_df["y_ghi_kwh_m2_day"].max(), diag_df["y_pred"].max()) * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("Observed GHI (kWh/m²/day)")
    ax.set_ylabel("RF predicted GHI (kWh/m²/day) — in-sample")
    ax.set_title("Training fit: RF prediction vs observed (every training row)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    err = diag_df["y_pred"] - diag_df["y_ghi_kwh_m2_day"]
    print(
        f"In-sample training fit (absolute scale):  "
        f"RMSE = {np.sqrt((err ** 2).mean()):.3f}  "
        f"MAE = {np.abs(err).mean():.3f}  "
        f"MBE = {err.mean():.3f}"
    )


def plot_covariate_shift_kde(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: Iterable[str],
) -> None:
    """Overlaid KDEs of training vs Katongole feature distributions.

    Skips features not present on either frame so the call site can be
    a constant list of candidates without guarding each one.
    """
    keep = [f for f in features if f in train_df.columns and f in val_df.columns]
    n_cols = 3
    n_rows = (len(keep) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    for ax, feat in zip(np.array(axes).flat, keep):
        t = train_df[feat].dropna().to_numpy()
        v = val_df[feat].dropna().to_numpy()
        lo = float(np.nanquantile(np.r_[t, v], 0.005))
        hi = float(np.nanquantile(np.r_[t, v], 0.995))
        ax.hist(t, bins=40, range=(lo, hi), alpha=0.5,
                density=True, label=f"Training (n={len(t):,})")
        ax.hist(v, bins=40, range=(lo, hi), alpha=0.5,
                density=True, label=f"Katongole (n={len(v):,})")
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)
    for ax in np.array(axes).flat[len(keep):]:
        ax.axis("off")
    fig.suptitle("Feature distributions: training vs Katongole inference")
    fig.tight_layout()
    plt.show()


def plot_calibration_discrepancy(calibration_df: pd.DataFrame) -> None:
    """Co-located comparison at Makerere — two ground sensors + 2 satellites."""
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


def plot_predicted_vs_observed_calibrated(
    comparison_calibrated: pd.DataFrame,
) -> None:
    """Three-panel monthly scatter: RF vs NASA vs CAMS, all 54x12 points."""
    series = [
        ("monthly_pred",            "RF (calibrated)", "C3"),
        ("sat_ghi_nasa_kwh_m2_day", "NASA CERES",      "C2"),
        ("sat_ghi_cams_kwh_m2_day", "CAMS",            "C1"),
    ]
    obs_col = "monthly_obs_calibrated"
    lim = (
        float(np.floor(
            comparison_calibrated[[obs_col] + [c for c, _, _ in series]].min().min()
        )),
        float(np.ceil(
            comparison_calibrated[[obs_col] + [c for c, _, _ in series]].max().max()
        )),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
    for ax, (col, label, color) in zip(axes, series):
        ax.scatter(
            comparison_calibrated[obs_col], comparison_calibrated[col],
            s=14, alpha=0.55, color=color,
        )
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Katongole observed (calibrated) (kWh/m²/day)")
        ax.set_ylabel(f"{label} (kWh/m²/day)")
        err = (comparison_calibrated[col] - comparison_calibrated[obs_col]).dropna()
        ax.set_title(
            f"{label}\nMBE={err.mean():.2f}  RMSE={np.sqrt((err ** 2).mean()):.2f}"
        )
        ax.grid(alpha=0.3)
    fig.suptitle("Katongole monthly: predicted vs calibrated observed")
    fig.tight_layout()
    plt.show()


def plot_per_station_mbe(comparison_calibrated: pd.DataFrame) -> None:
    """Per-station MBE bar chart; geohash-overlap stations coloured."""
    per_station = (
        comparison_calibrated.assign(
            rf_residual=lambda d: d["monthly_pred"] - d["monthly_obs_calibrated"],
        )
        .groupby("location")
        .agg(
            rf_mbe=("rf_residual", "mean"),
            rf_rmse=("rf_residual", lambda r: float(np.sqrt((r ** 2).mean()))),
            seen_in_training=("seen_in_training_geohash5", "first"),
        )
        .sort_values("rf_mbe")
    )
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["C2" if seen else "C0" for seen in per_station["seen_in_training"]]
    ax.bar(range(len(per_station)), per_station["rf_mbe"], color=colors, alpha=0.75)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(per_station)))
    ax.set_xticklabels(per_station.index, rotation=90, fontsize=7)
    ax.set_ylabel("RF MBE (kWh/m²/day)")
    ax.set_title(
        "Per-station mean bias error vs CALIBRATED Katongole climatology  "
        "(green = geohash overlap with training)"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plt.show()
    print("Worst over-prediction:")
    print(per_station.tail(5).round(3).to_string())
    print("\nWorst under-prediction:")
    print(per_station.head(5).round(3).to_string())


def plot_figure_2_grid(
    comparison_calibrated: pd.DataFrame,
    *,
    stations: Iterable[str] | None = None,
    validation_label: str = "2017-2022",
    n_panels: int = 16,
) -> list[str]:
    """Paper-style 4 × 4 monthly GHI grid for 16 stations.

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


def plot_feature_importances_top_n(
    bundle: "TrainedBundle",
    feature_columns: Iterable[str],
    *,
    top_n: int = 12,
) -> pd.Series:
    """Bar chart of the trained RF's top-N feature importances. Returns them."""
    importances = (
        bundle.regressor.feature_importances(tuple(feature_columns))
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    importances.head(top_n).plot.bar(ax=ax)
    ax.set_ylabel("Feature importance")
    ax.set_title(f"Top {top_n} RF feature importances (paper Fig. 3 equivalent)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    plt.show()
    return importances.head(top_n)


# ---------------------------------------------------------------------------
# Scoring tables.
# ---------------------------------------------------------------------------


def _score_row(yt: pd.Series, yp: pd.Series, *, split: str, model: str) -> dict:
    """One row of the Table-IV style score sheet."""
    from susse.metrics import (
        index_of_agreement, mean_bias_error, normalised_mae, normalised_rmse,
    )
    from susse.training import score_predictions

    scores = score_predictions(yt, yp)
    return {
        "split": split,
        "model": model,
        "n": scores.n_rows,
        "RMSE": round(scores.rmse, 3),
        "nRMSE_%": round(normalised_rmse(yt, yp), 2),
        "MAE": round(scores.mae, 3),
        "nMAE_%": round(normalised_mae(yt, yp), 2),
        "MBE": round(mean_bias_error(yt, yp), 3),
        "R²": round(scores.r2, 3),
        "IOA": round(index_of_agreement(yt, yp), 3),
    }


def score_table_iv(
    comparison: pd.DataFrame, *,
    label: str | None = None,
    obs_col: str = "monthly_obs",
    rf_label: str = "RF",
) -> pd.DataFrame:
    """Cross-station mean of paper-Table-IV style metrics.

    Returns a DataFrame with rows for RF, NASA CERES and CAMS evaluated
    against ``obs_col``. Pass ``obs_col="monthly_obs_calibrated"`` for the
    cross-network-calibrated comparison.
    """
    if label is None:
        label = f"{comparison['location'].nunique()} stations"
    series: Mapping[str, pd.Series] = {
        rf_label:     comparison["monthly_pred"],
        "NASA CERES": comparison["sat_ghi_nasa_kwh_m2_day"],
        "CAMS":       comparison["sat_ghi_cams_kwh_m2_day"],
    }
    rows = [
        _score_row(comparison[obs_col], yp, split=label, model=name)
        for name, yp in series.items()
    ]
    return pd.DataFrame(rows)
