"""Generic evaluation plots.

One function per figure, no shared state. Inputs are plain dataframes
/ series / mappings — no library-specific value objects — so these
plots are equally usable in inspection notebooks, paper notebooks,
and ad-hoc sessions. Paper-specific layouts (Figure 2 grid, calibration
diagnostic) live with the paper, not here.

All functions render the figure with :func:`matplotlib.pyplot.show`
and return ``None`` unless explicitly noted; the figure is shown
inline in a notebook and not held on. Callers wanting to embed or
save the figure should rebuild it from the same helper they wrote
into the notebook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import MeanBiasError, Metric

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ..training import TrainedBundle


def plot_target_distribution_per_station(
    df: pd.DataFrame,
    *,
    target_column: str,
    location_column: str = "location",
    value_label: str = "y",
) -> None:
    """Boxplot + per-station KDE of the target value."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    df.boxplot(
        column=target_column,
        by=location_column,
        rot=90,
        ax=axes[0],
        grid=False,
    )
    axes[0].set_title(f"{value_label} per station")
    axes[0].set_ylabel(value_label)
    axes[0].set_xlabel("")
    plt.suptitle("")

    ax = axes[1]
    locs = sorted(df[location_column].unique())
    for loc in locs:
        sub = df.loc[df[location_column] == loc, target_column]
        if len(sub) > 1:
            sub.plot.kde(ax=ax, alpha=0.5, lw=0.7)
    ax.set_xlabel(value_label)
    ax.set_ylabel("density")
    ax.set_title(f"KDE per station ({len(locs)} stations)")
    fig.tight_layout()
    plt.show()


def plot_predictor_vs_observed_scatter(
    df: pd.DataFrame,
    *,
    observed_column: str,
    predictor_columns: Mapping[str, str],
    value_label: str = "kWh/m²/day",
    limit: tuple[float, float] = (0, 10),
) -> None:
    """Scatter each predictor column against the observed column.

    Args:
        df: Frame holding the observed + predictor columns.
        observed_column: Name of the ground-truth column.
        predictor_columns: Mapping ``{display_label: column_name}``.
            One subplot per entry; subplots share x and y axes.
        value_label: Axis-label unit.
        limit: ``(low, high)`` for both axes — the identity line is
            drawn over this range.
    """
    y_obs = df[observed_column]
    fig, axes = plt.subplots(
        1,
        len(predictor_columns),
        figsize=(6 * len(predictor_columns), 5.5),
        sharex=True,
        sharey=True,
    )
    if len(predictor_columns) == 1:
        axes = [axes]
    for ax, (label, column) in zip(axes, predictor_columns.items()):
        ax.scatter(y_obs, df[column], s=3, alpha=0.2, color="C0")
        ax.plot(limit, limit, "k--", lw=1)
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.set_xlabel(f"Observed ({value_label})")
        ax.set_ylabel(f"{label} ({value_label})")
        err = df[column] - y_obs
        ax.set_title(
            f"{label} vs observed  "
            f"(MBE={err.mean():.2f}, RMSE={np.sqrt((err ** 2).mean()):.2f})"
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_feature_correlation_heatmap(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    target_column: str,
    top_n: int = 10,
) -> None:
    """Pairwise feature × target correlation heatmap + top-N table."""
    cols = list(feature_columns) + [target_column]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    plt.show()

    target_corr = (
        corr[target_column].drop(target_column).abs().sort_values(ascending=False)
    )
    print(f"Top {top_n} features by |correlation| with {target_column}:")
    print(target_corr.head(top_n).round(3).to_string())


def plot_pca_by_station(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    location_column: str = "location",
) -> None:
    """PCA(2) of standardised features, points coloured by station."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = df[list(feature_columns)].to_numpy()
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2).fit(scaled)
    pcs = pca.transform(scaled)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    loc_arr = df[location_column].to_numpy()
    unique_locs = sorted(set(loc_arr))
    cmap = plt.get_cmap("tab20", max(20, len(unique_locs)))
    for i, loc in enumerate(unique_locs):
        mask = loc_arr == loc
        ax.scatter(
            pcs[mask, 0],
            pcs[mask, 1],
            s=4,
            alpha=0.5,
            color=cmap(i % cmap.N),
            label=loc,
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} var)")
    ax.set_title("PCA of features — coloured by station")
    ax.legend(fontsize=6, ncol=3, loc="best", framealpha=0.7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_training_fit_scatter(
    df: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
    value_label: str = "kWh/m²/day",
) -> None:
    """In-sample predicted-vs-observed scatter; tight diagonal = healthy training."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        df[observed_column],
        df[predicted_column],
        s=2,
        alpha=0.25,
        color="C3",
    )
    hi = max(df[observed_column].max(), df[predicted_column].max()) * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"Observed ({value_label})")
    ax.set_ylabel(f"Predicted ({value_label})")
    ax.set_title("Predicted vs observed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    err = df[predicted_column] - df[observed_column]
    print(
        f"RMSE = {np.sqrt((err ** 2).mean()):.3f}  "
        f"MAE = {np.abs(err).mean():.3f}  "
        f"MBE = {err.mean():.3f}"
    )


def plot_training_fit_timeseries(
    df: pd.DataFrame,
    *,
    observed_column: str | None = None,
    prediction_series: Mapping[str, str],
    reference_series: Mapping[str, str] | None = None,
    location_column: str = "location",
    date_column: str = "date",
    n_stations: int = 6,
    value_label: str = "kWh/m²/day",
) -> None:
    """Weekly-resampled time series of predictions + named comparison series.

    Picks the ``n_stations`` stations with the most rows and restricts
    each station's panel to its most recent calendar year for
    legibility.

    Args:
        df: Frame holding the per-row date / location columns and every
            named series.
        observed_column: Optional ground-truth column. When provided,
            rendered prominently (blue line + circle markers); when
            ``None`` (the inference use case), the observed line is
            omitted entirely and the legend is built from
            predictions + references only.
        prediction_series: Mapping ``{display_label: column_name}`` of
            model outputs to render with the same prominence as the
            observed series (line + circle markers).
        reference_series: Optional mapping ``{display_label: column_name}``
            of context series (e.g. raw satellite estimates) rendered
            as thin, low-alpha background lines without markers. The
            visual contrast surfaces which series the model produced
            vs. which were inputs / external references.
        location_column: Station identifier column.
        date_column: Row date column.
        n_stations: Number of station panels to draw.
        value_label: Y-axis label unit.
    """
    plot_stations = (
        df.groupby(location_column)
        .size()
        .sort_values(ascending=False)
        .head(n_stations)
        .index.tolist()
    )
    actual_n = len(plot_stations)
    if actual_n == 0:
        raise ValueError(
            "plot_training_fit_timeseries: no stations found in "
            f"column {location_column!r}. Check the input DataFrame."
        )
    # Tight subplot grid that doesn't leave empty cells for small n.
    if actual_n == 1:
        n_cols = 1
    elif actual_n <= 4:
        n_cols = 2
    else:
        n_cols = 2
    n_rows = (actual_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 3 * n_rows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()
    prediction_palette = ["C3", "C4", "C5"]
    reference_palette = ["C2", "C1", "C6"]
    refs = reference_series or {}
    series_cols = list(prediction_series.values()) + list(refs.values())
    if observed_column is not None:
        series_cols = [observed_column] + series_cols
    for ax, loc in zip(flat_axes, plot_stations):
        sub = df[df[location_column] == loc].sort_values(date_column).copy()
        sub[date_column] = pd.to_datetime(sub[date_column])
        last_year = sub[date_column].dt.year.max()
        sub = sub[sub[date_column].dt.year == last_year]
        weekly = sub.set_index(date_column)[series_cols].resample("W").mean()
        if observed_column is not None:
            ax.plot(
                weekly.index,
                weekly[observed_column],
                "-o",
                ms=3,
                label="Observed",
                color="C0",
            )
        for color, (label, col) in zip(prediction_palette, prediction_series.items()):
            ax.plot(
                weekly.index,
                weekly[col],
                "-o",
                ms=3,
                label=label,
                color=color,
            )
        for color, (label, col) in zip(reference_palette, refs.items()):
            ax.plot(
                weekly.index,
                weekly[col],
                "-",
                lw=0.7,
                alpha=0.4,
                label=label,
                color=color,
            )
        ax.set_title(f"{loc} ({last_year}, n={len(sub):,})", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in flat_axes[actual_n:]:
        ax.set_visible(False)
    flat_axes[0].legend(fontsize=8, loc="lower left")
    fig.supylabel(value_label)
    fig.tight_layout()
    plt.show()


def plot_predicted_vs_observed_panels(
    df: pd.DataFrame,
    *,
    observed_column: str,
    predictions: Mapping[str, str],
    value_label: str = "kWh/m²/day",
) -> None:
    """One scatter panel per named prediction column against the observed column.

    Args:
        df: Frame holding the observed + prediction columns. NaN-bearing
            rows are dropped per panel.
        observed_column: Name of the ground-truth column.
        predictions: Mapping ``{display_label: column_name}``.
        value_label: Axis-label unit.
    """
    obs = df[observed_column]
    cols = [observed_column] + list(predictions.values())
    lim = (
        float(np.floor(df[cols].min().min())),
        float(np.ceil(df[cols].max().max())),
    )
    palette = ["C3", "C2", "C1", "C4", "C5"]
    fig, axes = plt.subplots(
        1,
        len(predictions),
        figsize=(5.5 * len(predictions), 5.5),
        sharex=True,
        sharey=True,
    )
    if len(predictions) == 1:
        axes = [axes]
    for ax, color, (label, col) in zip(axes, palette, predictions.items()):
        ax.scatter(obs, df[col], s=14, alpha=0.55, color=color)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"Observed ({value_label})")
        ax.set_ylabel(f"{label} ({value_label})")
        err = (df[col] - obs).dropna()
        ax.set_title(
            f"{label}\nMBE={err.mean():.2f}  " f"RMSE={np.sqrt((err ** 2).mean()):.2f}"
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_per_station_metric(
    df: pd.DataFrame,
    *,
    observed_column: str,
    predicted_column: str,
    location_column: str = "location",
    metric: Metric = MeanBiasError(),
    highlight_column: str | None = None,
    value_label: str = "kWh/m²/day",
) -> None:
    """Per-station bar chart of a chosen metric; optional categorical colour.

    Args:
        df: Frame holding the observed + predicted + location columns.
        observed_column: Name of the ground-truth column.
        predicted_column: Name of the prediction column.
        location_column: Station identifier column.
        metric: Which :class:`Metric` to compute per station. Default
            :class:`MeanBiasError`.
        highlight_column: Optional boolean column. When given, bars
            for rows whose ``highlight_column`` is True are coloured
            differently (typically to mark training overlap).
        value_label: Y-axis label unit.
    """
    rows: list[dict] = []
    for loc, sub in df.groupby(location_column):
        score = metric.compute(sub[observed_column], sub[predicted_column])
        row: dict = {location_column: loc, metric.name: score}
        if highlight_column is not None:
            row["_highlight"] = bool(sub[highlight_column].iloc[0])
        rows.append(row)
    per_station = pd.DataFrame(rows).set_index(location_column).sort_values(metric.name)
    fig, ax = plt.subplots(figsize=(15, 6))
    if highlight_column is None:
        ax.bar(
            range(len(per_station)), per_station[metric.name], color="C0", alpha=0.75
        )
    else:
        colors = ["C2" if h else "C0" for h in per_station["_highlight"]]
        ax.bar(
            range(len(per_station)), per_station[metric.name], color=colors, alpha=0.75
        )
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(per_station)))
    ax.set_xticklabels(per_station.index, rotation=90, fontsize=7)
    ax.set_ylabel(f"{metric.name} ({value_label})")
    title = f"Per-station {metric.name}"
    if highlight_column is not None:
        title += f"  (green = {highlight_column} True)"
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plt.show()
    print(f"Worst (highest {metric.name}):")
    print(per_station.tail(5).round(3).to_string())
    print(f"\nBest (lowest {metric.name}):")
    print(per_station.head(5).round(3).to_string())


def plot_covariate_shift_kde(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    features: Iterable[str],
    train_label: str = "Training",
    val_label: str = "Inference",
) -> None:
    """Overlaid histograms of training-set vs inference-set feature distributions.

    Skips features missing from either frame so callers can pass a
    constant candidate list without guarding each name.
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
        ax.hist(
            t,
            bins=40,
            range=(lo, hi),
            alpha=0.5,
            density=True,
            label=f"{train_label} (n={len(t):,})",
        )
        ax.hist(
            v,
            bins=40,
            range=(lo, hi),
            alpha=0.5,
            density=True,
            label=f"{val_label} (n={len(v):,})",
        )
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)
    for ax in np.array(axes).flat[len(keep) :]:
        ax.axis("off")
    fig.suptitle(f"Feature distributions: {train_label} vs {val_label}")
    fig.tight_layout()
    plt.show()


def plot_feature_importances_top_n(
    bundle: "TrainedBundle",
    *,
    feature_columns: Iterable[str],
    top_n: int = 12,
) -> pd.Series:
    """Bar chart of the trained model's top-N feature importances.

    Args:
        bundle: The trained-model bundle.
        feature_columns: Names of the model's input features, in the
            order they were fed to ``fit``.
        top_n: Number of importances to show.

    Returns:
        The top-N importances as a :class:`pandas.Series` indexed by
        feature name, sorted descending.

    Raises:
        TypeError: When ``bundle.regressor`` doesn't expose a
            ``feature_importances`` method (e.g. linear models).
            A silent fallback (returning zeros, dropping the call)
            would hide that the wrong model kind was loaded.
    """
    regressor = bundle.regressor
    importances_fn = getattr(regressor, "feature_importances", None)
    if importances_fn is None:
        raise TypeError(
            f"plot_feature_importances_top_n requires a regressor with "
            f"a `feature_importances` method; got "
            f"{type(regressor).__name__}, which does not expose one. "
            f"Tree-based models (RandomForest, GradientBoosting) "
            f"expose this; linear models do not."
        )
    importances = importances_fn(tuple(feature_columns)).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    importances.head(top_n).plot.bar(ax=ax)
    ax.set_ylabel("Feature importance")
    ax.set_title(f"Top {top_n} feature importances")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    plt.show()
    return importances.head(top_n)


__all__ = [
    "plot_covariate_shift_kde",
    "plot_feature_correlation_heatmap",
    "plot_feature_importances_top_n",
    "plot_pca_by_station",
    "plot_per_station_metric",
    "plot_predicted_vs_observed_panels",
    "plot_predictor_vs_observed_scatter",
    "plot_target_distribution_per_station",
    "plot_training_fit_scatter",
    "plot_training_fit_timeseries",
]
