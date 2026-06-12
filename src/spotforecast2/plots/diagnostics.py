# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Operational diagnostic plots for energy-load forecasting pipelines.

Ports five matplotlib helpers from the chapter-14 team4 operational script
(`bart26k-lecture/scripts/team4_4zones_submit.py`) into a reusable, stateless
API.  All functions return a `matplotlib.figure.Figure`; the caller is
responsible for saving and closing it.  No `plt.show()` is called and the
backend is never changed here (set `matplotlib.use("Agg")` before importing
`pyplot` in headless environments).
"""

from __future__ import annotations

import logging
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Family colour map — identical to the chapter-14 script so PNG diagnostics
# look the same whether run from the script or via this module.
# ---------------------------------------------------------------------------
_FAMILY_COLOR: dict[str, str] = {
    "lag": "#B22222",
    "weather/other": "#E07B00",
    "weather_window": "#F0A33C",
    "cyclical/RBF": "#1F4E79",
    "holiday": "#2E8B57",
    "polynomial": "#7B5EA7",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def feature_family(name: str) -> str:
    """Map a feature name to its diagnostic family label.

    This is the public, importable version of the `family_of` helper used
    inside the chapter-14 team4 operational script.  The mapping is
    intentionally coarse — it covers the feature names that `ConfigEntsoe`
    and `ForecasterRecursive` generate and is used exclusively for colour
    grouping in `plot_feature_importance_by_family`.

    Args:
        name: Feature name as returned by `LGBMRegressor.feature_name_`.

    Returns:
        One of: `"holiday"`, `"polynomial"`, `"weather_window"`,
        `"cyclical/RBF"`, `"lag"`, `"weather/other"`.

    Examples:
        ```{python}
        from spotforecast2.plots.diagnostics import feature_family

        print(feature_family("holiday_DE"))
        print(feature_family("brueckentag_NW"))
        print(feature_family("poly_hour_2"))
        print(feature_family("window_mean_72"))
        print(feature_family("sin_hour"))
        print(feature_family("lag_1"))
        print(feature_family("wind_speed_10m"))
        ```
    """
    c = name.lower()
    if "holiday" in c or "brueckentag" in c:
        return "holiday"
    if "poly" in c:
        return "polynomial"
    if "window" in c:
        return "weather_window"
    if any(k in c for k in ("sin", "cos", "rbf")):
        return "cyclical/RBF"
    if c.startswith("lag"):
        return "lag"
    return "weather/other"


def plot_acf_with_lags(
    acf: pd.DataFrame,
    key_lags: Sequence[int],
    conf: float,
) -> Figure:
    """Bar chart of autocorrelation values with annotated key lags.

    Ports `_plot_acf` from the chapter-14 team4 script.  The `acf` DataFrame
    is the output of
    `spotforecast2.stats.autocorrelation.calculate_lag_autocorrelation` and
    must contain at minimum the columns `"lag"` and `"autocorrelation"`.

    Confidence-band lines at `+conf` / `-conf` are drawn as dashed grey
    horizontal lines.  Each lag in `key_lags` that is present in `acf["lag"]`
    gets an orange arrow annotation.  Lags not found in the frame are silently
    skipped.

    Args:
        acf: DataFrame with columns `"lag"` (int) and `"autocorrelation"`
            (float), as returned by `calculate_lag_autocorrelation`.
        key_lags: Sequence of lag values to annotate (e.g. the PACF-selected
            lags from the pipeline).
        conf: Half-width of the confidence band, typically
            `1.96 / sqrt(n_obs)`.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.plots.diagnostics import plot_acf_with_lags

        rng = np.random.default_rng(0)
        n = 100
        acf = pd.DataFrame({"lag": range(n), "autocorrelation": rng.uniform(-0.3, 0.3, n)})
        fig = plot_acf_with_lags(acf, key_lags=[1, 24, 48], conf=0.1)
        print(type(fig).__name__)
        ```
    """
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.bar(acf["lag"], acf["autocorrelation"], width=0.9, color="#1F4E79")
    ax.axhline(conf, color="#999999", lw=0.8, ls="--")
    ax.axhline(-conf, color="#999999", lw=0.8, ls="--")
    for lag in key_lags:
        row = acf[acf["lag"] == lag]
        if row.empty:
            continue
        val = float(row["autocorrelation"].iloc[0])
        ax.annotate(
            f"lag {lag}",
            xy=(lag, val),
            xytext=(lag, val + 0.08),
            ha="center",
            fontsize=8,
            color="#E07B00",
            arrowprops=dict(arrowstyle="->", color="#E07B00", lw=0.8),
        )
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("ACF of Actual Load (annotated: data-selected key_lags)")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_feature_importance_by_family(
    names: Sequence[str],
    importances: Sequence[float],
    *,
    top_n: int = 20,
) -> Figure:
    """Horizontal bar chart of the top-N feature importances, coloured by family.

    Ports `_plot_importance` from the chapter-14 team4 script.  Feature
    families are determined by `feature_family`; the colour palette is the
    same as in the script so diagnostics look identical.

    Args:
        names: Feature names (e.g. `fc.estimator.feature_name_`).
        importances: Corresponding importance scores (e.g.
            `fc.estimator.feature_importances_`).
        top_n: Number of top features to display.  Defaults to 20.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        from spotforecast2.plots.diagnostics import plot_feature_importance_by_family

        names = ["lag_1", "lag_24", "holiday_DE", "wind_speed_10m",
                 "poly_hour_2", "window_mean_72", "sin_hour"]
        scores = [100, 80, 60, 55, 40, 35, 20]
        fig = plot_feature_importance_by_family(names, scores, top_n=5)
        print(type(fig).__name__)
        ```
    """
    ranking = sorted(
        zip(names, importances), key=lambda kv: kv[1], reverse=True
    )[:top_n]
    labels = [n for n, _ in ranking][::-1]
    values = [v for _, v in ranking][::-1]
    colors = [_FAMILY_COLOR.get(feature_family(n), "#888888") for n in labels]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("split count (feature importance)")
    ax.set_title(
        f"Top-{top_n} feature importances (coloured by family; lags in red)"
    )
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in _FAMILY_COLOR.values()
    ]
    ax.legend(handles, _FAMILY_COLOR.keys(), fontsize=7, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shap_summary(
    estimator: object,
    X: pd.DataFrame,
    *,
    max_samples: int = 2000,
) -> Figure:
    """SHAP bar-summary plot for a tree-based estimator.

    Ports `_plot_shap` from the chapter-14 team4 script.  `X` is subsampled
    to approximately `max_samples` rows (uniform stride `len(X) // max_samples`;
    lengths just above `max_samples` are passed in full) before computing SHAP
    values so the call stays fast even for large training matrices.

    The function uses `shap.TreeExplainer` and
    `shap.summary_plot(plot_type="bar", show=False)`, then captures the
    current matplotlib figure via `plt.gcf()`.  Because the figure is harvested
    from the global pyplot state this function is **not thread-safe**.  Callers
    must close the returned figure (e.g. `plt.close(fig)`) before performing
    other pyplot work.

    Args:
        estimator: A fitted tree-based estimator supported by
            `shap.TreeExplainer` (e.g. `LGBMRegressor`).
        X: Feature matrix; typically the design matrix returned by
            `ForecasterRecursive.create_train_X_y`.
        max_samples: Row budget passed to the SHAP explainer.  Defaults to 2000.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        #| eval: false
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMRegressor
        from spotforecast2.plots.diagnostics import plot_shap_summary

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 5)),
                         columns=[f"f{i}" for i in range(5)])
        y = X["f0"] + rng.standard_normal(200) * 0.1
        est = LGBMRegressor(n_estimators=20, verbose=-1)
        est.fit(X, y)
        fig = plot_shap_summary(est, X, max_samples=100)
        print(type(fig).__name__)
        ```
    """
    import shap

    step = max(1, len(X) // max_samples)
    X_sample = X.iloc[::step]
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(X_sample)
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_forecast_vs_reference(
    forecast: pd.Series,
    reference: pd.Series,
    *,
    forecast_label: str = "forecast",
    reference_label: str = "reference",
    unit_scale: float = 1e-3,
    unit: str = "GW",
) -> Figure:
    """Line plot comparing a forecast against an optional reference series.

    Ports `_plot_vs_entsoe` from the chapter-14 team4 script into a general,
    label-parametrised form.  The reference is reindexed to `forecast.index`;
    only the overlapping (non-NaN) timestamps are plotted.  If there is no
    overlap the reference line is omitted and an INFO message is logged — the
    function still returns a valid single-line figure rather than raising.

    Both series are scaled by `unit_scale` before plotting (default `1e-3`
    converts MW to GW).

    The overlap MAD (mean absolute deviation between `forecast` and
    `reference` over shared timestamps) is logged at INFO level when an
    overlap exists.  This mirrors the original script's behaviour and is
    useful for post-hoc sanity checks in operator logs.

    Args:
        forecast: Point forecast series with a `DatetimeIndex`.
        reference: Reference series (e.g. ENTSO-E day-ahead forecast).
            Will be reindexed to `forecast.index`; NaN rows after reindexing
            are treated as "no overlap" for that timestamp.
        forecast_label: Legend label for the forecast line.
        reference_label: Legend label for the reference line.
        unit_scale: Multiplicative scale applied to both series before
            plotting.  Defaults to `1e-3` (MW → GW).
        unit: Unit string used in the y-axis label.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.plots.diagnostics import plot_forecast_vs_reference

        idx = pd.date_range("2024-01-15", periods=24, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        forecast = pd.Series(40_000 + rng.standard_normal(24) * 1000, index=idx)
        reference = pd.Series(41_000 + rng.standard_normal(24) * 800, index=idx)

        fig = plot_forecast_vs_reference(
            forecast, reference,
            forecast_label="team_4 forecast",
            reference_label="ENTSO-E day-ahead",
        )
        print(type(fig).__name__)
        ```
    """
    ref_aligned = reference.reindex(forecast.index)
    overlap = ref_aligned.dropna().index

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.plot(
        forecast.index,
        forecast.values * unit_scale,
        marker="o",
        ms=3,
        color="#1F4E79",
        label=forecast_label,
    )
    if len(overlap) > 0:
        ax.plot(
            ref_aligned.index,
            ref_aligned.values * unit_scale,
            marker="x",
            ms=4,
            color="#E07B00",
            label=reference_label,
        )
        mad = float((forecast.loc[overlap] - ref_aligned.loc[overlap]).abs().mean())
        logger.info(
            "mean |%s - %s| over %d overlap hours: %.1f %s",
            forecast_label,
            reference_label,
            len(overlap),
            mad * unit_scale,
            unit,
        )
    else:
        logger.info(
            "%s not available for forecast period; plotting %s only.",
            reference_label,
            forecast_label,
        )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(f"Load [{unit}]")
    ax.set_title(f"{forecast_label} vs. {reference_label} — target day")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    ax.legend(fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
