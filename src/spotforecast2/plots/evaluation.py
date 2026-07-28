# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Campaign-level evaluation plots for forecasting entries over time.

Generalises the daily-metric timeline, hour-of-day error profile, and
day-ahead forecast overlay of a manuscript's results section into a
reusable, stateless API. All functions return a `matplotlib.figure.Figure`;
the caller is responsible for saving and closing it. None of them call
`plt.show()` nor mutate `matplotlib.rcParams` — styling and figure
lifecycle stay with the caller (set `matplotlib.use("Agg")` before
importing `pyplot` in headless environments).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _x_values(series: pd.Series, x: str):
    """Return the x coordinates for `series` given `x` ("index" or "hour")."""
    if x == "hour":
        return series.index.hour
    return series.index


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def plot_metric_timeline(
    series: Mapping[str, pd.Series],
    *,
    band: pd.DataFrame | None = None,
    band_label: str = "range",
    band_color: str = "0.9",
    colors: Mapping[str, str] | None = None,
    linewidth: float = 1.8,
    ylabel: str = "",
    legend_loc: str = "upper right",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.3, 2.9),
) -> Figure:
    """Daily-metric timeline for one or more entries, with an optional range band.

    Each series in `series` is plotted over its own (datetime) index after
    `dropna()`, so entries that join or leave the campaign at different
    dates are drawn only over their own scored period. When `band` is
    given, its row-wise min/max is drawn as a filled band underneath the
    lines (e.g. the range spanned by a group of reference entries).

    Args:
        series: Mapping of entry name to a metric series with a datetime
            index (e.g. daily MAE).
        band: Optional DataFrame whose row-wise min and max are filled
            between (e.g. the daily metric of a group of entries). Its
            index must be datetime-like and comparable to the `series`
            indices.
        band_label: Legend label for the band.
        band_color: Fill color for the band.
        colors: Mapping of entry name to line color. Entries missing from
            the mapping use matplotlib's default color cycle.
        linewidth: Line width for every series.
        ylabel: Y-axis label.
        legend_loc: `loc` argument forwarded to `ax.legend`.
        ax: Existing axes to draw into. When given, no new figure is
            created and the function returns `ax.figure`.
        figsize: Figure size used when `ax` is not given.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.plots.evaluation import plot_metric_timeline

        idx = pd.date_range("2024-01-01", periods=14, freq="D")
        rng = np.random.default_rng(0)
        series = {
            "forecaster": pd.Series(500 + rng.standard_normal(14) * 20, index=idx),
            "baseline": pd.Series(700 + rng.standard_normal(14) * 20, index=idx),
        }
        band = pd.DataFrame(
            {f"peer_{i}": 550 + rng.standard_normal(14) * 40 for i in range(4)},
            index=idx,
        )
        fig = plot_metric_timeline(series, band=band, ylabel="daily MAE (MW)")
        print(type(fig).__name__)
        ```
    """
    colors = colors or {}

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_lines = len(series)
    if band is not None:
        ax.fill_between(
            band.index,
            band.min(axis=1),
            band.max(axis=1),
            color=band_color,
            label=band_label,
            zorder=1,
        )
        n_lines += 1

    for name, values in series.items():
        clean = values.dropna()
        ax.plot(
            clean.index,
            clean.to_numpy(),
            color=colors.get(name),
            lw=linewidth,
            label=name,
            zorder=3,
        )

    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, fontsize=8, ncols=n_lines)
    ax.set_ylim(bottom=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if owns_figure:
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.tight_layout()

    return fig


def plot_error_profile(
    profile: pd.DataFrame,
    *,
    colors: Mapping[str, str] | None = None,
    linewidth: float = 1.8,
    marker: str = "o",
    marker_size: float = 3.0,
    zero_line: bool = True,
    xlabel: str = "",
    ylabel: str = "mean error",
    xticks: Sequence[float] | None = None,
    legend_loc: str = "lower center",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.3, 2.7),
) -> Figure:
    """Mean-error-by-group profile for one or more entries (e.g. hour of day).

    `profile` has one row per grouping key (e.g. hour of day, 0-23) and one
    column per entry; each column is drawn as its own line.

    Args:
        profile: DataFrame indexed by the grouping key, one column per
            entry.
        colors: Mapping of column name to line color. Columns missing from
            the mapping use matplotlib's default color cycle.
        linewidth: Line width for every entry.
        marker: Marker style for every entry.
        marker_size: Marker size for every entry.
        zero_line: When `True`, draw a horizontal reference line at 0.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        xticks: Explicit tick positions for the x axis. When `None`, the
            default matplotlib ticks are kept.
        legend_loc: `loc` argument forwarded to `ax.legend`.
        ax: Existing axes to draw into. When given, no new figure is
            created and the function returns `ax.figure`.
        figsize: Figure size used when `ax` is not given.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.plots.evaluation import plot_error_profile

        rng = np.random.default_rng(0)
        profile = pd.DataFrame(
            {
                "forecaster": rng.normal(0, 100, 24),
                "baseline": rng.normal(-200, 150, 24),
            },
            index=range(24),
        )
        fig = plot_error_profile(profile, xticks=range(0, 24, 3))
        print(type(fig).__name__)
        ```
    """
    colors = colors or {}

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if zero_line:
        ax.axhline(0, color="black", lw=0.8)

    for name in profile.columns:
        ax.plot(
            profile.index,
            profile[name],
            color=colors.get(name),
            lw=linewidth,
            marker=marker,
            ms=marker_size,
            label=name,
        )

    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, fontsize=8, ncols=len(profile.columns))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if owns_figure:
        fig.tight_layout()

    return fig


def plot_forecast_overlay(
    forecasts: Mapping[str, pd.Series],
    *,
    actual: pd.Series | None = None,
    actual_label: str = "actual",
    actual_color: str = "black",
    x: str = "index",
    colors: Mapping[str, str] | None = None,
    linestyles: Mapping[str, str] | None = None,
    linewidth: float = 1.8,
    xlabel: str = "",
    ylabel: str = "",
    xticks: Sequence[float] | None = None,
    legend_loc: str = "lower center",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.3, 2.9),
) -> Figure:
    """One or more forecasts overlaid against an optional actual series.

    `actual`, when given, is drawn first and slightly thicker than the
    forecast lines (`linewidth + 0.2`) so it reads as the reference trace.
    Each entry in `forecasts` is then drawn in insertion order.

    Args:
        forecasts: Mapping of entry name to forecast series.
        actual: Optional realised/ground-truth series, drawn first.
        actual_label: Legend label for `actual`.
        actual_color: Line color for `actual`.
        x: `"index"` plots every series against its own index as-is;
            `"hour"` plots against `index.hour` (e.g. to overlay several
            day-ahead forecasts for the same target day on a 0-23 axis).
        colors: Mapping of entry name to line color. Entries missing from
            the mapping use matplotlib's default color cycle.
        linestyles: Mapping of entry name to a matplotlib linestyle (e.g.
            `{"baseline": "--"}`). Entries missing from the mapping are
            drawn solid.
        linewidth: Line width for the forecast lines.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        xticks: Explicit tick positions for the x axis. When `None`, the
            default matplotlib ticks are kept.
        legend_loc: `loc` argument forwarded to `ax.legend`.
        ax: Existing axes to draw into. When given, no new figure is
            created and the function returns `ax.figure`.
        figsize: Figure size used when `ax` is not given.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.plots.evaluation import plot_forecast_overlay

        idx = pd.date_range("2024-01-15", periods=24, freq="h", tz="UTC")
        rng = np.random.default_rng(0)
        actual = pd.Series(40_000 + rng.standard_normal(24) * 500, index=idx)
        forecasts = {
            "forecaster": actual + rng.standard_normal(24) * 300,
            "baseline": actual + rng.standard_normal(24) * 800,
        }
        fig = plot_forecast_overlay(
            forecasts,
            actual=actual,
            x="hour",
            linestyles={"baseline": "--"},
            xticks=range(0, 24, 3),
        )
        print(type(fig).__name__)
        ```
    """
    colors = colors or {}
    linestyles = linestyles or {}

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_lines = 0
    if actual is not None:
        ax.plot(
            _x_values(actual, x),
            actual.to_numpy(),
            color=actual_color,
            lw=linewidth + 0.2,
            label=actual_label,
        )
        n_lines += 1

    for name, values in forecasts.items():
        ax.plot(
            _x_values(values, x),
            values.to_numpy(),
            color=colors.get(name),
            lw=linewidth,
            ls=linestyles.get(name, "-"),
            label=name,
        )
        n_lines += 1

    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, fontsize=8, ncols=n_lines)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if owns_figure:
        fig.tight_layout()

    return fig
