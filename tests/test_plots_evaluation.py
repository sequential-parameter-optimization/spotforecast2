# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2.plots.evaluation.

Headless matplotlib backend must be set BEFORE pyplot is imported.  The
`matplotlib.use()` call at module level (before any pyplot import) satisfies
that requirement.
"""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from spotforecast2.plots.evaluation import (
    plot_error_profile,
    plot_forecast_overlay,
    plot_metric_timeline,
)

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close every matplotlib figure after each test to avoid resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_metric_timeline
# ---------------------------------------------------------------------------


def _make_timeline_series(n: int = 14, seed: int = 0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return {
        "forecaster": pd.Series(500 + rng.standard_normal(n) * 20, index=idx),
        "baseline": pd.Series(700 + rng.standard_normal(n) * 20, index=idx),
    }


def test_metric_timeline_returns_figure():
    fig = plot_metric_timeline(_make_timeline_series())
    assert isinstance(fig, Figure)


def test_metric_timeline_line_count_matches_series():
    series = _make_timeline_series()
    fig = plot_metric_timeline(series)
    ax = fig.axes[0]
    assert len(ax.lines) == len(series)


def test_metric_timeline_band_drawn_when_given():
    series = _make_timeline_series()
    idx = next(iter(series.values())).index
    rng = np.random.default_rng(1)
    band = pd.DataFrame(
        {f"peer_{i}": 550 + rng.standard_normal(len(idx)) * 40 for i in range(4)},
        index=idx,
    )
    fig = plot_metric_timeline(series, band=band)
    ax = fig.axes[0]
    assert len(ax.collections) == 1  # fill_between produces one PolyCollection


def test_metric_timeline_no_band_by_default():
    fig = plot_metric_timeline(_make_timeline_series())
    ax = fig.axes[0]
    assert len(ax.collections) == 0


def test_metric_timeline_legend_ncols_includes_band():
    series = _make_timeline_series()
    idx = next(iter(series.values())).index
    band = pd.DataFrame({"peer_0": np.zeros(len(idx))}, index=idx)
    fig = plot_metric_timeline(series, band=band)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert len(legend.get_texts()) == len(series) + 1


def test_metric_timeline_ylim_bottom_zero():
    fig = plot_metric_timeline(_make_timeline_series())
    ax = fig.axes[0]
    ymin, _ = ax.get_ylim()
    assert ymin == 0


def test_metric_timeline_ax_injection_no_new_figure():
    _fig0, ax = plt.subplots()
    n_figs_before = len(plt.get_fignums())
    fig = plot_metric_timeline(_make_timeline_series(), ax=ax)
    assert fig is ax.figure
    assert len(plt.get_fignums()) == n_figs_before


# ---------------------------------------------------------------------------
# plot_error_profile
# ---------------------------------------------------------------------------


def _make_profile():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "forecaster": rng.normal(0, 100, 24),
            "baseline": rng.normal(-200, 150, 24),
        },
        index=range(24),
    )


def test_error_profile_returns_figure():
    fig = plot_error_profile(_make_profile())
    assert isinstance(fig, Figure)


def test_error_profile_zero_line_drawn_by_default():
    fig = plot_error_profile(_make_profile())
    ax = fig.axes[0]
    # axhline(0, ...) plus one line per column
    assert len(ax.lines) == 1 + 2


def test_error_profile_zero_line_toggle_off():
    fig = plot_error_profile(_make_profile(), zero_line=False)
    ax = fig.axes[0]
    assert len(ax.lines) == 2


def test_error_profile_ax_injection_no_new_figure():
    _fig0, ax = plt.subplots()
    n_figs_before = len(plt.get_fignums())
    fig = plot_error_profile(_make_profile(), ax=ax)
    assert fig is ax.figure
    assert len(plt.get_fignums()) == n_figs_before


# ---------------------------------------------------------------------------
# plot_forecast_overlay
# ---------------------------------------------------------------------------


def _make_overlay(n: int = 24, seed: int = 0):
    idx = pd.date_range("2024-01-15", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    actual = pd.Series(40_000 + rng.standard_normal(n) * 500, index=idx)
    forecasts = {
        "forecaster": actual + rng.standard_normal(n) * 300,
        "baseline": actual + rng.standard_normal(n) * 800,
    }
    return actual, forecasts


def test_forecast_overlay_returns_figure():
    actual, forecasts = _make_overlay()
    fig = plot_forecast_overlay(forecasts, actual=actual)
    assert isinstance(fig, Figure)


def test_forecast_overlay_actual_drawn_when_given():
    actual, forecasts = _make_overlay()
    fig = plot_forecast_overlay(forecasts, actual=actual)
    ax = fig.axes[0]
    assert len(ax.lines) == len(forecasts) + 1


def test_forecast_overlay_actual_absent_by_default():
    _actual, forecasts = _make_overlay()
    fig = plot_forecast_overlay(forecasts)
    ax = fig.axes[0]
    assert len(ax.lines) == len(forecasts)


def test_forecast_overlay_x_hour_uses_hour_values():
    actual, forecasts = _make_overlay()
    fig = plot_forecast_overlay(forecasts, actual=actual, x="hour")
    ax = fig.axes[0]
    xdata = ax.lines[0].get_xdata()
    assert list(xdata) == list(actual.index.hour)


def test_forecast_overlay_x_index_uses_datetime_index():
    actual, forecasts = _make_overlay()
    fig = plot_forecast_overlay(forecasts, actual=actual, x="index")
    ax = fig.axes[0]
    xdata = ax.lines[0].get_xdata()
    # datetime index values are converted to matplotlib date floats
    assert len(xdata) == len(actual)
    assert not np.array_equal(np.asarray(xdata), actual.index.hour.to_numpy())


def test_forecast_overlay_linestyles_applied():
    actual, forecasts = _make_overlay()
    fig = plot_forecast_overlay(forecasts, actual=actual, linestyles={"baseline": "--"})
    ax = fig.axes[0]
    styles = {line.get_label(): line.get_linestyle() for line in ax.lines}
    assert styles["baseline"] == "--"
    assert styles["forecaster"] == "-"


def test_forecast_overlay_ax_injection_no_new_figure():
    actual, forecasts = _make_overlay()
    _fig0, ax = plt.subplots()
    n_figs_before = len(plt.get_fignums())
    fig = plot_forecast_overlay(forecasts, actual=actual, ax=ax)
    assert fig is ax.figure
    assert len(plt.get_fignums()) == n_figs_before


# ---------------------------------------------------------------------------
# Review-driven additions: zero-line color, x-mode validation
# ---------------------------------------------------------------------------


def test_error_profile_zero_line_color_parameterized():
    fig = plot_error_profile(_make_profile(), zero_line_color="#0b0b0b")
    zero_lines = [ln for ln in fig.axes[0].lines if ln.get_color() == "#0b0b0b"]
    assert zero_lines


def test_forecast_overlay_invalid_x_raises():
    idx = pd.date_range("2024-01-15", periods=24, freq="h", tz="UTC")
    series = pd.Series(range(24), index=idx, dtype=float)
    with pytest.raises(ValueError, match="x must be"):
        plot_forecast_overlay({"f": series}, x="Hour")
