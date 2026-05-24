# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Plotly periodogram chart."""

import numpy as np
import pandas as pd

from spotforecast2_safe.stats.spectral import compute_periodogram
from spotforecast2.plots.spectral import plot_periodogram


def _sine_series(period: int = 24, n: int = 512) -> pd.Series:
    t = np.arange(n)
    return pd.Series(np.sin(2 * np.pi * t / period))


def test_plot_periodogram_returns_figure_with_log_xaxis():
    """The x-axis is rendered on a log scale."""
    result = compute_periodogram(_sine_series())
    fig = plot_periodogram(result)
    assert fig.layout.xaxis.type == "log"


def test_plot_periodogram_named_period_ticks_present():
    """Default tick labels include the documented named periods."""
    result = compute_periodogram(_sine_series())
    fig = plot_periodogram(result)
    tick_text = list(fig.layout.xaxis.ticktext)
    assert any("day" in t for t in tick_text)
    assert any("week" in t for t in tick_text)


def test_plot_periodogram_accepts_series():
    """A bare ``pd.Series`` indexed by frequency also works."""
    result = compute_periodogram(_sine_series())
    series = result.spectrum["power"]
    fig = plot_periodogram(series)
    assert len(fig.data) == 1
