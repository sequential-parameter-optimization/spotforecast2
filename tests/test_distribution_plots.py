# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Plotly distribution helpers."""

import numpy as np
import pandas as pd

from spotforecast2.plots.distribution import plot_distribution, plot_distributions


def test_plot_distribution_returns_figure_with_histogram_when_few_uniques():
    """Series with few unique values renders as histogram."""
    y = pd.Series([0, 1, 0, 1, 2, 2, 1, 0] * 10)
    fig = plot_distribution(y)
    assert fig.data[0].type == "histogram"


def test_plot_distribution_returns_figure_with_kde_when_many_uniques():
    """Series with many unique values renders as KDE (scatter)."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.standard_normal(500))
    fig = plot_distribution(y)
    assert fig.data[0].type == "scatter"


def test_plot_distribution_handles_empty_series():
    """All-NaN series produces an empty figure without crashing."""
    y = pd.Series([np.nan, np.nan, np.nan])
    fig = plot_distribution(y)
    assert len(fig.data) == 0


def test_plot_distributions_produces_one_trace_per_column():
    """Multi-column DataFrame produces one trace per column."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.standard_normal(200),
            "b": rng.integers(0, 4, 200),
            "c": rng.standard_normal(200),
        }
    )
    fig = plot_distributions(df)
    assert len(fig.data) == 3
