# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract test: the new comparison/evaluation figure functions must never
mutate `matplotlib.rcParams` — styling stays with the caller.

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

from spotforecast2.plots.comparison import (
    plot_critical_difference,
    plot_rank_stability,
)
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


def test_no_rcparams_mutation_across_all_five_functions():
    before = dict(matplotlib.rcParams)

    ranks = pd.DataFrame({"mae": [1, 2, 3], "rmse": [1, 3, 2]}, index=["a", "b", "c"])
    plot_rank_stability(ranks, highlight={"a": "#2a78d6"})

    positions = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    entries = list(positions.index)
    sig_matrix = pd.DataFrame(1.0, index=entries, columns=entries)
    sig_matrix.loc["a", "c"] = 0.01
    sig_matrix.loc["c", "a"] = 0.01
    plot_critical_difference(positions, sig_matrix)

    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    rng = np.random.default_rng(0)
    series = {"a": pd.Series(rng.standard_normal(10), index=idx)}
    plot_metric_timeline(series)

    profile = pd.DataFrame({"a": rng.standard_normal(24)}, index=range(24))
    plot_error_profile(profile)

    hour_idx = pd.date_range("2024-01-15", periods=24, freq="h", tz="UTC")
    actual = pd.Series(rng.standard_normal(24), index=hour_idx)
    forecasts = {"a": actual + rng.standard_normal(24) * 0.1}
    plot_forecast_overlay(forecasts, actual=actual)

    after = dict(matplotlib.rcParams)
    assert before == after
