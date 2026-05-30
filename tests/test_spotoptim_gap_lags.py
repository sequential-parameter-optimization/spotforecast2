# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests: the SpotOptim lag search supports non-contiguous ("gap") lags.

Motivated by an observation in the ddmo lecture (chapter 14): a warm-started run
seeded with the sparse PACF set ``[1, 2, 3, 15, 24, 25, 26, 169]`` reported
``Best lags: [1 2 ... 24]`` (gapless), which looked as if the optimizer could
only ever return a contiguous ``1..max`` range.

These tests pin the actual contract and show the gaplessness is a *selection*
outcome, not a structural limitation:

* ``initialize_lags`` expands an *integer* candidate to ``1..max`` but preserves
  a *list* candidate verbatim (only sorting it). So a winning scalar candidate
  ``"24"`` is reported as ``[1..24]`` while a list candidate keeps its gaps.
* A sparse candidate (e.g. ``[1, 2, 168]``, including the weekly lag) survives
  end-to-end -- through the categorical search, the results frame, and the
  refitted forecaster -- and can be returned verbatim.
* Warm-starting with a sparse weekly-lag seed genuinely evaluates that seed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing.forecaster_config import initialize_lags
from spotforecast2_safe.splitter import TimeSeriesFold
from spotforecast2.model_selection import (
    build_warm_start_x0,
    spotoptim_search_forecaster,
)


def _series(n: int = 400, seed: int = 0) -> pd.Series:
    """A stationary series with daily (24 h) and weekly (168 h) structure."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = (
        np.sin(2 * np.pi * t / 24)
        + 0.5 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.1, n)
    )
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.Series(y, index=idx, name="load")


def test_initialize_lags_int_expands_but_list_preserves_gaps():
    """The single transform that decides contiguity: int -> 1..max, list -> as-is."""
    lags_int, _, max_int = initialize_lags("ForecasterRecursive", 24)
    assert list(lags_int) == list(range(1, 25))  # scalar -> contiguous
    assert max_int == 24

    lags_gap, _, max_gap = initialize_lags("ForecasterRecursive", [1, 2, 168])
    assert list(lags_gap) == [1, 2, 168]  # list -> gaps preserved (NOT 1..168)
    assert max_gap == 168

    # Unsorted list is sorted but never densified.
    lags_uns, _, _ = initialize_lags("ForecasterRecursive", [168, 1, 24, 2])
    assert list(lags_uns) == [1, 2, 24, 168]


def test_spotoptim_preserves_gap_and_expands_scalar_candidates():
    """Both candidate kinds decode correctly regardless of which one wins.

    A gap candidate ``[1, 2, 168]`` must appear verbatim in the results frame,
    and a scalar candidate ``"24"`` must appear expanded to ``1..24``.
    """
    y = _series()
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
    cv = TimeSeriesFold(steps=24, initial_train_size=300, refit=False)
    search_space = {"lags": ["[1, 2, 168]", "24"], "alpha": (0.1, 1.0)}

    results, _ = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=6,
        n_initial=3,
        random_state=0,
        return_best=True,
        verbose=False,
        show_progress=False,
    )

    evaluated = {tuple(np.asarray(lag).tolist()) for lag in results["lags"]}
    assert (1, 2, 168) in evaluated  # gap candidate preserved verbatim
    assert tuple(range(1, 25)) in evaluated  # scalar "24" expanded to 1..24


def test_spotoptim_can_return_a_gap_lag_set_verbatim():
    """When the only sensible winner is sparse, the best lags keep their gaps."""
    y = _series()
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
    cv = TimeSeriesFold(steps=24, initial_train_size=300, refit=False)
    # Single sparse candidate -> it must be the winner, returned untouched.
    search_space = {"lags": ["[1, 2, 168]"], "alpha": (0.1, 1.0)}

    results, _ = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=3,
        n_initial=2,
        random_state=0,
        return_best=True,
        verbose=False,
        show_progress=False,
    )

    best = list(np.asarray(results.loc[0, "lags"]))
    assert best == [1, 2, 168]  # gap preserved, NOT densified to 1..168
    # The refitted forecaster actually uses the sparse lags.
    assert list(np.asarray(forecaster.lags)) == [1, 2, 168]


def test_warm_start_seeds_and_evaluates_a_weekly_gap_lag():
    """A warm-start seed carrying the weekly lag is genuinely evaluated."""
    y = _series()
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
    key_lags = [1, 2, 24, 168]
    seed = str(key_lags)
    search_space = {"alpha": (0.1, 10.0), "lags": [seed, "24"]}

    x0 = build_warm_start_x0(search_space, forecaster, key_lags)
    assert x0 is not None

    cv = TimeSeriesFold(steps=24, initial_train_size=300, refit=False)
    results, _ = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=4,
        n_initial=3,
        random_state=1,
        return_best=True,
        verbose=False,
        show_progress=False,
        kwargs_spotoptim={"x0": x0},
    )

    evaluated = [list(np.asarray(lag)) for lag in results["lags"]]
    assert key_lags in evaluated  # the seeded gap-lag set was really tried
