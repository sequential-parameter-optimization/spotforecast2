# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for warm-starting the SpotOptim search with ``warm_start_lags``.

Covers the ``build_warm_start_x0`` helper (full-dim seed point, clipping,
graceful ``None`` cases) and an end-to-end check that the seeded lag
configuration becomes a real, evaluated candidate.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.splitter import TimeSeriesFold

from spotforecast2.model_selection import (
    build_warm_start_x0,
    spotoptim_search_forecaster,
)
from spotforecast2.model_selection.spotoptim_search import convert_search_space


def test_build_warm_start_x0_simple():
    """Seed point has the right length, lags index, and estimator value."""
    fc = ForecasterRecursive(estimator=Ridge(alpha=2.0), lags=3)
    ss = {"alpha": (0.1, 10.0), "lags": ["[1, 2, 24]", "24"]}
    x0 = build_warm_start_x0(ss, fc, [1, 2, 24])
    assert x0 is not None
    assert x0.shape == (2,)
    assert x0[0] == 2.0  # Ridge alpha, within (0.1, 10.0)
    assert x0[1] == 0.0  # index of "[1, 2, 24]" in the lags candidates


def test_build_warm_start_x0_clips_out_of_bounds():
    """Estimator defaults outside the search range are clipped to the bound."""
    fc = ForecasterRecursive(estimator=Ridge(alpha=100.0), lags=3)
    ss = {"alpha": (0.01, 10.0), "lags": ["[1, 2, 24]"]}
    x0 = build_warm_start_x0(ss, fc, [1, 2, 24])
    assert x0[0] == 10.0  # clipped from 100.0 to the upper bound


def test_build_warm_start_x0_midpoint_fallback():
    """A numeric dim with no matching estimator param falls back to midpoint."""
    fc = ForecasterRecursive(estimator=Ridge(), lags=3)
    ss = {"estimator__missing": (2.0, 8.0), "lags": ["[1, 2, 24]"]}
    x0 = build_warm_start_x0(ss, fc, [1, 2, 24])
    assert x0[0] == 5.0  # (2.0 + 8.0) / 2


def test_build_warm_start_x0_none_without_lags():
    """No ``lags`` factor in the search space -> no warm start."""
    fc = ForecasterRecursive(estimator=Ridge(), lags=3)
    assert build_warm_start_x0({"alpha": (0.1, 10.0)}, fc, [1, 2, 24]) is None


def test_build_warm_start_x0_none_when_seed_absent():
    """Seed must be a candidate in the lags factor, else no warm start."""
    fc = ForecasterRecursive(estimator=Ridge(), lags=3)
    ss = {"alpha": (0.1, 10.0), "lags": ["24", "48"]}
    assert build_warm_start_x0(ss, fc, [1, 2, 24]) is None


def test_warm_start_seed_is_evaluated():
    """End-to-end: the seeded lag set becomes an evaluated candidate."""
    idx = pd.date_range("2022-01-01", periods=400, freq="h")
    rng = np.random.default_rng(0)
    y = pd.Series(
        np.sin(np.arange(400) * 2 * np.pi / 24) + rng.standard_normal(400) * 0.1,
        index=idx,
        name="load",
    )
    fc = ForecasterRecursive(estimator=Ridge(), lags=24)
    key_lags = [1, 2, 24]
    seed = str(key_lags)
    ss = {"alpha": (0.01, 10.0), "lags": [seed, "12", "24"]}
    x0 = build_warm_start_x0(ss, fc, key_lags)
    assert x0 is not None

    cv = TimeSeriesFold(steps=24, initial_train_size=300, refit=False)
    results, _ = spotoptim_search_forecaster(
        forecaster=fc,
        y=y,
        cv=cv,
        search_space=ss,
        metric="mean_absolute_error",
        n_trials=4,
        n_initial=3,
        random_state=1,
        return_best=True,
        verbose=False,
        show_progress=False,
        kwargs_spotoptim={"x0": x0},
    )

    # The seeded lag set must appear among the evaluated configurations.
    evaluated = [list(np.asarray(lag)) for lag in results["lags"]]
    assert key_lags in evaluated


def test_warm_start_x0_round_trips_to_seed():
    """The x0 lags coordinate decodes back to the seeded candidate."""
    fc = ForecasterRecursive(estimator=Ridge(alpha=1.0), lags=3)
    key_lags = [1, 2, 24]
    seed = str(key_lags)
    ss = {"alpha": (0.01, 10.0), "lags": ["24", seed, "12"]}
    x0 = build_warm_start_x0(ss, fc, key_lags)
    bounds, var_type, var_name, _ = convert_search_space(ss)
    lags_idx = var_name.index("lags")
    # x0 stores the candidate index; it must point back at the seed string.
    assert bounds[lags_idx][int(x0[lags_idx])] == seed


class _MockConfig:
    warm_start_lags = [1, 2, 24]
    random_state = 1
    n_trials_spotoptim = 4
    n_initial_spotoptim = 3


class _MockTask:
    """Minimal task surface used by ``SpotOptimStrategy.prepare_forecaster``."""

    def __init__(self):
        import logging

        self.config = _MockConfig()
        self.logger = logging.getLogger("test_warm_start")
        self._show_progress = False
        self.saved = {}

    def cv_ts(self, y_train):
        return TimeSeriesFold(steps=24, initial_train_size=300, refit=False)

    def save_tuning_results(self, target, task_name, best_params, best_lags):
        self.saved = dict(best_params=best_params, best_lags=best_lags)

    def create_forecaster(self, target):
        return ForecasterRecursive(estimator=Ridge(), lags=24)


def test_strategy_injects_seed_and_forwards_x0(monkeypatch):
    """The strategy injects warm_start_lags as a candidate and forwards x0."""
    from spotforecast2.multitask.strategies import SpotOptimStrategy

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        df = pd.DataFrame(
            {
                "lags": [np.array([1, 2, 24])],
                "params": [{"alpha": 1.0}],
                "mean_absolute_error": [0.1],
            }
        )
        return df, None

    # Patch the package attribute the strategy's lazy import reads. The dotted
    # string form resolves the same sys.modules object that
    # `from spotforecast2.model_selection import spotoptim_search_forecaster`
    # binds inside prepare_forecaster.
    monkeypatch.setattr(
        "spotforecast2.model_selection.spotoptim_search_forecaster", fake_search
    )

    y = pd.Series(
        np.sin(np.arange(400) * 2 * np.pi / 24),
        index=pd.date_range("2022-01-01", periods=400, freq="h"),
        name="load",
    )
    task = _MockTask()
    strat = SpotOptimStrategy(
        search_space={"alpha": (0.01, 10.0), "lags": ["24", "12"]}
    )
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=24)

    tuned = strat.prepare_forecaster(task, "load", forecaster, y, None)

    # Seed injected as a candidate and x0 forwarded to the search.
    assert "[1, 2, 24]" in captured["search_space"]["lags"]
    assert "x0" in captured["kwargs_spotoptim"]
    # Best lags applied to the returned forecaster.
    assert list(np.asarray(tuned.lags)) == [1, 2, 24]


def test_strategy_no_x0_when_flag_disabled(monkeypatch):
    """With warm_start_lags None, no x0 is passed and lags is untouched."""
    from spotforecast2.multitask.strategies import SpotOptimStrategy

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        df = pd.DataFrame(
            {
                "lags": [np.array([24])],
                "params": [{"alpha": 1.0}],
                "mean_absolute_error": [0.1],
            }
        )
        return df, None

    monkeypatch.setattr(
        "spotforecast2.model_selection.spotoptim_search_forecaster", fake_search
    )

    task = _MockTask()
    task.config.warm_start_lags = None
    y = pd.Series(
        np.sin(np.arange(400) * 2 * np.pi / 24),
        index=pd.date_range("2022-01-01", periods=400, freq="h"),
        name="load",
    )
    strat = SpotOptimStrategy(
        search_space={"alpha": (0.01, 10.0), "lags": ["24", "12"]}
    )
    strat.prepare_forecaster(
        task, "load", ForecasterRecursive(estimator=Ridge(), lags=24), y, None
    )

    assert captured["kwargs_spotoptim"] is None
    assert captured["search_space"]["lags"] == ["24", "12"]
