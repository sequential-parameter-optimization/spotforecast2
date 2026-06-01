# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parallel SpotOptim tuning (``n_jobs > 1``).

Two things are exercised:

1. ``SpotOptimStrategy`` forwards the configured worker count to SpotOptim via
   ``kwargs_spotoptim["n_jobs"]``.
2. ``spotoptim_search_forecaster`` returns a complete, correct ``results`` table
   in parallel mode. In parallel the objective runs in worker processes, so its
   side-effect accumulators never reach the parent; the results are rebuilt from
   the optimizer's own evaluation history (``optimizer.X_`` / ``y_``). The
   rebuilt table must match a sequential run on the seeded initial design.

Requires the steady-state ``inverse_transform`` fix in ``spotoptim >= 0.12.3``
(without it a log10-transformed hyperparameter crashes the GP surrogate).
"""

import types

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.splitter import TimeSeriesFold
from spotforecast2.model_selection import spotoptim_search_forecaster
from spotforecast2.multitask.strategies import SpotOptimStrategy


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 168)
    y = y + rng.normal(0, 0.1, n) + 100.0
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.Series(y, index=idx, name="load")


def _run(n_jobs):
    forecaster = ForecasterRecursive(estimator=Ridge(alpha=1.0), lags=24)
    cv = TimeSeriesFold(steps=24, initial_train_size=300, refit=False)
    search_space = {"lags": ["24", "[1, 2, 24]"], "alpha": (0.01, 10.0, "log10")}
    kwargs = {"n_jobs": n_jobs} if n_jobs and n_jobs > 1 else None
    results, optimizer = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=_series(),
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=8,
        n_initial=4,
        random_state=123,
        return_best=False,
        verbose=False,
        show_progress=False,
        kwargs_spotoptim=kwargs,
    )
    return results, optimizer


def test_parallel_results_are_not_empty_and_decode():
    results, optimizer = _run(n_jobs=2)
    # The optimizer evaluated every trial...
    assert len(optimizer.X_) == 8
    # ...and the results table is fully reconstructed (not empty -- the bug this
    # guards against returned 0 rows in parallel).
    assert len(results) == 8
    assert "lags" in results.columns
    assert "alpha" in results.columns
    assert "mean_absolute_error" in results.columns
    # Decoded hyperparameters are sane: alpha in its natural bound, lags parsed
    # back to a list of ints.
    assert results["alpha"].between(0.01, 10.0).all()
    assert all(isinstance(lg, (list, np.ndarray)) for lg in results["lags"])
    assert np.isfinite(results["mean_absolute_error"]).all()


def test_parallel_matches_sequential_best():
    seq, _ = _run(n_jobs=1)
    par, _ = _run(n_jobs=2)
    # The seeded initial design is shared, so the best score agrees closely.
    assert seq["mean_absolute_error"].min() == pytest.approx(
        par["mean_absolute_error"].min(), rel=1e-3
    )


def test_strategy_forwards_n_jobs_to_spotoptim(monkeypatch):
    """``SpotOptimStrategy`` puts ``config.n_jobs_spotoptim`` into the
    ``kwargs_spotoptim`` passed to ``spotoptim_search_forecaster``."""
    captured = {}

    class _Stop(Exception):
        pass

    def _fake_search(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop()

    # prepare_forecaster imports the name from spotforecast2.model_selection at
    # call time, so patch it on that package.
    import spotforecast2.model_selection as ms

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", _fake_search)

    config = types.SimpleNamespace(
        warm_start_lags=False,
        lags_consider=None,
        n_jobs_spotoptim=-1,
        random_state=42,
        n_trials_spotoptim=5,
        n_initial_spotoptim=3,
    )
    task = types.SimpleNamespace(
        config=config,
        cv_ts=lambda y: object(),
        logger=types.SimpleNamespace(info=lambda *a, **k: None),
        _show_progress=False,
    )

    strategy = SpotOptimStrategy()
    with pytest.raises(_Stop):
        strategy.prepare_forecaster(
            task=task,
            target="Actual Load",
            forecaster=object(),
            y_train=_series(),
            exog_train=None,
        )

    assert captured["kwargs_spotoptim"] == {"n_jobs": -1}


def test_strategy_omits_n_jobs_when_unset(monkeypatch):
    """With ``n_jobs_spotoptim=None`` (default) no ``n_jobs`` is forwarded, so the
    optimizer stays sequential."""
    captured = {}

    class _Stop(Exception):
        pass

    def _fake_search(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop()

    import spotforecast2.model_selection as ms

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", _fake_search)

    config = types.SimpleNamespace(
        warm_start_lags=False,
        lags_consider=None,
        n_jobs_spotoptim=None,
        random_state=42,
        n_trials_spotoptim=5,
        n_initial_spotoptim=3,
    )
    task = types.SimpleNamespace(
        config=config,
        cv_ts=lambda y: object(),
        logger=types.SimpleNamespace(info=lambda *a, **k: None),
        _show_progress=False,
    )

    with pytest.raises(_Stop):
        SpotOptimStrategy().prepare_forecaster(
            task=task,
            target="Actual Load",
            forecaster=object(),
            y_train=_series(),
            exog_train=None,
        )

    # No warm start, no n_jobs -> kwargs_spotoptim collapses to None.
    assert captured["kwargs_spotoptim"] is None
