# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Training strategies for the multitask pipeline.

Re-exports ``TrainingStrategy``, ``LazyStrategy``, and ``DefaultsStrategy``
from the safe package (tuning-free strategies).  Defines ``OptunaStrategy``
and ``SpotOptimStrategy`` here because they depend on ``spotforecast2``-only
packages (``optuna``, ``spotoptim``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import pandas as pd

# Re-export tuning-free strategies from the safe package so importers of
# ``spotforecast2.multitask.strategies`` continue to find all four names.
from spotforecast2_safe.multitask.strategies import (  # noqa: F401  (re-exported)
    DefaultsStrategy,
    LazyStrategy,
    TrainingStrategy,
)

from spotforecast2.multitask.search_spaces import (
    _default_optuna_search_space,
    _default_spotoptim_search_space,
)


class OptunaStrategy:
    """Approach 3 — Optuna Bayesian tuning, then apply best params."""

    name = "optuna"

    def __init__(self, search_space: Optional[Callable] = None) -> None:
        self.search_space = search_space

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        from spotforecast2.model_selection import bayesian_search_forecaster

        search_space = self.search_space or _default_optuna_search_space
        cv = task.cv_ts(y_train)
        tuning_results, _ = bayesian_search_forecaster(
            forecaster=forecaster,
            y=y_train,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            exog=exog_train,
            n_trials=task.config.n_trials_optuna,
            random_state=task.config.random_state,
            return_best=True,
            verbose=task.config.verbose,
            show_progress=getattr(task, "_show_progress", False),
        )
        best_params = tuning_results.iloc[0].params
        best_lags = tuning_results.iloc[0].lags
        task.logger.info("  Best params: %s", best_params)
        task.logger.info("  Best lags: %s", best_lags)
        task.save_tuning_results(
            target=target,
            task_name="optuna",
            best_params=best_params,
            best_lags=best_lags,
        )
        tuned = task.create_forecaster(target=target)
        tuned.set_params(**best_params)
        if hasattr(tuned, "set_lags"):
            tuned.set_lags(best_lags)
        return tuned


class SpotOptimStrategy:
    """Approach 4 — SpotOptim surrogate-model tuning, then apply best params."""

    name = "spotoptim"

    def __init__(self, search_space: Optional[Dict[str, Any]] = None) -> None:
        self.search_space = search_space

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        from spotforecast2.model_selection import (
            build_warm_start_x0,
            spotoptim_search_forecaster,
        )

        search_space = self.search_space or _default_spotoptim_search_space()
        cv = task.cv_ts(y_train)

        # Warm start: inject ``lags_consider`` as a candidate lag set and seed
        # the optimizer's first evaluation with it.  Only dict search spaces
        # with a ``"lags"`` list are eligible; anything else falls through to a
        # normal cold-start run.
        kwargs_spotoptim: Dict[str, Any] = {}
        lags_seed = getattr(task.config, "lags_consider", None)
        if (
            getattr(task.config, "warm_start_lags", False)
            and lags_seed
            and isinstance(search_space, dict)
            and isinstance(search_space.get("lags"), list)
        ):
            seed_str = str(list(lags_seed))
            search_space = deepcopy(search_space)
            if seed_str not in search_space["lags"]:
                search_space["lags"] = [seed_str, *search_space["lags"]]
            x0 = build_warm_start_x0(search_space, forecaster, lags_seed)
            if x0 is not None:
                kwargs_spotoptim["x0"] = x0
                task.logger.info("  Warm-start lags seeded: %s", seed_str)

        # Parallel evaluation: forward the configured worker count straight to
        # SpotOptim (``kwargs_spotoptim`` is spread into its constructor).
        n_jobs_spotoptim = getattr(task.config, "n_jobs_spotoptim", None)
        if n_jobs_spotoptim is not None:
            kwargs_spotoptim["n_jobs"] = n_jobs_spotoptim
            task.logger.info("  SpotOptim n_jobs: %s", n_jobs_spotoptim)

        tuning_results, _ = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_train,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            exog=exog_train,
            return_best=True,
            random_state=task.config.random_state,
            verbose=False,
            n_trials=task.config.n_trials_spotoptim,
            n_initial=task.config.n_initial_spotoptim,
            show_progress=getattr(task, "_show_progress", False),
            kwargs_spotoptim=kwargs_spotoptim or None,
        )
        best_params = tuning_results.iloc[0].params
        best_lags = tuning_results.iloc[0].lags
        task.logger.info("  Best params: %s", best_params)
        task.logger.info("  Best lags: %s", best_lags)
        task.save_tuning_results(
            target=target,
            task_name="spotoptim",
            best_params=best_params,
            best_lags=best_lags,
        )
        tuned = task.create_forecaster(target=target)
        tuned.set_params(**best_params)
        if hasattr(tuned, "set_lags"):
            tuned.set_lags(best_lags)
        return tuned
