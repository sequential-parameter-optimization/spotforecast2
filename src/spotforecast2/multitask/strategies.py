# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Training strategies for the multitask pipeline.

Each strategy encapsulates the per-target "prepare a forecaster for the
final fit" step that the existing ``execute_lazy`` / ``execute_optuna`` /
``execute_spotoptim`` functions perform between ``create_forecaster()`` and
``_train_and_predict_target()``.

The protocol is introduced here as scaffolding for the ENTSO-E integration
(ADR-001 §5).  Step 4 of the migration only introduces the file — the
concrete strategies are not yet called from ``BaseTask``.  Step 5 will
rewire ``execute_*`` to delegate to these classes.

The four concrete strategies map onto the ENTSO-E publication's "Approach
1-4":

- :class:`LazyStrategy` — Approach 1.  Optionally applies cached tuning
  results; otherwise leaves the forecaster at default parameters.
- :class:`DefaultsStrategy` — Approach 2.  Explicit "train with defaults,
  no tuning, no cached params."  Stub: raises ``NotImplementedError`` until
  the ENTSO-E integration wires it in.
- :class:`OptunaStrategy` — Approach 3.  Runs Optuna Bayesian search and
  applies best parameters.
- :class:`SpotOptimStrategy` — Approach 4.  Runs SpotOptim surrogate search
  and applies best parameters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol

import pandas as pd


def _default_optuna_search_space(trial: Any) -> Dict[str, Any]:
    """Built-in Optuna search space for LightGBM.

    Lives here (rather than alongside :func:`execute_optuna`) so that
    :class:`OptunaStrategy` can reach the default without re-importing the
    ``optuna`` shim — which would re-introduce the cyclic import the
    strategy seam was meant to dissolve.
    """
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100),
        "lags": trial.suggest_categorical(
            "lags",
            [24, 48, [1, 2, 24], [1, 2, 24, 48], [1, 2, 23, 24, 47, 48]],
        ),
    }


def _default_spotoptim_search_space() -> Dict[str, Any]:
    """Built-in SpotOptim search space for LightGBM."""
    return {
        "num_leaves": (8, 256),
        "max_depth": (3, 16),
        "learning_rate": (0.0001, 0.1, "log10"),
        "n_estimators": (10, 1000, "log10"),
        "bagging_fraction": (0.5, 1.0),
        "feature_fraction": (0.5, 1.0),
        "reg_alpha": (0.01, 100.0),
        "reg_lambda": (0.01, 100.0),
        "lags": [
            "[1, 2, 3, 11, 12, 22, 23, 24, 47, 48, 167, 168]",
            "48",
            "24",
            "[1, 2, 24, 48]",
            "[1, 2, 23, 24, 47, 48]",
            "[1, 2, 11, 12, 23, 24, 167, 168]",
        ],
    }


class TrainingStrategy(Protocol):
    """Strategy interface for preparing a forecaster before the final fit.

    Implementations return a forecaster with any tuning/parameter changes
    applied.  The final ``forecaster.fit(...)`` and prediction packaging are
    performed by ``BaseTask._train_and_predict_target`` after this call.
    """

    name: str

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Return a forecaster ready for the final fit step."""


class LazyStrategy:
    """Approach 1 — Lazy fitting with optional cached tuning.

    Mirrors the body of ``execute_lazy`` between ``create_forecaster()`` and
    ``_train_and_predict_target()``.
    """

    name = "lazy"

    def __init__(
        self,
        use_tuned_params: bool = True,
        max_age_days: Optional[float] = None,
    ) -> None:
        self.use_tuned_params = use_tuned_params
        self.max_age_days = max_age_days

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        del y_train, exog_train  # unused by this strategy
        if not self.use_tuned_params:
            return forecaster
        tuned = task.load_tuning_results(target=target, max_age_days=self.max_age_days)
        if tuned is None:
            return forecaster
        task.logger.info(
            "  Applying cached %s tuning results (from %s).",
            tuned["task_name"],
            tuned["timestamp"],
        )
        forecaster.set_params(**tuned["best_params"])
        if hasattr(forecaster, "set_lags"):
            forecaster.set_lags(tuned["best_lags"])
        return forecaster


class DefaultsStrategy:
    """Approach 2 — Train with defaults, no tuning, no cached params.

    Stub: raises ``NotImplementedError``.  Will be wired in during the
    ENTSO-E integration (ADR-001 §5).
    """

    name = "defaults"

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        raise NotImplementedError(
            "DefaultsStrategy training will be wired in the ENTSO-E "
            "integration; see ADR-001 §5."
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
            verbose=task.verbose,
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
        from spotforecast2.model_selection import spotoptim_search_forecaster

        search_space = self.search_space or _default_spotoptim_search_space()
        cv = task.cv_ts(y_train)
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
