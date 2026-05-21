# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optuna Bayesian hyperparameter tuning task — Task 3.

Uses Optuna's TPE sampler to search for optimal LightGBM
hyperparameters, then re-fits with the best discovered parameters.
"""

from typing import Any, Callable, Dict, Optional

from spotforecast2.multitask.base import BaseTask
from spotforecast2.multitask.strategies import OptunaStrategy


def _default_optuna_search_space(trial: Any) -> Dict[str, Any]:
    """Built-in Optuna search space for LightGBM."""
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


def execute_optuna(
    task: BaseTask,
    show: bool = True,
    search_space: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Execute Optuna Bayesian tuning for all targets on ``task``.

    Thin wrapper around ``BaseTask._run_strategy`` using ``OptunaStrategy``.

    Args:
        task: A BaseTask (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.
        search_space: Callable ``(trial) -> dict`` defining the Optuna
            search space.  ``None`` uses the built-in default.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["optuna"]``.
        When ``task.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so PredictTask can load them directly.
    """
    strategy = OptunaStrategy(search_space=search_space)
    return task._run_strategy(
        strategy,
        task_name="task 3: Optuna Tuned",
        results_key="optuna",
        show=show,
        log_prefix="[task 3] ",
    )


class OptunaTask(BaseTask):
    """Task 3 — Optuna Bayesian hyperparameter tuning.

    Uses Optuna's TPE sampler to search for optimal LightGBM
    hyperparameters, then re-fits with the best discovered parameters.

    Examples:
        ```{python}
        from spotforecast2.multitask import OptunaTask

        task = OptunaTask(n_trials_optuna=5, predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Optuna trials: {task.config.n_trials_optuna}")
        ```
    """

    _task_name = "optuna"

    @staticmethod
    def _default_optuna_search_space(trial: Any) -> Dict[str, Any]:
        """Built-in Optuna search space for LightGBM."""
        return _default_optuna_search_space(trial)

    def run(
        self,
        show: bool = True,
        search_space: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run Optuna Bayesian tuning for all targets.

        Args:
            show: If ``True``, display prediction figures.
            search_space: Callable ``(trial) -> dict``.  ``None`` uses
                the built-in default.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["optuna"]``.
        """
        return execute_optuna(self, show=show, search_space=search_space)
