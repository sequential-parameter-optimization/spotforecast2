# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optuna Bayesian hyperparameter tuning task — Task 3.

Uses Optuna's TPE sampler to search for optimal LightGBM
hyperparameters, then re-fits with the best discovered parameters.
"""

from typing import Any, Callable, Dict, Optional

from spotforecast2.manager.multitask.base import BaseTask
from spotforecast2.model_selection import bayesian_search_forecaster


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
    task._ensure_pipeline_ready()
    if search_space is None:
        search_space = _default_optuna_search_space
    results: Dict[str, Dict[str, Any]] = {}

    for target in task.config.targets:
        task.logger.info(
            "[task 3] Target '%s': Optuna tuning (%d trials)...",
            target,
            task.config.n_trials_optuna,
        )
        y_train, exog_train, exog_future = task._get_target_data(target)
        forecaster = task.create_forecaster()

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
            return_best=False,
            verbose=False,
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

        forecaster_tuned = task.create_forecaster()
        forecaster_tuned.set_params(**best_params)
        if hasattr(forecaster_tuned, "set_lags"):
            forecaster_tuned.set_lags(best_lags)

        results[target] = task._train_and_predict_target(
            target=target,
            task_name="task 3: Optuna Tuned",
            forecaster=forecaster_tuned,
            y_train=y_train,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        if show:
            task._show_prediction_figure(
                results[target], target, "task 3: Optuna Tuned"
            )

    task.results["optuna"] = results
    if getattr(task, "auto_save_models", True):
        task.save_models(task_name="optuna")
    agg_pkg = task._aggregate_and_show(results, "task 3: Optuna Tuned", show=show)
    return agg_pkg


class OptunaTask(BaseTask):
    """Task 3 — Optuna Bayesian hyperparameter tuning.

    Uses Optuna's TPE sampler to search for optimal LightGBM
    hyperparameters, then re-fits with the best discovered parameters.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import OptunaTask

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
