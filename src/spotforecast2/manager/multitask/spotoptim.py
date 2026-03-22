# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SpotOptim surrogate-model Bayesian tuning task — Task 4.

Uses ``spotoptim`` for surrogate-model-based Bayesian optimisation.
Effective with small trial budgets.
"""

from typing import Any, Dict, Optional

from spotforecast2.manager.multitask.base import BaseTask
from spotforecast2.model_selection import spotoptim_search_forecaster
from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold


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


def execute_spotoptim(
    task: BaseTask,
    show: bool = True,
    search_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute SpotOptim tuning for all targets on ``task``.

    Args:
        task: A class `BaseTask` (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.
        search_space: Dictionary defining the SpotOptim search space.
            ``None`` uses the built-in default.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["spotoptim"]``.
    """
    task._ensure_pipeline_ready()
    if search_space is None:
        search_space = _default_spotoptim_search_space()
    results: Dict[str, Dict[str, Any]] = {}

    for target in task.config.targets:
        task.logger.info(
            "[task 4] Target '%s': SpotOptim tuning (%d trials)...",
            target,
            task.config.n_trials_spotoptim,
        )
        y_train, exog_train, exog_future = task._get_target_data(target)
        forecaster = task.create_forecaster()

        end_cv = task.config.end_train_ts - task.config.delta_val
        n_train_cv = len(y_train.loc[:end_cv])
        cv = TimeSeriesFold(
            steps=task.config.predict_size,
            refit=False,
            initial_train_size=n_train_cv,
            fixed_train_size=(task.config.train_size is not None),
            gap=0,
            allow_incomplete_fold=True,
        )

        tuning_results, optimizer = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_train,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            exog=exog_train,
            return_best=True,
            random_state=task.config.random_state,
            verbose=True,
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

        forecaster_tuned = task.create_forecaster()
        forecaster_tuned.set_params(**best_params)
        if hasattr(forecaster_tuned, "set_lags"):
            forecaster_tuned.set_lags(best_lags)

        results[target] = task._train_and_predict_target(
            target=target,
            task_name="task 4: SpotOptim Tuned",
            forecaster=forecaster_tuned,
            y_train=y_train,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        if show:
            task._show_prediction_figure(
                results[target], target, "task 4: SpotOptim Tuned"
            )

    task.results["spotoptim"] = results
    agg_pkg = task._aggregate_and_show(results, "task 4: SpotOptim Tuned", show=show)
    return agg_pkg


class SpotOptimTask(BaseTask):
    """Task 4 — SpotOptim surrogate-model Bayesian tuning.

    Uses ``spotoptim`` for surrogate-model-based Bayesian optimisation.
    Effective with small trial budgets.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import SpotOptimTask

        task = SpotOptimTask(n_trials_spotoptim=10, predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"SpotOptim trials: {task.config.n_trials_spotoptim}")
        ```
    """

    _task_name = "spotoptim"

    @staticmethod
    def _default_spotoptim_search_space() -> Dict[str, Any]:
        """Built-in SpotOptim search space for LightGBM."""
        return _default_spotoptim_search_space()

    def run(
        self,
        show: bool = True,
        search_space: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run SpotOptim surrogate-model tuning for all targets.

        Args:
            show: If ``True``, display prediction figures.
            search_space: Dictionary defining the SpotOptim search space.
                ``None`` uses the built-in default.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["spotoptim"]``.
        """
        return execute_spotoptim(self, show=show, search_space=search_space)
