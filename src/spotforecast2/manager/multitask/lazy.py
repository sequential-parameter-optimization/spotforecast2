# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lazy-fitting task — Task 1.

Fits each target with default LightGBM parameters (no tuning).
"""

from typing import Any, Dict

from spotforecast2.manager.multitask.base import BaseTask


def execute_lazy(task: BaseTask, show: bool = True) -> Dict[str, Any]:
    """Execute lazy fitting for all targets on ``task``.

    Args:
        task: A class `BaseTask` (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["lazy"]``.
    """
    task._ensure_pipeline_ready()
    results: Dict[str, Dict[str, Any]] = {}

    for target in task.config.targets:
        task.logger.info("[task 1] Target '%s': fitting with default params...", target)
        y_train, exog_train, exog_future = task._get_target_data(target)
        forecaster = task.create_forecaster()
        results[target] = task._train_and_predict_target(
            target=target,
            task_name="task 1: Lazy Fitting",
            forecaster=forecaster,
            y_train=y_train,
            exog_train=exog_train,
            exog_future=exog_future,
        )
        if show:
            task._show_prediction_figure(
                results[target], target, "task 1: Lazy Fitting"
            )

    task.results["lazy"] = results
    agg_pkg = task._aggregate_and_show(results, "task 1: Lazy Fitting", show=show)
    return agg_pkg


class LazyTask(BaseTask):
    """Task 1 — Lazy Fitting with default LightGBM parameters.

    Creates an unfitted forecaster per target and fits with default
    hyperparameters.  No cross-validation or tuning is performed.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import LazyTask

        task = LazyTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "lazy"

    def run(self, show: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Run lazy fitting for all targets.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["lazy"]``.
        """
        return execute_lazy(self, show=show)
