# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Explicit pre-training task — Task 2.

Fits each target with default parameters and serialises the forecaster
to disk before building the prediction package.
"""

from typing import Any, Dict

from spotforecast2_safe.manager.predictor import build_prediction_package

from spotforecast2.manager.multitask.base import BaseTask


def execute_training(task: BaseTask, show: bool = True) -> Dict[str, Any]:
    """Execute explicit pre-training for all targets on ``task``.

    Args:
        task: A class `BaseTask` (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["training"]``.
    """
    task._ensure_pipeline_ready()
    results: Dict[str, Dict[str, Any]] = {}

    for target in task.config.targets:
        task.logger.info("[task 2] Target '%s': explicit pre-training...", target)
        y_train, exog_train, exog_future = task._get_target_data(target)
        forecaster = task.create_forecaster()
        forecaster.fit(y=y_train, exog=exog_train)
        task._save_forecaster(forecaster, "task 2: Trained (No Tuning)", target)
        task.logger.info("  [task 2] '%s': forecaster fitted and serialized.", target)

        pred_pkg = build_prediction_package(
            forecaster=forecaster,
            target=target,
            y_train=y_train,
            predict_size=task.config.predict_size,
            exog_train=exog_train,
            exog_future=exog_future,
            df_test=task.df_test,
        )
        results[target] = pred_pkg
        if show:
            task._show_prediction_figure(
                pred_pkg, target, "task 2: Explicit Pre-Training"
            )

    task.results["training"] = results
    agg_pkg = task._aggregate_and_show(
        results, "task 2: Explicit Pre-Training", show=show
    )
    return agg_pkg


class TrainTask(BaseTask):
    """Task 2 — Explicit Pre-Training with default parameters.

    Fits each forecaster, serialises it to disk, then builds the
    prediction package from the pre-fitted model.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import TrainTask

        task = TrainTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "training"

    def run(self, show: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Run explicit pre-training for all targets.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["training"]``.
        """
        return execute_training(self, show=show)
