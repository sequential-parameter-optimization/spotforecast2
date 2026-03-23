# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lazy-fitting task — Task 1.

Fits each target with default LightGBM parameters (no tuning).
When cached tuning results are available (from Optuna or SpotOptim),
they are loaded and applied automatically so that the lazy task
benefits from prior tuning without re-running the search.
"""

from typing import Any, Dict, Optional

from spotforecast2.manager.multitask.base import BaseTask


def execute_lazy(
    task: BaseTask,
    show: bool = True,
    use_tuned_params: bool = True,
    max_age_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute lazy fitting for all targets on ``task``.

    When ``use_tuned_params`` is ``True`` (the default), previously saved
    tuning results are loaded from cache and applied to the forecaster.
    If no cached results are found the forecaster uses default parameters.

    Args:
        task: A BaseTask (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.
        use_tuned_params: If ``True``, attempt to load cached tuning
            results (best parameters and lags) for each target.
        max_age_days: Maximum age in days for cached tuning results.
            ``None`` accepts any age.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["lazy"]``.
        When ``task.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so PredictTask can load them directly.
    """
    task._ensure_pipeline_ready()
    results: Dict[str, Dict[str, Any]] = {}

    for target in task.config.targets:
        task.logger.info("[task 1] Target '%s': fitting with default params...", target)
        y_train, exog_train, exog_future = task._get_target_data(target)
        forecaster = task.create_forecaster()

        if use_tuned_params:
            tuned = task.load_tuning_results(target=target, max_age_days=max_age_days)
            if tuned is not None:
                task.logger.info(
                    "  Applying cached %s tuning results (from %s).",
                    tuned["task_name"],
                    tuned["timestamp"],
                )
                forecaster.set_params(**tuned["best_params"])
                if hasattr(forecaster, "set_lags"):
                    forecaster.set_lags(tuned["best_lags"])

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
    if getattr(task, "auto_save_models", True):
        task.save_models(task_name="lazy")
    agg_pkg = task._aggregate_and_show(results, "task 1: Lazy Fitting", show=show)
    return agg_pkg


class LazyTask(BaseTask):
    """Task 1 — Lazy Fitting with default LightGBM parameters.

    Creates an unfitted forecaster per target and fits with default
    hyperparameters.  No cross-validation or tuning is performed.

    When cached tuning results are available (saved by
    OptunaTask or SpotOptimTask), they are loaded and applied automatically
    so that the lazy task benefits from prior tuning without re-running
    the search.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import LazyTask

        task = LazyTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "lazy"

    def run(
        self,
        show: bool = True,
        use_tuned_params: bool = True,
        max_age_days: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run lazy fitting for all targets.

        Args:
            show: If ``True``, display prediction figures.
            use_tuned_params: If ``True``, load and apply cached tuning
                results for each target.
            max_age_days: Maximum age in days for cached tuning results.
                ``None`` accepts any age.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["lazy"]``.
        """
        return execute_lazy(
            self,
            show=show,
            use_tuned_params=use_tuned_params,
            max_age_days=max_age_days,
        )
