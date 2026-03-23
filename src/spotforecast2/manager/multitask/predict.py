# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Predict-only task — loads previously saved models and forecasts.

``PredictTask`` does not train any model.  It loads fitted forecasters
that were persisted by a prior ``LazyTask``, ``OptunaTask``, or
``SpotOptimTask`` run and uses them to generate predictions for all
configured targets.  If no saved models are found, execution halts
with an informative error.
"""

from typing import Any, Dict, Optional

from spotforecast2.manager.multitask.base import BaseTask
from spotforecast2_safe.manager.predictor import build_prediction_package


def execute_predict(
    task: BaseTask,
    show: bool = True,
    task_name: Optional[str] = None,
    max_age_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute prediction-only mode using previously saved models.

    Loads the most recent fitted forecaster for every configured target
    from the cache directory.  No training is performed.  If no saved
    models can be found the function raises ``RuntimeError``.

    Args:
        task: A ``BaseTask`` (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.
        task_name: Restrict model loading to a specific source task
            (``"lazy"``, ``"optuna"``, or ``"spotoptim"``).
            ``None`` loads the most recent model regardless of source.
        max_age_days: Maximum age in days for saved models.
            Models older than this are ignored.  ``None`` accepts any age.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["predict"]``.

    Raises:
        RuntimeError: If no saved models are found in the cache directory,
            or if a target has no matching saved model.
    """
    task._ensure_pipeline_ready()

    loaded_models = task.load_models(
        task_name=task_name,
        max_age_days=max_age_days,
    )

    if not loaded_models:
        raise RuntimeError(
            "No saved models found in the cache directory "
            f"'{task.data_frame_name}'. "
            "Run LazyTask, OptunaTask, or SpotOptimTask first to train "
            "and save models before using PredictTask."
        )

    results: Dict[str, Dict[str, Any]] = {}

    for target in task.config.targets:
        if target not in loaded_models:
            raise RuntimeError(
                f"No saved model found for target '{target}'. "
                "Run a training task (lazy, optuna, or spotoptim) for "
                "this target before using PredictTask."
            )

        task.logger.info("[predict] Target '%s': loading saved model...", target)

        forecaster = loaded_models[target]
        y_train, exog_train, exog_future = task._get_target_data(target)

        results[target] = build_prediction_package(
            forecaster=forecaster,
            target=target,
            y_train=y_train,
            predict_size=task.config.predict_size,
            exog_train=exog_train,
            exog_future=exog_future,
            df_test=task.df_test,
        )
        if show:
            task._show_prediction_figure(
                results[target], target, "task 5: Predict (loaded models)"
            )

    task.results["predict"] = results
    agg_pkg = task._aggregate_and_show(
        results, "task 5: Predict (loaded models)", show=show
    )
    return agg_pkg


class PredictTask(BaseTask):
    """Task 5 — Predict-only using previously saved models.

    Loads fitted forecasters that were persisted by a prior
    ``LazyTask``, ``OptunaTask``, or ``SpotOptimTask`` run and produces
    predictions for all configured targets.  No training or tuning is
    performed.

    If no saved models exist in the cache directory the ``run`` method
    raises ``RuntimeError`` with an informative message.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import PredictTask

        task = PredictTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "predict"

    def run(
        self,
        show: bool = True,
        task_name: Optional[str] = None,
        max_age_days: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run prediction using previously saved models.

        Args:
            show: If ``True``, display prediction figures.
            task_name: Restrict model loading to a specific source task
                (``"lazy"``, ``"optuna"``, or ``"spotoptim"``).
                ``None`` loads the most recent model regardless of source.
            max_age_days: Maximum age in days for saved models.
                Models older than this are ignored.  ``None`` accepts
                any age.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["predict"]``.

        Raises:
            RuntimeError: If no saved models are found in the cache
                directory, or if a target has no matching model.
        """
        return execute_predict(
            self,
            show=show,
            task_name=task_name,
            max_age_days=max_age_days,
        )
