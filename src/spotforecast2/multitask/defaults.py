# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Defaults task — Task 2.

Fits each target with the factory's default hyperparameters.  Distinct from
:class:`spotforecast2.multitask.lazy.LazyTask` in one respect only:
``DefaultsTask`` never consults the tuning-result cache.  Use it when you
want a deterministic baseline that is guaranteed not to inherit any prior
Optuna or SpotOptim run.

ENTSO-E "Approach 2: Training without Tuning" routes here.
"""

from typing import Any, Dict

from spotforecast2.multitask.base import BaseTask
from spotforecast2.multitask.strategies import DefaultsStrategy


def execute_defaults(
    task: BaseTask,
    show: bool = True,
) -> Dict[str, Any]:
    """Execute defaults fitting for all targets on ``task``.

    Thin wrapper around ``BaseTask._run_strategy`` using ``DefaultsStrategy``.

    Args:
        task: A BaseTask (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.

    Returns:
        Aggregated prediction package (weighted combination of all targets,
        or the single-target package when ``len(config.targets) == 1``).
        Per-target packages are stored on ``task.results["defaults"]``.
        When ``task.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so ``PredictTask(task_name="defaults")`` can
        load them directly.
    """
    return task._run_strategy(
        DefaultsStrategy(),
        task_name="task 2: Defaults",
        results_key="defaults",
        show=show,
        log_prefix="[task 2] ",
    )


class DefaultsTask(BaseTask):
    """Task 2 — Defaults fitting (no tuning, no cached params).

    Creates an unfitted forecaster per target via ``config.forecaster_factory``
    (or the package default) and fits with whatever parameters that factory
    chooses.  Unlike ``LazyTask``, never reads the tuning-result cache.

    Examples:
        ```{python}
        from spotforecast2.multitask import DefaultsTask

        task = DefaultsTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "defaults"

    def run(
        self,
        show: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run defaults fitting for all targets.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package.  Per-target packages are stored
            on ``self.results["defaults"]``.
        """
        return execute_defaults(self, show=show)
