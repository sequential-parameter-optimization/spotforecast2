# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SpotOptim surrogate-model Bayesian tuning task — Task 4.

Uses ``spotoptim`` for surrogate-model-based Bayesian optimisation.
Effective with small trial budgets.
"""

from typing import Any, Dict, Optional

from spotforecast2.multitask.base import BaseTask
from spotforecast2.multitask.strategies import SpotOptimStrategy


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

    Thin wrapper around ``BaseTask._run_strategy`` using ``SpotOptimStrategy``.

    Args:
        task: A class `BaseTask` (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.
        search_space: Dictionary defining the SpotOptim search space.
            ``None`` uses the built-in default.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["spotoptim"]``.
        When ``task.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so PredictTask can load them directly.
    """
    strategy = SpotOptimStrategy(search_space=search_space)
    return task._run_strategy(
        strategy,
        task_name="task 4: SpotOptim Tuned",
        results_key="spotoptim",
        show=show,
        log_prefix="[task 4] ",
    )


class SpotOptimTask(BaseTask):
    """Task 4 — SpotOptim surrogate-model Bayesian tuning.

    Uses ``spotoptim`` for surrogate-model-based Bayesian optimisation.
    Effective with small trial budgets.

    Examples:
        ```{python}
        from spotforecast2.multitask import SpotOptimTask

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
