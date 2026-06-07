# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SpotOptim surrogate-model Bayesian tuning task — Task 4.

Uses ``spotoptim`` for surrogate-model-based Bayesian optimisation.
Effective with small trial budgets.
"""

from typing import Any, Dict, Optional

from spotforecast2.multitask.base import BaseTask
from spotforecast2.multitask.search_spaces import _default_spotoptim_search_space
from spotforecast2.multitask.strategies import SpotOptimStrategy

__all__ = [
    "SpotOptimTask",
    "execute_spotoptim",
    "_default_spotoptim_search_space",
]


def execute_spotoptim(
    task: BaseTask,
    show: bool = True,
    search_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute SpotOptim tuning for all targets on ``task``.

    Thin wrapper around ``BaseTask._run_strategy`` using ``SpotOptimStrategy``.

    Args:
        task: A `BaseTask` (or subclass) instance with prepared data.
        show: If ``True``, display prediction figures.
        search_space: Dictionary defining the SpotOptim search space.
            ``None`` uses the built-in default.

    Returns:
        Aggregated prediction package (weighted combination of all targets).
        Per-target packages are stored on ``task.results["spotoptim"]``.
        When ``task.config.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so PredictTask can load them directly.

    Examples:
        ```{python}
        # Demonstrate the strategy wiring that execute_spotoptim sets up.
        # A full run requires prepared data; here we inspect the strategy object
        # and the search space it would use.
        from spotforecast2.multitask.strategies import SpotOptimStrategy
        from spotforecast2.multitask.search_spaces import _default_spotoptim_search_space

        strategy = SpotOptimStrategy()
        print(f"Strategy name: {strategy.name}")
        assert strategy.search_space is None  # uses built-in default when None

        default_space = _default_spotoptim_search_space()
        print(f"Search space keys: {list(default_space.keys())[:4]}")
        assert "lags" in default_space
        assert "estimator__num_leaves" in default_space

        # Custom search space can be injected:
        custom_space = {"lags": ["24", "48"], "estimator__num_leaves": (8, 64)}
        strategy_custom = SpotOptimStrategy(search_space=custom_space)
        assert strategy_custom.search_space is custom_space
        print(f"Custom space lags options: {strategy_custom.search_space['lags']}")
        ```
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

        Examples:
            ```{python}
            # Construct the task and verify configuration before running.
            # A full run requires prepared data (prepare_data, impute, etc.);
            # this example demonstrates construction and config inspection.
            from spotforecast2.multitask.spotoptim import SpotOptimTask

            task = SpotOptimTask(
                n_trials_spotoptim=5,
                n_initial_spotoptim=3,
                predict_size=24,
                auto_save_models=False,
            )
            print(f"Task type: {task.TASK}")
            print(f"Trials: {task.config.n_trials_spotoptim}")
            print(f"Initial evaluations: {task.config.n_initial_spotoptim}")
            assert task.config.n_trials_spotoptim == 5
            assert task.config.auto_save_models is False
            ```
        """
        return execute_spotoptim(self, show=show, search_space=search_space)
