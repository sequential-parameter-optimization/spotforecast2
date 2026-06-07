# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optuna Bayesian hyperparameter tuning task — Task 3.

Uses Optuna's TPE sampler to search for optimal LightGBM
hyperparameters, then re-fits with the best discovered parameters.
"""

from typing import Any, Callable, Dict, Optional

from spotforecast2.multitask.base import BaseTask
from spotforecast2.multitask.search_spaces import _default_optuna_search_space
from spotforecast2.multitask.strategies import OptunaStrategy

__all__ = [
    "OptunaTask",
    "execute_optuna",
    "_default_optuna_search_space",
]


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
        When ``task.config.auto_save_models`` is ``True`` (the default), fitted
        models are saved to disk so PredictTask can load them directly.

    Examples:
        ```{python}
        import warnings
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
        from spotforecast2.multitask import OptunaTask
        from spotforecast2.multitask.optuna import execute_optuna

        data_home = get_package_data_home()
        df = fetch_data(filename=str(data_home / "demo10.csv"))
        tiny_df = df.iloc[:500][["A"]]

        task = OptunaTask(
            n_trials_optuna=2,
            predict_size=24,
            auto_save_models=False,
            lags_consider=[1, 2, 24],
            number_folds=2,
            verbose=False,
        )
        task.prepare_data(demo_data=tiny_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = execute_optuna(task, show=False)

        assert isinstance(result, dict)
        assert "future_pred" in result
        print("execute_optuna result keys:", sorted(result.keys()))
        ```
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

        Examples:
            ```{python}
            import warnings
            from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
            from spotforecast2.multitask import OptunaTask

            data_home = get_package_data_home()
            df = fetch_data(filename=str(data_home / "demo10.csv"))
            tiny_df = df.iloc[:500][["A"]]

            task = OptunaTask(
                n_trials_optuna=2,
                predict_size=24,
                auto_save_models=False,
                lags_consider=[1, 2, 24],
                number_folds=2,
                verbose=False,
            )
            task.prepare_data(demo_data=tiny_df)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = task.run(show=False)

            assert "future_pred" in result
            assert result.get("validation_passed") is True
            print("OptunaTask.run result keys:", sorted(result.keys()))
            ```
        """
        return execute_optuna(self, show=show, search_space=search_space)
