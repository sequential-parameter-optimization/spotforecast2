# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backward-compatible MultiTask dispatcher.

class `MultiTask` preserves the original API where a single ``TASK``
parameter selects which pipeline mode to run.  It inherits from
class `~.base.BaseTask` and delegates ``run()`` to the appropriate
task-specific function.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from spotforecast2.manager.multitask.base import BaseTask
from spotforecast2.manager.multitask.lazy import execute_lazy
from spotforecast2.manager.multitask.train import execute_training
from spotforecast2.manager.multitask.optuna import (
    OptunaTask,
    execute_optuna,
)
from spotforecast2.manager.multitask.spotoptim import (
    SpotOptimTask,
    execute_spotoptim,
)


class MultiTask(BaseTask):
    """Orchestrates a multi-target time-series forecasting pipeline.

    The typical usage flow is:

    1. Instantiate with configuration arguments.
    2. Call method `prepare_data` to load, resample, and validate data.
    3. Call method `detect_outliers` to apply hard bounds and IsolationForest.
    4. Call method `impute` to fill gaps.
    5. Call method `build_exogenous_features` to construct weather / calendar /
       day-night / holiday covariates.
    6. Call method `run` (or individual ``run_task_*`` methods) to train,
       predict, and aggregate.

    Args:
        task: Pipeline task mode — ``"lazy"``, ``"training"``, ``"optuna"``,
            or ``"spotoptim"``. Defaults to ``"lazy"``.
        data_frame_name: Active dataset identifier (e.g. ``"demo10"``).
        data_source: Input CSV filename.
        data_test: Test CSV filename.
        data_home: Path to the data directory.
        cache_data: Whether to cache intermediate data to disk.
        cache_home: Cache directory path.
        agg_weights: Per-target aggregation weights.
        index_name: Datetime column name in the raw CSV.
        number_folds: Number of validation folds.
        predict_size: Forecast horizon in hours.
        bounds: Per-column hard outlier bounds ``(lower, upper)``.
        contamination: IsolationForest contamination fraction.
        imputation_method: Gap-filling strategy.
        use_exogenous_features: Whether to build exogenous features.
        n_trials_optuna: Number of Optuna Bayesian-search trials.
        n_trials_spotoptim: Number of SpotOptim surrogate-search trials.
        n_initial_spotoptim: Initial random evaluations for SpotOptim.
        log_level: Logging level for the pipeline logger.
        config_overrides: Extra keyword arguments forwarded to
            class `~spotforecast2_safe.manager.configurator.config_multi.ConfigMulti`.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import MultiTask

        mt = MultiTask(
            task="lazy",
            data_frame_name="demo10",
            predict_size=24,
            n_trials_optuna=5,
        )
        print(f"Task: {mt.TASK}")
        print(f"Predict size: {mt.config.predict_size}")
        print(f"Data source: {mt.data_source}")
        ```

        ```{python}
        from spotforecast2.manager.multitask import MultiTask

        mt = MultiTask(
            task="training",
            agg_weights=[1.0, -1.0, 0.5],
            contamination=0.05,
        )
        print(f"Agg weights: {mt.config.agg_weights}")
        print(f"Contamination: {mt.config.contamination}")
        print(f"Imputation: {mt.imputation_method}")
        ```
    """

    def __init__(
        self,
        *,
        task: str = "lazy",
        data_frame_name: str = "demo10",
        data_source: str = "demo10.csv",
        data_test: str = "demo11.csv",
        data_home: Optional[Path] = None,
        cache_data: bool = True,
        cache_home: Optional[Path] = None,
        agg_weights: Optional[List[float]] = None,
        index_name: str = "DateTime",
        number_folds: int = 10,
        predict_size: int = 24,
        bounds: Optional[List[tuple]] = None,
        contamination: float = 0.03,
        imputation_method: str = "weighted",
        use_exogenous_features: bool = True,
        n_trials_optuna: int = 15,
        n_trials_spotoptim: int = 10,
        n_initial_spotoptim: int = 5,
        log_level: int = logging.INFO,
        **config_overrides: Any,
    ) -> None:
        # Set _task_name before super().__init__ so self.TASK is correct
        self._task_name = task
        super().__init__(
            data_frame_name=data_frame_name,
            data_source=data_source,
            data_test=data_test,
            data_home=data_home,
            cache_data=cache_data,
            cache_home=cache_home,
            agg_weights=agg_weights,
            index_name=index_name,
            number_folds=number_folds,
            predict_size=predict_size,
            bounds=bounds,
            contamination=contamination,
            imputation_method=imputation_method,
            use_exogenous_features=use_exogenous_features,
            n_trials_optuna=n_trials_optuna,
            n_trials_spotoptim=n_trials_spotoptim,
            n_initial_spotoptim=n_initial_spotoptim,
            log_level=log_level,
            **config_overrides,
        )

    # ------------------------------------------------------------------
    # Task-specific convenience methods
    # ------------------------------------------------------------------

    def run_task_lazy(self, show: bool = True) -> Dict[str, Any]:
        """Task 1 — Lazy Fitting with default LightGBM parameters.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["lazy"]``.
        """
        return execute_lazy(self, show=show)

    def run_task_training(self, show: bool = True) -> Dict[str, Any]:
        """Task 2 — Explicit Pre-Training with default parameters.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["training"]``.
        """
        return execute_training(self, show=show)

    def run_task_optuna(
        self,
        search_space: Optional[Callable] = None,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Task 3 — Optuna Bayesian hyperparameter tuning.

        Args:
            search_space: Callable ``(trial) -> dict``.
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["optuna"]``.
        """
        return execute_optuna(self, show=show, search_space=search_space)

    def run_task_spotoptim(
        self,
        search_space: Optional[Dict[str, Any]] = None,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Task 4 — SpotOptim surrogate-model Bayesian tuning.

        Args:
            search_space: Dictionary defining the SpotOptim search space.
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["spotoptim"]``.
        """
        return execute_spotoptim(self, show=show, search_space=search_space)

    # ------------------------------------------------------------------
    # Run dispatcher
    # ------------------------------------------------------------------

    def run(
        self,
        task: Optional[str] = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the task specified by ``task`` (or ``self.TASK``).

        Args:
            task: Override the task mode.  ``None`` uses ``self.TASK``.
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target results are stored
            on ``self.results[<task_key>]``.

        Raises:
            ValueError: If ``task`` is not one of ``"lazy"``,
                ``"training"``, ``"optuna"``, ``"spotoptim"``.
            RuntimeError: If method `prepare_data` has not been called.
        """
        task = task or self.TASK
        dispatch = {
            "lazy": self.run_task_lazy,
            "training": self.run_task_training,
            "optuna": self.run_task_optuna,
            "spotoptim": self.run_task_spotoptim,
        }
        if task not in dispatch:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(dispatch.keys())}"
            )
        return dispatch[task](show=show)

    # ------------------------------------------------------------------
    # Default search spaces (backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_optuna_search_space(trial: Any) -> Dict[str, Any]:
        """Built-in Optuna search space for LightGBM."""
        return OptunaTask._default_optuna_search_space(trial)

    @staticmethod
    def _default_spotoptim_search_space() -> Dict[str, Any]:
        """Built-in SpotOptim search space for LightGBM."""
        return SpotOptimTask._default_spotoptim_search_space()
