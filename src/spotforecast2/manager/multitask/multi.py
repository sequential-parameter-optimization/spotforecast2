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
        TASK: Pipeline task mode — ``"lazy"``, ``"training"``, ``"optuna"``,
            or ``"spotoptim"``. Defaults to ``"lazy"``.
        DATA_FRAME_NAME: Active dataset identifier (e.g. ``"demo10"``).
        DATA_SOURCE: Input CSV filename.
        DATA_TEST: Test CSV filename.
        DATA_HOME: Path to the data directory.
        CACHE_DATA: Whether to cache intermediate data to disk.
        CACHE_HOME: Cache directory path.
        AGG_WEIGHTS: Per-target aggregation weights.
        INDEX_NAME: Datetime column name in the raw CSV.
        NUMBER_FOLDS: Number of validation folds.
        PREDICT_SIZE: Forecast horizon in hours.
        BOUNDS: Per-column hard outlier bounds ``(lower, upper)``.
        CONTAMINATION: IsolationForest contamination fraction.
        IMPUTATION_METHOD: Gap-filling strategy.
        USE_EXOGENOUS_FEATURES: Whether to build exogenous features.
        N_TRIALS_OPTUNA: Number of Optuna Bayesian-search trials.
        N_TRIALS_SPOTOPTIM: Number of SpotOptim surrogate-search trials.
        N_INITIAL_SPOTOPTIM: Initial random evaluations for SpotOptim.
        log_level: Logging level for the pipeline logger.
        config_overrides: Extra keyword arguments forwarded to
            class `~spotforecast2_safe.manager.configurator.config_multi.ConfigMulti`.

    Examples:
        ```{python}
        from spotforecast2.manager.multitask import MultiTask

        mt = MultiTask(
            TASK="lazy",
            DATA_FRAME_NAME="demo10",
            PREDICT_SIZE=24,
            N_TRIALS_OPTUNA=5,
        )
        print(f"Task: {mt.TASK}")
        print(f"Predict size: {mt.config.predict_size}")
        print(f"Data source: {mt.DATA_SOURCE}")
        ```

        ```{python}
        from spotforecast2.manager.multitask import MultiTask

        mt = MultiTask(
            TASK="training",
            AGG_WEIGHTS=[1.0, -1.0, 0.5],
            CONTAMINATION=0.05,
        )
        print(f"Agg weights: {mt.config.agg_weights}")
        print(f"Contamination: {mt.config.contamination}")
        print(f"Imputation: {mt.IMPUTATION_METHOD}")
        ```
    """

    def __init__(
        self,
        *,
        TASK: str = "lazy",
        DATA_FRAME_NAME: str = "demo10",
        DATA_SOURCE: str = "demo10.csv",
        DATA_TEST: str = "demo11.csv",
        DATA_HOME: Optional[Path] = None,
        CACHE_DATA: bool = True,
        CACHE_HOME: Optional[Path] = None,
        AGG_WEIGHTS: Optional[List[float]] = None,
        INDEX_NAME: str = "DateTime",
        NUMBER_FOLDS: int = 10,
        PREDICT_SIZE: int = 24,
        BOUNDS: Optional[List[tuple]] = None,
        CONTAMINATION: float = 0.03,
        IMPUTATION_METHOD: str = "weighted",
        USE_EXOGENOUS_FEATURES: bool = True,
        N_TRIALS_OPTUNA: int = 15,
        N_TRIALS_SPOTOPTIM: int = 10,
        N_INITIAL_SPOTOPTIM: int = 5,
        log_level: int = logging.INFO,
        **config_overrides: Any,
    ) -> None:
        # Set _task_name before super().__init__ so self.TASK is correct
        self._task_name = TASK
        super().__init__(
            DATA_FRAME_NAME=DATA_FRAME_NAME,
            DATA_SOURCE=DATA_SOURCE,
            DATA_TEST=DATA_TEST,
            DATA_HOME=DATA_HOME,
            CACHE_DATA=CACHE_DATA,
            CACHE_HOME=CACHE_HOME,
            AGG_WEIGHTS=AGG_WEIGHTS,
            INDEX_NAME=INDEX_NAME,
            NUMBER_FOLDS=NUMBER_FOLDS,
            PREDICT_SIZE=PREDICT_SIZE,
            BOUNDS=BOUNDS,
            CONTAMINATION=CONTAMINATION,
            IMPUTATION_METHOD=IMPUTATION_METHOD,
            USE_EXOGENOUS_FEATURES=USE_EXOGENOUS_FEATURES,
            N_TRIALS_OPTUNA=N_TRIALS_OPTUNA,
            N_TRIALS_SPOTOPTIM=N_TRIALS_SPOTOPTIM,
            N_INITIAL_SPOTOPTIM=N_INITIAL_SPOTOPTIM,
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
