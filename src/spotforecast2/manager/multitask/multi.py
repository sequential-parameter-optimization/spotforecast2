# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backward-compatible MultiTask dispatcher.

MultiTask preserves the original API where a single ``task``
parameter selects which pipeline mode to run.  It inherits from
BaseTask and delegates run() to the appropriate task-specific function.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from spotforecast2.manager.multitask.base import BaseTask
from spotforecast2.manager.multitask.clean import execute_clean
from spotforecast2.manager.multitask.lazy import execute_lazy
from spotforecast2.manager.multitask.optuna import (
    OptunaTask,
    execute_optuna,
)
from spotforecast2.manager.multitask.predict import execute_predict
from spotforecast2.manager.multitask.spotoptim import (
    SpotOptimTask,
    execute_spotoptim,
)


class MultiTask(BaseTask):
    """Orchestrates a multi-target time-series forecasting pipeline.

    Data must be provided either as a pandas DataFrame via ``dataframe``.
    A test dataset can optionally be provided via ``data_test``.

    The typical usage flow is:

    1. Instantiate with configuration arguments.
    2. Call method ``prepare_data`` to load, resample, and validate data.
    3. Call method ``detect_outliers`` to apply hard bounds and IsolationForest.
    4. Call method ``impute`` to fill gaps.
    5. Call method ``build_exogenous_features`` to construct weather / calendar /
       day-night / holiday covariates.
    6. Call method ``run`` (or individual ``run_task_*`` methods) to train,
       predict, and aggregate.

    Args:
        task: Pipeline task mode — ``"lazy"``, ``"optuna"``,
            ``"spotoptim"``, ``"predict"``, or ``"clean"``.
            Defaults to ``"lazy"``.
        dataframe: Pre-loaded input DataFrame with Train data. The DataFrame must contain a
            datetime column matching ``index_name`` plus at least one
            numeric target column. Optional for the "clean" task, but required for all other tasks.
        data_test: Pre-loaded input DataFrame with Test data. The DataFrame must contain a
            datetime column matching ``index_name`` plus at least one
            numeric target column. Optional.
        cache_data: Whether to cache intermediate data to disk.
        cache_home: Cache directory path.
        agg_weights: Per-target aggregation weights.
        index_name: Datetime column name in the raw CSV / DataFrame.
        number_folds: Number of validation folds.
        predict_size: Forecast horizon in hours.
        bounds: Per-column hard outlier bounds ``(lower, upper)``.
        contamination: IsolationForest contamination fraction.
        imputation_method: Gap-filling strategy.
        use_exogenous_features: Whether to build exogenous features.
        n_trials_optuna: Number of Optuna Bayesian-search trials.
        n_trials_spotoptim: Number of SpotOptim surrogate-search trials.
        n_initial_spotoptim: Initial random evaluations for SpotOptim.
        auto_save_models: Whether to automatically save fitted models to
            disk after each training run.  Defaults to ``True`` so that
            saved models are immediately available for the predict task
            without any manual call to ``save_models``.
        train_days: Length of the training window in days.  Controls
            ``TRAIN_SIZE`` and ``config.train_size``.  Defaults to
            ``365 * 2`` (two years).
        val_days: Length of each validation fold in days.  The total
            validation span is ``val_days * number_folds``.  Controls
            ``DELTA_VAL`` and ``config.delta_val``.  Defaults to
            ``7 * 10`` (ten weeks).
        log_level: Logging level for the pipeline logger.
        dry_run: If ``True``, do not clean cache or save models.  Useful for testing and debugging.
        config_overrides: Extra keyword arguments forwarded to
            ConfigMulti.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2.manager.multitask import MultiTask
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

        data_home = get_package_data_home()
        df = fetch_data(filename=str(data_home / "demo10.csv"))

        mt = MultiTask(dataframe=df, predict_size=24)
        print(f"DataFrame stored: {mt._dataframe is not None}")
        print(f"Task: {mt.TASK}")
        ```

    """

    def __init__(
        self,
        *,
        task: str = "lazy",
        dataframe: Optional[pd.DataFrame] = None,
        data_test: Optional[pd.DataFrame] = None,
        data_frame_name: str = "default",
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
        auto_save_models: bool = True,
        train_days: int = 365 * 2,
        val_days: int = 7 * 2,
        log_level: int = logging.INFO,
        verbose: bool = False,
        dry_run: bool = False,
        show_progress: bool = False,
        **config_overrides: Any,
    ) -> None:
        # Set _task_name before super().__init__ so self.TASK is correct
        self._task_name = task
        self._dry_run = dry_run
        self._show_progress = show_progress
        super().__init__(
            dataframe=dataframe,
            data_test=data_test,
            data_frame_name=data_frame_name,
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
            auto_save_models=auto_save_models,
            train_days=train_days,
            val_days=val_days,
            log_level=log_level,
            verbose=verbose,
            **config_overrides,
        )

    # ------------------------------------------------------------------
    # Task-specific convenience methods
    # ------------------------------------------------------------------

    def run_task_lazy(self, show: bool = True) -> Dict[str, Any]:
        """Lazy Fitting with default LightGBM parameters.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["lazy"]``.
        """
        return execute_lazy(self, show=show)

    def run_task_optuna(
        self,
        search_space: Optional[Callable] = None,
        show: bool = True,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """Optuna Bayesian hyperparameter tuning.

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
        """SpotOptim surrogate-model Bayesian tuning.

        Args:
            search_space: Dictionary defining the SpotOptim search space.
            show: If ``True``, display prediction figures.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["spotoptim"]``.
        """
        return execute_spotoptim(self, show=show, search_space=search_space)

    def run_task_predict(
        self,
        show: bool = True,
        task_name: Optional[str] = None,
        max_age_days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict-only using previously saved models.

        Loads fitted models from the cache directory and produces
        predictions without any training.  Raises ``RuntimeError``
        if no saved models are found.

        Args:
            show: If ``True``, display prediction figures.
            task_name: Restrict model loading to a specific source task
                (``"lazy"``, ``"optuna"``, or ``"spotoptim"``).
                ``None`` loads the most recent model regardless of source.
            max_age_days: Maximum age in days for saved models.
                ``None`` accepts any age.

        Returns:
            Aggregated prediction package. Per-target results in
            ``self.results["predict"]``.

        Raises:
            RuntimeError: If no saved models are found.
        """
        return execute_predict(
            self, show=show, task_name=task_name, max_age_days=max_age_days
        )

    def run_task_clean(
        self,
        show: bool = True,
        dry_run: bool = False,
        cache_home: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Remove all cached data from the pipeline cache directory.

        Does not require prepare_data() to be called first.

        Args:
            show: Accepted for API consistency.  Not used by the clean task.
            dry_run: If ``True``, report what would be deleted without
                actually removing anything.
            cache_home: Override the directory to clean.  ``None`` uses
                the cache directory configured on this instance.

        Returns:
            Dict with keys status, cache_dir, and deleted_items.

        Raises:
            RuntimeError: If the cache directory cannot be removed.
        """
        return execute_clean(self, cache_home=cache_home, dry_run=dry_run)

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
                ``"optuna"``, ``"spotoptim"``, ``"predict"``, ``"clean"``.
            RuntimeError: If method `prepare_data` has not been called
                (for training and prediction tasks).
        """
        task = task or self.TASK
        dispatch = {
            "lazy": self.run_task_lazy,
            "optuna": self.run_task_optuna,
            "spotoptim": self.run_task_spotoptim,
            "predict": self.run_task_predict,
        }
        if task not in {*dispatch, "clean"}:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {sorted({*dispatch, 'clean'})}"
            )
        if task == "clean":
            return self.run_task_clean(show=show, dry_run=self._dry_run)
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
