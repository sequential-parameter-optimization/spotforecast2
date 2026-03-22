# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Multi-target forecasting pipeline orchestrator.

This module provides the :class:`MultiTask` class, which encapsulates the
full multi-target time-series forecasting workflow described in
``docs/tasks/multi.qmd``.  It coordinates data preparation, outlier
detection, imputation, exogenous feature engineering, model training,
hyperparameter tuning, prediction, and aggregation for an arbitrary
number of target columns.

Public API
----------
- :class:`MultiTask` — the main pipeline orchestrator.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from astral import LocationInfo
from lightgbm import LGBMRegressor

from spotforecast2_safe.data.fetch_data import (
    fetch_data,
    get_cache_home,
    get_package_data_home,
)
from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti
from spotforecast2_safe.manager.exo.calendar import (
    get_calendar_features,
    get_day_night_features,
    get_holiday_features,
)
from spotforecast2_safe.manager.exo.weather import get_weather_features
from spotforecast2_safe.manager.features import (
    apply_cyclical_encoding,
    create_interaction_features,
    get_target_data,
    merge_data_and_covariates,
    select_exogenous_features,
)
from spotforecast2_safe.manager.persistence import save_forecaster as _save_forecaster
from spotforecast2_safe.manager.predictor import build_prediction_package
from spotforecast2_safe.preprocessing import (
    RollingFeatures as RollingFeaturesUnified,
)
from spotforecast2_safe.preprocessing.curate_data import (
    agg_and_resample_data,
    basic_ts_checks,
    get_start_end,
    reset_index,
)
from spotforecast2_safe.preprocessing.outlier import (
    get_outliers,
    manual_outlier_removal,
)
from spotforecast2_safe.processing.agg_predict import agg_predict

from spotforecast2.manager.plotter import PredictionFigure, plot_with_outliers
from spotforecast2.model_selection import (
    bayesian_search_forecaster,
    spotoptim_search_forecaster,
)
from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold
from spotforecast2.preprocessing.imputation import apply_imputation

logger = logging.getLogger(__name__)


class MultiTask:
    """Orchestrates a multi-target time-series forecasting pipeline.

    ``MultiTask`` wraps the full workflow from ``multi.qmd`` into a
    reusable Python class.  It accepts the same capitalized arguments
    that the notebook defines in its *Arguments* section (``TASK``,
    ``DATA_FRAME_NAME``, ``DATA_SOURCE``, …, ``N_INITIAL_SPOTOPTIM``)
    and exposes each pipeline stage as a method.

    The typical usage flow is:

    1. Instantiate with configuration arguments.
    2. Call :meth:`prepare_data` to load, resample, and validate data.
    3. Call :meth:`detect_outliers` to apply hard bounds and IsolationForest.
    4. Call :meth:`impute` to fill gaps.
    5. Call :meth:`build_exogenous_features` to construct weather / calendar /
       day-night / holiday covariates.
    6. Call :meth:`run` (or individual ``run_task_*`` methods) to train,
       predict, and aggregate.

    All intermediate state (``df_pipeline``, ``config``, ``exogenous_features``,
    ``results``, …) is stored as instance attributes for inspection.

    Args:
        TASK: Pipeline task mode — ``"lazy"``, ``"training"``, ``"optuna"``,
            or ``"spotoptim"``. Defaults to ``"lazy"``.
        DATA_FRAME_NAME: Active dataset identifier (e.g. ``"demo10"``).
            Defaults to ``"demo10"``.
        DATA_SOURCE: Input CSV filename. Defaults to ``"demo10.csv"``.
        DATA_TEST: Test CSV filename. Defaults to ``"demo11.csv"``.
        DATA_HOME: Path to the data directory.  ``None`` uses the package
            bundled data via ``get_package_data_home()``.
        CACHE_DATA: Whether to cache intermediate data to disk.
            Defaults to ``True``.
        CACHE_HOME: Cache directory path.  ``None`` uses the default
            ``~/spotforecast2_cache/``.
        AGG_WEIGHTS: Per-target aggregation weights — one entry per target
            in the dataset.  Positive values add, negative values invert
            the target's contribution.
        INDEX_NAME: Datetime column name in the raw CSV.
            Defaults to ``"DateTime"``.
        NUMBER_FOLDS: Number of validation folds for hyperparameter tuning.
            Defaults to ``10``.
        PREDICT_SIZE: Forecast horizon in hours. Defaults to ``24``.
        BOUNDS: Per-column hard outlier bounds ``(lower, upper)`` for
            manual removal.  ``None`` disables hard-bound filtering.
        CONTAMINATION: IsolationForest contamination fraction.
            Defaults to ``0.03``.
        IMPUTATION_METHOD: Gap-filling strategy — ``"weighted"`` or
            ``"linear"``. Defaults to ``"weighted"``.
        USE_EXOGENOUS_FEATURES: Whether to build exogenous features.
            Defaults to ``True``.
        N_TRIALS_OPTUNA: Number of Optuna Bayesian-search trials (task 3).
            Defaults to ``15``.
        N_TRIALS_SPOTOPTIM: Number of SpotOptim surrogate-search trials
            (task 4). Defaults to ``10``.
        N_INITIAL_SPOTOPTIM: Initial random evaluations for SpotOptim
            (task 4). Defaults to ``5``.
        config_overrides: Extra keyword arguments forwarded to
            :class:`~spotforecast2_safe.manager.configurator.config_multi.ConfigMulti`.
        log_level: Logging level for the pipeline logger.
            Defaults to ``logging.INFO``.

    Attributes:
        config (ConfigMulti): Centralised pipeline configuration.
        df_pipeline (pd.DataFrame): Pipeline DataFrame after preparation.
        df_test (pd.DataFrame): Test DataFrame (ground truth).
        weight_func (Optional[WeightFunction]): Sample-weight function
            from imputation (``None`` when linear interpolation is used).
        exogenous_features (Optional[pd.DataFrame]): Combined exogenous
            feature matrix.
        exog_feature_names (List[str]): Selected exogenous feature names.
        data_with_exog (Optional[pd.DataFrame]): Merged target + exogenous
            data for training.
        exo_pred (Optional[pd.DataFrame]): Exogenous covariates for the
            forecast horizon.
        results (Dict[str, Dict]): Per-task mapping of target name to
            prediction package.

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
        # Store capitalized arguments as instance attributes
        self.TASK = TASK
        self.DATA_FRAME_NAME = DATA_FRAME_NAME
        self.DATA_SOURCE = DATA_SOURCE
        self.DATA_TEST = DATA_TEST
        self.DATA_HOME = DATA_HOME if DATA_HOME is not None else get_package_data_home()
        self.CACHE_DATA = CACHE_DATA
        self.CACHE_HOME = CACHE_HOME
        self.AGG_WEIGHTS = AGG_WEIGHTS
        self.INDEX_NAME = INDEX_NAME
        self.NUMBER_FOLDS = NUMBER_FOLDS
        self.PREDICT_SIZE = PREDICT_SIZE
        self.BOUNDS = BOUNDS
        self.CONTAMINATION = CONTAMINATION
        self.IMPUTATION_METHOD = IMPUTATION_METHOD
        self.USE_EXOGENOUS_FEATURES = USE_EXOGENOUS_FEATURES
        self.N_TRIALS_OPTUNA = N_TRIALS_OPTUNA
        self.N_TRIALS_SPOTOPTIM = N_TRIALS_SPOTOPTIM
        self.N_INITIAL_SPOTOPTIM = N_INITIAL_SPOTOPTIM

        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Derived constants
        self.TRAIN_SIZE = pd.Timedelta(days=365)
        self.DELTA_VAL = pd.Timedelta(days=7 * NUMBER_FOLDS)

        # Pipeline state (populated by methods)
        self.df_pipeline: Optional[pd.DataFrame] = None
        self.df_pipeline_original: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.weight_func: Optional[Any] = None
        self.exogenous_features: Optional[pd.DataFrame] = None
        self.weather_aligned: Optional[pd.DataFrame] = None
        self.exog_feature_names: List[str] = []
        self.data_with_exog: Optional[pd.DataFrame] = None
        self.exo_pred: Optional[pd.DataFrame] = None
        self.results: Dict[str, Dict[str, Any]] = {}

        # Build ConfigMulti — merge explicit arguments with overrides
        self.config = self._build_config(**config_overrides)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _build_config(self, **overrides: Any) -> ConfigMulti:
        """Create a :class:`ConfigMulti` from the stored pipeline arguments.

        Any extra ``**overrides`` are forwarded to the
        :class:`ConfigMulti` constructor, overriding the defaults
        derived from the capitalized arguments.
        """
        kwargs: Dict[str, Any] = {
            "predict_size": self.PREDICT_SIZE,
            "contamination": self.CONTAMINATION,
            "imputation_method": self.IMPUTATION_METHOD,
            "use_exogenous_features": self.USE_EXOGENOUS_FEATURES,
            "index_name": self.INDEX_NAME,
            "data_source": self.DATA_SOURCE,
            "data_test": self.DATA_TEST,
            "data_home": self.DATA_HOME,
            "cache_home": get_cache_home(self.CACHE_HOME),
            "cache_data": self.CACHE_DATA,
            "n_trials_optuna": self.N_TRIALS_OPTUNA,
            "n_trials_spotoptim": self.N_TRIALS_SPOTOPTIM,
            "n_initial_spotoptim": self.N_INITIAL_SPOTOPTIM,
            "task": self.TASK,
            "train_size": self.TRAIN_SIZE,
            "delta_val": self.DELTA_VAL,
        }
        if self.BOUNDS is not None:
            kwargs["bounds"] = self.BOUNDS
        if self.AGG_WEIGHTS is not None:
            kwargs["agg_weights"] = self.AGG_WEIGHTS
        kwargs.update(overrides)
        return ConfigMulti(**kwargs)

    # ------------------------------------------------------------------
    # Step 1 — Data Preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        demo_data: Optional[pd.DataFrame] = None,
        df_test: Optional[pd.DataFrame] = None,
    ) -> "MultiTask":
        """Load, resample, validate, and configure the pipeline data.

        When ``demo_data`` / ``df_test`` are ``None`` the method loads
        them from ``DATA_SOURCE`` / ``DATA_TEST`` via
        :func:`~spotforecast2_safe.data.fetch_data.fetch_data`.

        After loading, the data is resampled to hourly frequency,
        validated with :func:`basic_ts_checks`, and the
        :attr:`config` date ranges are derived.

        Args:
            demo_data: Pre-loaded input DataFrame.  ``None`` triggers
                automatic loading from ``DATA_HOME / DATA_SOURCE``.
            df_test: Pre-loaded test DataFrame.  ``None`` triggers
                automatic loading from ``DATA_HOME / DATA_TEST``.

        Returns:
            ``self`` (for method chaining).
        """
        # Load data
        if demo_data is None:
            data_in_path = self.DATA_HOME / self.DATA_SOURCE
            demo_data = fetch_data(filename=str(data_in_path))
        if df_test is None:
            data_test_path = self.DATA_HOME / self.DATA_TEST
            df_test = fetch_data(filename=str(data_test_path))

        demo_data = reset_index(demo_data, index_name=self.INDEX_NAME)
        df_test = reset_index(df_test, index_name=self.INDEX_NAME)
        self.df_test = df_test

        # Derive download range and training cut-off
        first_ts = pd.Timestamp(demo_data[self.INDEX_NAME].iloc[0])
        last_ts = pd.Timestamp(demo_data[self.INDEX_NAME].iloc[-1])
        self.config.start_download = first_ts.strftime("%Y%m%d%H%M")
        self.config.end_download = last_ts.strftime("%Y%m%d%H%M")
        self.config.end_train_default = last_ts.isoformat()

        # Derive ALL_TARGETS
        all_targets = [c for c in demo_data.columns if c != self.INDEX_NAME]
        if self.config.targets is None:
            self.config.targets = all_targets

        # Set pipeline DataFrame
        df_pipeline = demo_data.set_index(self.INDEX_NAME)
        df_pipeline = agg_and_resample_data(df_pipeline, verbose=True)
        basic_ts_checks(df_pipeline, verbose=True)

        # Sync targets
        self.config.targets = [
            c for c in self.config.targets if c in df_pipeline.columns
        ]

        # Derive date ranges
        (
            self.config.data_start,
            self.config.data_end,
            self.config.cov_start,
            self.config.cov_end,
        ) = get_start_end(
            data=df_pipeline,
            forecast_horizon=self.config.predict_size,
            verbose=True,
        )

        self.df_pipeline = df_pipeline
        self.logger.info("Pipeline data shape after preparation: %s", df_pipeline.shape)
        return self

    # ------------------------------------------------------------------
    # Step 2 — Outlier Detection and Removal
    # ------------------------------------------------------------------

    def detect_outliers(self) -> "MultiTask":
        """Apply hard-bound filtering and IsolationForest outlier detection.

        Hard bounds from ``config.bounds`` are applied first via
        :func:`manual_outlier_removal`, setting out-of-bound values to
        ``NaN``.  Then, if ``config.use_outlier_detection`` is ``True``,
        :func:`mark_outliers` applies IsolationForest-based detection.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If :meth:`prepare_data` has not been called.
        """
        if self.df_pipeline is None:
            raise RuntimeError("Call prepare_data() before detect_outliers().")

        self.df_pipeline_original = self.df_pipeline.copy()

        if self.config.bounds is not None:
            for i, (lower, upper) in enumerate(self.config.bounds):
                col = self.df_pipeline.columns[i]
                self.df_pipeline, _ = manual_outlier_removal(
                    data=self.df_pipeline,
                    column=col,
                    lower_threshold=lower,
                    upper_threshold=upper,
                    verbose=True,
                )

        if self.config.use_outlier_detection:
            outliers = get_outliers(
                data=self.df_pipeline,
                contamination=self.config.contamination,
            )
            for col, outlier_vals in outliers.items():
                self.logger.info(
                    "%s: %d outliers automatically detected", col, len(outlier_vals)
                )

        self.logger.info("Outlier detection complete.")
        return self

    # ------------------------------------------------------------------
    # Step 2b — Plot outliers
    # ------------------------------------------------------------------

    def plot_with_outliers(self) -> None:
        """Visualise original vs. cleaned data with outlier markers.

        Delegates to
        :func:`~spotforecast2.manager.plotter.plot_with_outliers`.

        Raises:
            RuntimeError: If :meth:`detect_outliers` has not been called.
        """
        if self.df_pipeline_original is None:
            raise RuntimeError("Call detect_outliers() before plot_with_outliers().")

        plot_with_outliers(
            df_pipeline=self.df_pipeline,
            df_pipeline_original=self.df_pipeline_original,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Step 3 — Imputation
    # ------------------------------------------------------------------

    def impute(self) -> "MultiTask":
        """Fill missing values using the configured imputation strategy.

        Delegates to
        :func:`~spotforecast2.preprocessing.imputation.apply_imputation`
        which selects between linear interpolation and weighted gap-filling
        based on ``config.imputation_method``.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If :meth:`prepare_data` has not been called.
        """
        if self.df_pipeline is None:
            raise RuntimeError("Call prepare_data() before impute().")

        self.df_pipeline, self.weight_func = apply_imputation(
            df_pipeline=self.df_pipeline,
            config=self.config,
            logger=self.logger,
        )
        self.logger.info(
            "Imputation complete (method=%s).", self.config.imputation_method
        )
        return self

    # ------------------------------------------------------------------
    # Steps 4-7 — Exogenous Features
    # ------------------------------------------------------------------

    def build_exogenous_features(self) -> "MultiTask":
        """Build, combine, encode, and merge exogenous feature covariates.

        Executes pipeline steps 4 through 7 from ``multi.qmd``:

        * Step 4 — weather, calendar, day/night, holiday features.
        * Step 5 — concatenation, backfill, cyclical encoding, interactions.
        * Step 6 — feature selection.
        * Step 7 — merge with target data.

        When ``config.use_exogenous_features`` is ``False``, no features
        are built and the method returns immediately.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If :meth:`prepare_data` has not been called.
        """
        if self.df_pipeline is None:
            raise RuntimeError("Call prepare_data() before build_exogenous_features().")

        if not self.config.use_exogenous_features:
            self.logger.info("Exogenous features disabled — target-only pipeline.")
            return self

        self.logger.info("Building exogenous features...")

        # 4a. Weather
        weather_features, self.weather_aligned = get_weather_features(
            data=self.df_pipeline,
            start=self.config.data_start,
            cov_end=self.config.cov_end,
            forecast_horizon=self.config.predict_size,
            latitude=self.config.latitude,
            longitude=self.config.longitude,
            timezone=self.config.timezone,
            freq="h",
            verbose=True,
        )
        self.logger.info("  Weather features: %s", weather_features.shape)

        # 4b. Calendar
        calendar_features = get_calendar_features(
            start=self.config.data_start,
            cov_end=self.config.cov_end,
            freq="h",
            timezone=self.config.timezone,
        )
        self.logger.info("  Calendar features: %s", calendar_features.shape)

        # 4c. Day/night
        location = LocationInfo(
            latitude=self.config.latitude,
            longitude=self.config.longitude,
            timezone=self.config.timezone,
        )
        sun_light_features = get_day_night_features(
            start=self.config.data_start,
            cov_end=self.config.cov_end,
            location=location,
            freq="h",
            timezone=self.config.timezone,
        )
        self.logger.info("  Day/night features: %s", sun_light_features.shape)

        # 4d. Holidays
        holiday_features = get_holiday_features(
            data=self.df_pipeline,
            start=self.config.data_start,
            cov_end=self.config.cov_end,
            forecast_horizon=self.config.predict_size,
            tz=self.config.timezone,
            freq="h",
            country_code=self.config.country_code,
            state=self.config.state,
        )
        self.logger.info("  Holiday features: %s", holiday_features.shape)

        # Step 5 — Combine
        self.exogenous_features = pd.concat(
            [calendar_features, sun_light_features, weather_features, holiday_features],
            axis=1,
        )

        missing_exog = self.exogenous_features.isnull().sum().sum()
        if missing_exog != 0:
            self.logger.warning(
                "Exogenous features contain %d missing values — backfilling.",
                missing_exog,
            )
            self.exogenous_features = self.exogenous_features.bfill().ffill()

        self.exogenous_features = apply_cyclical_encoding(
            data=self.exogenous_features,
            drop_original=False,
        )
        self.exogenous_features = create_interaction_features(
            exogenous_features=self.exogenous_features,
            weather_aligned=self.weather_aligned,
        )
        self.logger.info(
            "Combined exogenous features shape: %s", self.exogenous_features.shape
        )

        # Step 6 — Select
        self.exog_feature_names = select_exogenous_features(
            exogenous_features=self.exogenous_features,
            weather_aligned=self.weather_aligned,
            include_weather_windows=self.config.include_weather_windows,
            include_holiday_features=self.config.include_holiday_features,
            include_poly_features=self.config.include_poly_features,
        )
        self.logger.info(
            "Selected %d exogenous features for training.", len(self.exog_feature_names)
        )

        # Step 7 — Merge
        self.data_with_exog, _, self.exo_pred = merge_data_and_covariates(
            data=self.df_pipeline,
            exogenous_features=self.exogenous_features,
            target_columns=self.config.targets,
            exog_features=self.exog_feature_names,
            start=self.config.data_start,
            end=self.config.data_end,
            cov_end=self.config.cov_end,
            forecast_horizon=self.config.predict_size,
            cast_dtype="float32",
        )
        self.logger.info("Merged data shape: %s", self.data_with_exog.shape)
        self.logger.info(
            "Exogenous prediction covariates shape: %s", self.exo_pred.shape
        )

        return self

    # ------------------------------------------------------------------
    # Training-window setup
    # ------------------------------------------------------------------

    def _setup_training_window(self) -> None:
        """Derive ``config.end_train_ts`` and ``config.start_train_ts``."""
        self.config.end_train_ts = pd.to_datetime(
            self.config.end_train_default, utc=True
        )
        self.config.start_train_ts = self.config.end_train_ts - self.config.train_size
        self.config.start_train_ts = max(
            self.config.start_train_ts, self.df_pipeline.index.min()
        )
        self.logger.info(
            "Training window: %s to %s",
            self.config.start_train_ts,
            self.config.end_train_ts,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def create_forecaster(self) -> Any:
        """Create a fresh :class:`ForecasterRecursive` with shared configuration.

        The forecaster is configured with:

        * A ``LGBMRegressor`` seeded with ``config.random_state``.
        * The largest lag from ``config.lags_consider``.
        * Rolling-window mean features of size ``config.window_size``.
        * The sample-weight function from :meth:`impute` (``None`` when
          linear interpolation was used).

        Returns:
            A new, unfitted ``ForecasterRecursive`` instance.

        Examples:
            ```{python}
            from spotforecast2.manager.multitask import MultiTask

            mt = MultiTask(PREDICT_SIZE=24)
            forecaster = mt.create_forecaster()
            print(f"Type: {type(forecaster).__name__}")
            print(f"Lags: {forecaster.lags}")
            ```
        """
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        return ForecasterRecursive(
            estimator=LGBMRegressor(random_state=self.config.random_state, verbose=-1),
            lags=self.config.lags_consider[-1],
            window_features=RollingFeaturesUnified(
                stats=["mean"], window_sizes=self.config.window_size
            ),
            weight_func=self.weight_func,
        )

    def _save_forecaster(self, forecaster: Any, task_name: str, target: str) -> None:
        """Save a fitted forecaster to the cache directory."""
        model_dir = get_cache_home(self.config.cache_home) / "unified_pipeline"
        path = _save_forecaster(forecaster, model_dir, target, task_name=task_name)
        self.logger.info("  Saved: %s", path)

    def _train_and_predict_target(
        self,
        target: str,
        task_name: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit, save, and build prediction package for one target."""
        forecaster.fit(y=y_train, exog=exog_train)
        self._save_forecaster(forecaster, task_name, target)
        return build_prediction_package(
            forecaster=forecaster,
            target=target,
            y_train=y_train,
            predict_size=self.config.predict_size,
            exog_train=exog_train,
            exog_future=exog_future,
            df_test=self.df_test,
        )

    def _get_target_data(self, target: str) -> tuple:
        """Extract ``(y_train, exog_train, exog_future)`` for one target."""
        return get_target_data(
            target=target,
            df_pipeline=self.df_pipeline,
            config=self.config,
            data_with_exog=self.data_with_exog,
            exog_feature_names=(
                self.exog_feature_names if self.exog_feature_names else None
            ),
            exo_pred=self.exo_pred,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _agg_predictor(
        self,
        results: Dict[str, Dict[str, Any]],
        targets: List[str],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Aggregate per-target prediction packages into a weighted forecast.

        Args:
            results: Mapping of target name to prediction package dict.
            targets: Ordered list of target names to aggregate.
            weights: Aggregation weights aligned with ``targets``.

        Returns:
            Aggregated prediction package dict.
        """
        future_preds_df = pd.DataFrame({t: results[t]["future_pred"] for t in targets})
        train_preds_df = pd.DataFrame({t: results[t]["train_pred"] for t in targets})
        train_actuals_df = pd.DataFrame(
            {t: results[t]["train_actual"] for t in targets}
        )

        agg_future_pred = agg_predict(future_preds_df, weights=weights)
        agg_train_pred = agg_predict(train_preds_df, weights=weights)
        agg_train_actual = agg_predict(train_actuals_df, weights=weights)

        # Aggregate test actuals when available
        test_series = {
            t: results[t]["test_actual"]
            for t in targets
            if "test_actual" in results[t] and len(results[t]["test_actual"]) > 0
        }
        agg_test_actual = None
        if test_series:
            test_actuals_df = pd.DataFrame(test_series).reindex(
                sorted(set().union(*[s.index for s in test_series.values()]))
            )
            test_weights = [
                weights[i] for i, t in enumerate(targets) if t in test_series
            ]
            agg_test_actual = agg_predict(test_actuals_df, weights=test_weights)

        agg_pkg: Dict[str, Any] = {
            "train_actual": agg_train_actual,
            "train_pred": agg_train_pred,
            "future_actual": pd.Series(dtype="float64"),
            "future_pred": agg_future_pred,
            "metrics_train": {},
            "metrics_future": {},
            "metrics_future_one_day": {},
            "validation_passed": True,
        }
        if agg_test_actual is not None:
            agg_pkg["test_actual"] = agg_test_actual
        return agg_pkg

    # ------------------------------------------------------------------
    # Task runners
    # ------------------------------------------------------------------

    def run_task_lazy(self, show: bool = True) -> Dict[str, Dict[str, Any]]:
        """Task 1 — Lazy Fitting with default LightGBM parameters.

        Creates an unfitted forecaster per target and delegates fitting
        to :meth:`_train_and_predict_target`.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Per-target prediction packages.

        Raises:
            RuntimeError: If :meth:`prepare_data` has not been called.
        """
        self._ensure_pipeline_ready()
        results: Dict[str, Dict[str, Any]] = {}

        for target in self.config.targets:
            self.logger.info(
                "[task 1] Target '%s': fitting with default params...", target
            )
            y_train, exog_train, exog_future = self._get_target_data(target)
            forecaster = self.create_forecaster()
            results[target] = self._train_and_predict_target(
                target=target,
                task_name="task 1: Lazy Fitting",
                forecaster=forecaster,
                y_train=y_train,
                exog_train=exog_train,
                exog_future=exog_future,
            )
            if show:
                self._show_prediction_figure(
                    results[target], target, "task 1: Lazy Fitting"
                )

        self.results["lazy"] = results
        self._aggregate_and_show(results, "task 1: Lazy Fitting", show=show)
        return results

    def run_task_training(self, show: bool = True) -> Dict[str, Dict[str, Any]]:
        """Task 2 — Explicit Pre-Training with default parameters.

        Fits each forecaster explicitly, serialises it to disk, then
        builds the prediction package from the pre-fitted model.

        Args:
            show: If ``True``, display prediction figures.

        Returns:
            Per-target prediction packages.

        Raises:
            RuntimeError: If :meth:`prepare_data` has not been called.
        """
        self._ensure_pipeline_ready()
        results: Dict[str, Dict[str, Any]] = {}

        for target in self.config.targets:
            self.logger.info("[task 2] Target '%s': explicit pre-training...", target)
            y_train, exog_train, exog_future = self._get_target_data(target)
            forecaster = self.create_forecaster()
            forecaster.fit(y=y_train, exog=exog_train)
            self._save_forecaster(forecaster, "task 2: Trained (No Tuning)", target)
            self.logger.info(
                "  [task 2] '%s': forecaster fitted and serialized.", target
            )

            pred_pkg = build_prediction_package(
                forecaster=forecaster,
                target=target,
                y_train=y_train,
                predict_size=self.config.predict_size,
                exog_train=exog_train,
                exog_future=exog_future,
                df_test=self.df_test,
            )
            results[target] = pred_pkg
            if show:
                self._show_prediction_figure(
                    pred_pkg, target, "task 2: Explicit Pre-Training"
                )

        self.results["training"] = results
        self._aggregate_and_show(results, "task 2: Explicit Pre-Training", show=show)
        return results

    def run_task_optuna(
        self,
        search_space: Optional[Callable] = None,
        show: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Task 3 — Optuna Bayesian hyperparameter tuning.

        Uses Optuna's TPE sampler to search for optimal LightGBM
        hyperparameters, then re-fits with the best discovered
        parameters.

        Args:
            search_space: Callable ``(trial) -> dict`` defining the
                Optuna search space.  ``None`` uses a built-in default.
            show: If ``True``, display prediction figures.

        Returns:
            Per-target prediction packages.

        Raises:
            RuntimeError: If :meth:`prepare_data` has not been called.
        """
        self._ensure_pipeline_ready()
        if search_space is None:
            search_space = self._default_optuna_search_space
        results: Dict[str, Dict[str, Any]] = {}

        for target in self.config.targets:
            self.logger.info(
                "[task 3] Target '%s': Optuna tuning (%d trials)...",
                target,
                self.config.n_trials_optuna,
            )
            y_train, exog_train, exog_future = self._get_target_data(target)
            forecaster = self.create_forecaster()

            end_cv = self.config.end_train_ts - self.config.delta_val
            n_train_cv = len(y_train.loc[:end_cv])
            cv = TimeSeriesFold(
                steps=self.config.predict_size,
                refit=False,
                initial_train_size=n_train_cv,
                fixed_train_size=(self.config.train_size is not None),
                gap=0,
                allow_incomplete_fold=True,
            )

            tuning_results, _ = bayesian_search_forecaster(
                forecaster=forecaster,
                y=y_train,
                cv=cv,
                search_space=search_space,
                metric="mean_absolute_error",
                exog=exog_train,
                n_trials=self.config.n_trials_optuna,
                random_state=self.config.random_state,
                return_best=False,
                verbose=False,
            )

            best_params = tuning_results.iloc[0].params
            best_lags = tuning_results.iloc[0].lags
            self.logger.info("  Best params: %s", best_params)
            self.logger.info("  Best lags: %s", best_lags)

            forecaster_tuned = self.create_forecaster()
            forecaster_tuned.set_params(**best_params)
            if hasattr(forecaster_tuned, "set_lags"):
                forecaster_tuned.set_lags(best_lags)

            results[target] = self._train_and_predict_target(
                target=target,
                task_name="task 3: Optuna Tuned",
                forecaster=forecaster_tuned,
                y_train=y_train,
                exog_train=exog_train,
                exog_future=exog_future,
            )
            if show:
                self._show_prediction_figure(
                    results[target], target, "task 3: Optuna Tuned"
                )

        self.results["optuna"] = results
        self._aggregate_and_show(results, "task 3: Optuna Tuned", show=show)
        return results

    def run_task_spotoptim(
        self,
        search_space: Optional[Dict[str, Any]] = None,
        show: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Task 4 — SpotOptim surrogate-model Bayesian tuning.

        Uses ``spotoptim`` for surrogate-model-based Bayesian
        optimisation.  Effective with small trial budgets.

        Args:
            search_space: Dictionary defining the SpotOptim search space.
                ``None`` uses a built-in default.
            show: If ``True``, display prediction figures.

        Returns:
            Per-target prediction packages.

        Raises:
            RuntimeError: If :meth:`prepare_data` has not been called.
        """
        self._ensure_pipeline_ready()
        if search_space is None:
            search_space = self._default_spotoptim_search_space()
        results: Dict[str, Dict[str, Any]] = {}

        for target in self.config.targets:
            self.logger.info(
                "[task 4] Target '%s': SpotOptim tuning (%d trials)...",
                target,
                self.config.n_trials_spotoptim,
            )
            y_train, exog_train, exog_future = self._get_target_data(target)
            forecaster = self.create_forecaster()

            end_cv = self.config.end_train_ts - self.config.delta_val
            n_train_cv = len(y_train.loc[:end_cv])
            cv = TimeSeriesFold(
                steps=self.config.predict_size,
                refit=False,
                initial_train_size=n_train_cv,
                fixed_train_size=(self.config.train_size is not None),
                gap=0,
                allow_incomplete_fold=True,
            )

            tuning_results, optimizer = spotoptim_search_forecaster(
                forecaster=forecaster,
                y=y_train,
                cv=cv,
                search_space=search_space,
                metric="mean_absolute_error",
                exog=exog_train,
                return_best=True,
                random_state=self.config.random_state,
                verbose=True,
                n_trials=self.config.n_trials_spotoptim,
                n_initial=self.config.n_initial_spotoptim,
            )

            best_params = tuning_results.iloc[0].params
            best_lags = tuning_results.iloc[0].lags
            self.logger.info("  Best params: %s", best_params)
            self.logger.info("  Best lags: %s", best_lags)

            forecaster_tuned = self.create_forecaster()
            forecaster_tuned.set_params(**best_params)
            if hasattr(forecaster_tuned, "set_lags"):
                forecaster_tuned.set_lags(best_lags)

            results[target] = self._train_and_predict_target(
                target=target,
                task_name="task 4: SpotOptim Tuned",
                forecaster=forecaster_tuned,
                y_train=y_train,
                exog_train=exog_train,
                exog_future=exog_future,
            )
            if show:
                self._show_prediction_figure(
                    results[target], target, "task 4: SpotOptim Tuned"
                )

        self.results["spotoptim"] = results
        self._aggregate_and_show(results, "task 4: SpotOptim Tuned", show=show)
        return results

    # ------------------------------------------------------------------
    # Run dispatcher
    # ------------------------------------------------------------------

    def run(
        self, task: Optional[str] = None, show: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Run the task specified by ``task`` (or ``self.TASK``).

        This is the main entry point for executing training and
        prediction after data preparation is complete.

        Args:
            task: Override the task mode.  ``None`` uses ``self.TASK``.
            show: If ``True``, display prediction figures.

        Returns:
            Per-target prediction packages for the requested task.

        Raises:
            ValueError: If ``task`` is not one of ``"lazy"``,
                ``"training"``, ``"optuna"``, ``"spotoptim"``.
            RuntimeError: If :meth:`prepare_data` has not been called.
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
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _show_prediction_figure(
        self,
        pred_pkg: Dict[str, Any],
        target: str,
        task_name: str,
    ) -> None:
        """Display a prediction figure for one target."""
        fig = PredictionFigure(
            pred_pkg,
            title=f"Prediction for Target '{target}' ({task_name})",
        ).make_plot()
        fig.show()

    def _aggregate_and_show(
        self,
        results: Dict[str, Dict[str, Any]],
        task_name: str,
        show: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate results and optionally display the combined figure."""
        if self.config.agg_weights is None:
            self.logger.info("No agg_weights configured — skipping aggregation.")
            return None

        active_weights = self.config.agg_weights[: len(self.config.targets)]
        agg_pkg = self._agg_predictor(
            results=results,
            targets=self.config.targets,
            weights=active_weights,
        )
        if show:
            fig = PredictionFigure(
                agg_pkg,
                title=(
                    f"Aggregated Forecast: Weighted Combination of "
                    f"Targets {self.config.targets} ({task_name})"
                ),
            ).make_plot()
            fig.show()
        return agg_pkg

    # ------------------------------------------------------------------
    # Default search spaces
    # ------------------------------------------------------------------

    @staticmethod
    def _default_optuna_search_space(trial: Any) -> Dict[str, Any]:
        """Built-in Optuna search space for LightGBM."""
        return {
            "num_leaves": trial.suggest_int("num_leaves", 8, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100),
            "lags": trial.suggest_categorical(
                "lags",
                [24, 48, [1, 2, 24], [1, 2, 24, 48], [1, 2, 23, 24, 47, 48]],
            ),
        }

    @staticmethod
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

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _ensure_pipeline_ready(self) -> None:
        """Raise if data has not been prepared and training window not set."""
        if self.df_pipeline is None:
            raise RuntimeError("Pipeline data not prepared. Call prepare_data() first.")
        if self.config.end_train_ts is None:
            self._setup_training_window()

    # ------------------------------------------------------------------
    # Pipeline summary
    # ------------------------------------------------------------------

    def log_summary(self) -> None:
        """Log a summary of the current pipeline configuration.

        Prints a formatted block of key pipeline settings to the
        logger at INFO level.
        """
        self.logger.info("=" * 60)
        self.logger.info("DATA PROCESSING PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(
            "  Outlier detection:   %s",
            (
                f"ON (IsolationForest, contamination={self.config.contamination})"
                if self.config.use_outlier_detection
                else "OFF"
            ),
        )
        self.logger.info("  Imputation method:   %s", self.config.imputation_method)
        self.logger.info(
            "  Exogenous features:  %s",
            (
                f"ON ({len(self.exog_feature_names)} features selected)"
                if self.config.use_exogenous_features
                else "OFF"
            ),
        )
        self.logger.info(
            "  Weight function:     %s",
            "YES" if self.weight_func is not None else "NO",
        )
        if self.data_with_exog is not None:
            self.logger.info("  Merged data shape:   %s", self.data_with_exog.shape)
            self.logger.info("  Prediction exog:     %s", self.exo_pred.shape)
        self.logger.info("=" * 60)
