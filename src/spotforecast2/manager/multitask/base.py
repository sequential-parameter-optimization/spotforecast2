# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Base class for multi-target forecasting pipeline tasks.

Provides BaseTask, which contains all shared data-preparation,
outlier-detection, imputation, exogenous-feature engineering, forecaster
creation, prediction-packaging, and aggregation logic.  Task-specific
subclasses (LazyTask, OptunaTask, SpotOptimTask) inherit from BaseTask
and implement the run method.  PredictTask also inherits from BaseTask
but skips training entirely, loading saved models instead.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from astral import LocationInfo
from lightgbm import LGBMRegressor

from spotforecast2_safe.data.fetch_data import get_cache_home
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
from joblib import dump as _joblib_dump
from joblib import load as _joblib_load
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

from sklearn.model_selection import TimeSeriesSplit as _SklearnTimeSeriesSplit

from spotforecast2.manager.plotter import PredictionFigure, plot_with_outliers
from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold
from spotforecast2_safe.preprocessing.imputation import apply_imputation
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

logger = logging.getLogger(__name__)


def agg_predictor(
    results: Dict[str, Dict[str, Any]],
    targets: List[str],
    weights: List[float],
) -> Dict[str, Any]:
    """Aggregate per-target prediction packages into a weighted forecast.

    Combines future predictions, training predictions, and training actuals
    from per-target prediction packages into an aggregated package compatible
    with ``PredictionFigure``.  This is a module-level convenience function;
    the same logic is available as BaseTask.agg_predictor.

    Args:
        results: Mapping of target name to prediction package (as returned
            by
            build_prediction_package).
        targets: Ordered list of target names to aggregate.
        weights: Per-target aggregation weights aligned with ``targets``.

    Returns:
        Aggregated prediction package with keys ``train_actual``,
        ``train_pred``, ``future_pred``, ``future_actual``,
        ``metrics_train``, ``metrics_future``, ``metrics_future_one_day``,
        ``validation_passed``, and (when present in all sources)
        ``test_actual``.
    """
    future_preds_df = pd.DataFrame({t: results[t]["future_pred"] for t in targets})
    train_preds_df = pd.DataFrame({t: results[t]["train_pred"] for t in targets})
    train_actuals_df = pd.DataFrame({t: results[t]["train_actual"] for t in targets})

    agg_future_pred = agg_predict(future_preds_df, weights=weights)
    agg_train_pred = agg_predict(train_preds_df, weights=weights)
    agg_train_actual = agg_predict(train_actuals_df, weights=weights)

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
        test_weights = [weights[i] for i, t in enumerate(targets) if t in test_series]
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


class BaseTask:
    """Shared base for all multi-target forecasting pipeline tasks.

    ``BaseTask`` encapsulates the data-preparation pipeline (steps 1–7)
    and all helper methods shared across the five task modes (lazy,
    optuna, spotoptim, predict, clean).  Subclasses implement the run method with
    task-specific training, tuning, or prediction logic.

    Args:
        dataframe:
            Pre-loaded input DataFrame with training data.
            The DataFrame must contain a datetime column matching
            ``index_name`` plus at least one numeric target column.
        data_test:
            Pre-loaded input DataFrame with test data (ground truth
            for the forecast horizon).  The DataFrame must contain a
            datetime column matching ``index_name`` plus at least one
            numeric target column.  Optional.
        data_frame_name:
            Identifier for the active dataset, used for
            cache-directory naming and model file naming.
        cache_data:
            Whether to cache intermediate data to disk. Boolean flag.
        cache_home:
            Cache directory path. String or Path.
        agg_weights:
            Per-target aggregation weights.
        index_name:
            Datetime column name in the raw CSV.
        number_folds:
            Number of validation folds for hyperparameter tuning.
        predict_size:
            Forecast horizon in hours.
        bounds:
            Per-column hard outlier bounds ``(lower, upper)``.
        contamination:
            IsolationForest contamination fraction.
        imputation_method:
            Gap-filling strategy — ``"weighted"`` or ``"linear"``.
        use_exogenous_features:
            Whether to build exogenous features.
        train_days:
            Number of days in the training window.
        val_days:
            Number of days in each validation fold.
            Note that the total validation window is ``val_days * number_folds``.
            Each fold is a contiguous block of ``val_days`` days, and folds are
            non-overlapping and sequential immediately after the training window.
        n_trials_optuna:
            Number of Optuna Bayesian-search trials.
        n_trials_spotoptim:
            Number of SpotOptim surrogate-search trials.
        n_initial_spotoptim:
            Initial random evaluations for SpotOptim.
        auto_save_models:
            Whether to automatically save fitted models to
            disk after each training run.  Defaults to ``True`` so that
            saved models are immediately available for PredictTask without
            any manual call to save_models().
        log_level:
            Logging level for the pipeline logger.
        verbose:
            Whether to print verbose messages during data preparation and outlier detection.
            Defaults to ``False``.
        config_overrides:
            Extra keyword arguments forwarded to
            ConfigMulti.

    Attributes:
        config (ConfigMulti): Centralised pipeline configuration.
        df_pipeline (pd.DataFrame): Pipeline DataFrame after preparation.
        df_test (pd.DataFrame): Test DataFrame (ground truth).
        weight_func: Sample-weight function from imputation.
        exogenous_features (pd.DataFrame): Combined exogenous feature matrix.
        exog_feature_names (List[str]): Selected exogenous feature names.
        data_with_exog (pd.DataFrame): Merged target + exogenous data.
        exo_pred (pd.DataFrame): Exogenous covariates for the forecast horizon.
        results (Dict[str, Dict]): Per-task mapping of target name to
            prediction package.
        agg_results (Dict): Mapping of task name to aggregated prediction
            package.
    """

    _task_name: str = "lazy"

    def __init__(
        self,
        *,
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
        **config_overrides: Any,
    ) -> None:
        # Task identifier (overridden by subclasses via _task_name)
        self.TASK = self._task_name

        # Store constructor arguments as instance attributes
        self._dataframe = dataframe
        self.data_frame_name = data_frame_name
        self.data_test = data_test
        self.cache_data = cache_data
        self.cache_home = cache_home
        self.agg_weights = agg_weights
        self.index_name = index_name
        self.number_folds = number_folds
        self.predict_size = predict_size
        self.bounds = bounds
        self.contamination = contamination
        self.imputation_method = imputation_method
        self.use_exogenous_features = use_exogenous_features
        self.n_trials_optuna = n_trials_optuna
        self.n_trials_spotoptim = n_trials_spotoptim
        self.n_initial_spotoptim = n_initial_spotoptim
        self.auto_save_models = auto_save_models
        self.train_days = train_days
        self.val_days = val_days
        self.verbose = verbose
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Derived constants
        self.TRAIN_SIZE = pd.Timedelta(days=self.train_days)
        self.DELTA_VAL = pd.Timedelta(days=self.val_days * self.number_folds)

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
        self.agg_results: Dict[str, Any] = {}

        # Build ConfigMulti — merge explicit arguments with overrides
        self.config = self._build_config(**config_overrides)
        self._attach_file_handler()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _attach_file_handler(self) -> None:
        """Attach a FileHandler to self.logger writing to get_cache_home()/logging/."""
        log_dir = get_cache_home(self.cache_home) / "logging"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.data_frame_name}.log"
        # Loggers are singletons — avoid adding duplicate FileHandlers
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file:
                return
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(handler)

    def _build_config(self, **overrides: Any) -> ConfigMulti:
        """Create a ConfigMulti from the stored pipeline arguments."""
        kwargs: Dict[str, Any] = {
            "predict_size": self.predict_size,
            "contamination": self.contamination,
            "imputation_method": self.imputation_method,
            "use_exogenous_features": self.use_exogenous_features,
            "index_name": self.index_name,
            "cache_home": get_cache_home(self.cache_home),
            "cache_data": self.cache_data,
            "n_trials_optuna": self.n_trials_optuna,
            "n_trials_spotoptim": self.n_trials_spotoptim,
            "n_initial_spotoptim": self.n_initial_spotoptim,
            "task": self.TASK,
            "train_size": self.TRAIN_SIZE,
            "delta_val": self.DELTA_VAL,
        }
        if self.bounds is not None:
            kwargs["bounds"] = self.bounds
        if self.agg_weights is not None:
            kwargs["agg_weights"] = self.agg_weights
        kwargs.update(overrides)
        return ConfigMulti(**kwargs)

    # ------------------------------------------------------------------
    # Step 1 — Data Preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        demo_data: Optional[pd.DataFrame] = None,
        df_test: Optional[pd.DataFrame] = None,
    ) -> "BaseTask":
        """Load, resample, validate, and configure the pipeline data.

        Uses the following precedence for the training data:

        1. ``demo_data`` argument (if provided).
        2. ``self._dataframe`` set via the constructor.

        Similarly for test data:

        1. ``df_test`` argument (if provided).
        2. ``self.data_test`` set via the constructor.

        Args:
            demo_data: Pre-loaded input DataFrame.  When ``None``, the
                constructor ``dataframe`` is used.
            df_test: Pre-loaded test DataFrame.  When ``None``, the
                constructor ``data_test`` is used.

        Returns:
            ``self`` (for method chaining).

        Raises:
            ValueError: If no data source is available (no ``demo_data``,
                no constructor ``dataframe``).

        Examples:
            ```{python}
            import pandas as pd
            from spotforecast2.manager.multitask import MultiTask
            from spotforecast2_safe.data.fetch_data import (
                fetch_data, get_package_data_home,
            )

            data_home = get_package_data_home()
            df = fetch_data(filename=str(data_home / "demo10.csv"))

            mt = MultiTask(dataframe=df, predict_size=24)
            mt.prepare_data()
            print(f"Pipeline shape: {mt.df_pipeline.shape}")
            print(f"Targets: {mt.config.targets}")
            ```
        """
        if demo_data is None:
            demo_data = self._dataframe
        if demo_data is None:
            raise ValueError(
                "No data source provided. Pass a DataFrame via the "
                "'dataframe' constructor argument or the 'demo_data' "
                "parameter of prepare_data()."
            )

        if df_test is None:
            df_test = self.data_test

        demo_data = reset_index(demo_data, index_name=self.index_name)
        if df_test is not None:
            df_test = reset_index(df_test, index_name=self.index_name)
        self.df_test = df_test

        first_ts = pd.Timestamp(demo_data[self.index_name].iloc[0])
        last_ts = pd.Timestamp(demo_data[self.index_name].iloc[-1])
        self.config.start_download = first_ts.strftime("%Y%m%d%H%M")
        self.config.end_download = last_ts.strftime("%Y%m%d%H%M")
        self.config.end_train_default = last_ts.isoformat()

        all_targets = [c for c in demo_data.columns if c != self.index_name]
        if self.config.targets is None:
            self.config.targets = all_targets

        df_pipeline = demo_data.set_index(self.index_name)
        df_pipeline = agg_and_resample_data(df_pipeline, verbose=self.verbose)
        basic_ts_checks(df_pipeline, verbose=self.verbose)

        self.config.targets = [
            c for c in self.config.targets if c in df_pipeline.columns
        ]

        (
            self.config.data_start,
            self.config.data_end,
            self.config.cov_start,
            self.config.cov_end,
        ) = get_start_end(
            data=df_pipeline,
            forecast_horizon=self.config.predict_size,
            verbose=self.verbose,
        )

        self.df_pipeline = df_pipeline
        self.logger.info("Pipeline data shape after preparation: %s", df_pipeline.shape)
        return self

    # ------------------------------------------------------------------
    # Step 2 — Outlier Detection and Removal
    # ------------------------------------------------------------------

    def detect_outliers(self) -> "BaseTask":
        """Apply hard-bound filtering and IsolationForest outlier detection.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If method `prepare_data` has not been called.
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
                    verbose=self.verbose,
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

        Raises:
            RuntimeError: If method `detect_outliers` has not been called.
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

    def impute(self) -> "BaseTask":
        """Fill missing values using the configured imputation strategy.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If method `prepare_data` has not been called.
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

    def build_exogenous_features(self) -> "BaseTask":
        """Build, combine, encode, and merge exogenous feature covariates.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If method `prepare_data` has not been called.
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
            verbose=self.verbose,
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

    def cv_ts(self, y_train: pd.Series) -> TimeSeriesFold:
        """Build a TimeSeriesFold for cross-validation.

        Constructs the cross-validation splitter used by all tuning tasks
        (OptunaTask, SpotOptimTask).

        Internally uses sklearn.model_selection.TimeSeriesSplit to
        compute split boundaries that respect temporal ordering and avoid
        data leakage between folds.  Classical cross-validation techniques
        such as ``KFold`` assume i.i.d. samples and yield unreliable
        estimates on time series data;
        sklearn.model_selection.TimeSeriesSplit instead ensures
        every test fold consists only of observations that come *after* the
        corresponding training observations.

        The validation boundary is determined by ``config.end_train_ts`` minus
        ``config.delta_val``.  When ``config.train_size`` is set, the sklearn
        splitter uses a *sliding* fixed-size training window
        (``max_train_size``); otherwise an expanding window is used so that
        each subsequent fold sees more historical data.

        Args:
            y_train: Training time series for the current target.  Used both
                to determine the validation boundary and as the sequence passed
                to sklearn.model_selection.TimeSeriesSplit.split to
                derive ``initial_train_size``.

        Returns:
            A configured ``TimeSeriesFold`` instance ready to be passed to
            a model-selection function.

        """
        end_cv = self.config.end_train_ts - self.config.delta_val
        n_train_cv = len(y_train.loc[:end_cv])

        # Fixed sliding window when a training-size limit is configured;
        # expanding window otherwise (sklearn default).
        max_train_size: Optional[int] = (
            n_train_cv if self.config.train_size is not None else None
        )

        skl_cv = _SklearnTimeSeriesSplit(
            n_splits=self.number_folds,
            max_train_size=max_train_size,
            test_size=self.config.predict_size,
            gap=0,
        )

        # The first fold's training-set size determines the initial training
        # window passed to TimeSeriesFold, ensuring both splitters agree on
        # where training ends and validation begins.
        splits = list(skl_cv.split(y_train))
        initial_train_size = len(splits[0][0]) if splits else n_train_cv

        return TimeSeriesFold(
            steps=self.config.predict_size,
            refit=False,
            initial_train_size=initial_train_size,
            fixed_train_size=(self.config.train_size is not None),
            gap=0,
            allow_incomplete_fold=True,
        )

    def create_forecaster(self) -> Any:
        """Create a fresh ForecasterRecursive with shared configuration.

        Returns:
            A new, unfitted ``ForecasterRecursive`` instance.

        Examples:
            ```{python}
            from spotforecast2.manager.multitask import LazyTask

            task = LazyTask(predict_size=24)
            forecaster = task.create_forecaster()
            print(f"Type: {type(forecaster).__name__}")
            print(f"Lags: {forecaster.lags}")
            ```
        """
        return ForecasterRecursive(
            estimator=LGBMRegressor(random_state=self.config.random_state, verbose=-1),
            lags=self.config.lags_consider[-1],
            window_features=RollingFeaturesUnified(
                stats=["mean"], window_sizes=self.config.window_size
            ),
            weight_func=self.weight_func,
        )

    # ------------------------------------------------------------------
    # Tuning-result persistence
    # ------------------------------------------------------------------

    def save_tuning_results(
        self,
        target: str,
        task_name: str,
        best_params: Dict[str, Any],
        best_lags: Any,
    ) -> Path:
        """Save tuning results (best parameters and lags) to a JSON file.

        The file is stored under ``<cache_home>/tuning_results/`` with a
        datetime-stamped filename so that loaders can determine freshness.

        Filename format::

            <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.json

        Args:
            target: Name of the forecast target column.
            task_name: Tuning algorithm identifier (e.g. ``"optuna"``,
                ``"spotoptim"``).
            best_params: Best hyperparameters discovered during tuning.
            best_lags: Best lag configuration (int, list, or nested list).

        Returns:
            Path to the saved JSON file.

        Examples:
            ```{python}
            from spotforecast2.manager.multitask import LazyTask

            task = LazyTask(data_frame_name="demo10")
            path = task.save_tuning_results(
                target="target_0",
                task_name="optuna",
                best_params={"n_estimators": 100, "learning_rate": 0.05},
                best_lags=[1, 2, 24],
            )
            print(path.name)
            ```
        """
        tuning_dir = get_cache_home(self.config.cache_home) / "tuning_results"
        tuning_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_frame_name}_{target}_{task_name}_{timestamp}.json"
        filepath = tuning_dir / filename

        # Convert lags to a JSON-safe type
        lags_serializable: Any = best_lags
        if hasattr(best_lags, "tolist"):
            lags_serializable = best_lags.tolist()

        payload = {
            "data_frame_name": self.data_frame_name,
            "target": target,
            "task_name": task_name,
            "timestamp": timestamp,
            "best_params": best_params,
            "best_lags": lags_serializable,
        }

        with open(filepath, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        self.logger.info("  Saved tuning results: %s", filepath)
        return filepath

    def load_tuning_results(
        self,
        target: str,
        task_name: Optional[str] = None,
        max_age_days: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent tuning results for a target from cache.

        Scans ``<cache_home>/tuning_results/`` for files matching the
        current ``data_frame_name`` and ``target``.  Optionally filters by
        ``task_name`` and discards results older than ``max_age_days``.

        Args:
            target: Name of the forecast target column.
            task_name: If given, only consider results from this tuning
                algorithm (e.g. ``"optuna"`` or ``"spotoptim"``).
                ``None`` accepts any algorithm.
            max_age_days: Maximum age in days.  Results older than this
                are ignored.  ``None`` accepts any age.

        Returns:
            A dictionary with keys ``best_params``, ``best_lags``,
            ``task_name``, ``target``, ``data_frame_name``, and
            ``timestamp``; or ``None`` if no matching file was found.

        Examples:
            ```{python}
            from spotforecast2.manager.multitask import LazyTask

            task = LazyTask(data_frame_name="demo10")
            # Save first so there is something to load
            task.save_tuning_results(
                target="target_0",
                task_name="optuna",
                best_params={"n_estimators": 100},
                best_lags=24,
            )
            result = task.load_tuning_results(target="target_0")
            print(result["best_params"])
            ```
        """
        tuning_dir = get_cache_home(self.config.cache_home) / "tuning_results"
        if not tuning_dir.exists():
            return None

        prefix = f"{self.data_frame_name}_{target}_"
        candidates: List[Path] = sorted(
            (
                p
                for p in tuning_dir.glob(f"{prefix}*.json")
                if task_name is None or f"_{task_name}_" in p.name
            ),
            reverse=True,
        )

        now = datetime.now(timezone.utc)
        for candidate in candidates:
            try:
                with open(candidate) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue

            if max_age_days is not None:
                ts = datetime.strptime(data["timestamp"], "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                age_days = (now - ts).total_seconds() / 86400
                if age_days > max_age_days:
                    continue

            self.logger.info("  Loaded tuning results from: %s", candidate)
            return data

        return None

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    # Maps each task key to the task_name string used in filenames.
    _TASK_MODEL_NAMES: Dict[str, str] = {
        "lazy": "task 1: Lazy Fitting",
        "optuna": "task 3: Optuna Tuned",
        "spotoptim": "task 4: SpotOptim Tuned",
    }

    def save_models(
        self,
        task_name: str,
        forecasters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Save fitted forecaster models to the cache directory.

        Each model is serialised with ``joblib`` (compress=3) into
        ``<cache_home>/models/<data_frame_name>/`` using a datetime-stamped
        filename so that multiple snapshots can coexist.

        Filename format::

            <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.joblib

        If ``forecasters`` is ``None`` the method collects fitted models
        from ``self.results[task_name]``, where each prediction package is
        expected to contain a ``"forecaster"`` key.

        Args:
            task_name: Task identifier (``"lazy"``, ``"optuna"``, or
                ``"spotoptim"``).
            forecasters: Optional mapping ``{target: fitted_forecaster}``.
                When ``None``, models are taken from the prediction
                packages stored in ``self.results``.

        Returns:
            Mapping ``{target: Path}`` of saved model file paths.

        Raises:
            ValueError: If ``task_name`` is not one of ``"lazy"``,
                ``"optuna"``, ``"spotoptim"``.
            RuntimeError: If no fitted models are available for the
                requested task.

        """
        if task_name not in self._TASK_MODEL_NAMES:
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Choose from {list(self._TASK_MODEL_NAMES)}"
            )

        # Resolve forecaster objects
        if forecasters is None:
            task_results = self.results.get(task_name)
            if not task_results:
                raise RuntimeError(
                    f"No results for task '{task_name}'. "
                    "Run the task before calling save_models()."
                )
            forecasters = {}
            for target, pkg in task_results.items():
                if "forecaster" not in pkg:
                    raise RuntimeError(
                        f"Prediction package for target '{target}' does not "
                        "contain a 'forecaster' key."
                    )
                forecasters[target] = pkg["forecaster"]

        model_dir = (
            get_cache_home(self.config.cache_home) / "models" / self.data_frame_name
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        saved: Dict[str, Path] = {}

        for target, forecaster in forecasters.items():
            filename = f"{self.data_frame_name}_{target}_{task_name}_{timestamp}.joblib"
            filepath = model_dir / filename
            _joblib_dump(forecaster, filepath, compress=3)
            saved[target] = filepath
            self.logger.info("  Saved model: %s", filepath)

        return saved

    def load_models(
        self,
        task_name: Optional[str] = None,
        target: Optional[str] = None,
        max_age_days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load the most recent fitted models from the cache directory.

        Scans ``<cache_home>/models/<data_frame_name>/`` for ``.joblib``
        files matching the current ``data_frame_name``.  Optionally
        filters by ``task_name``, ``target``, and ``max_age_days``.

        Args:
            task_name: If given, only load models from this task
                (``"lazy"``, ``"optuna"``, or ``"spotoptim"``).
                ``None`` accepts any task.
            target: If given, only load the model for this target
                column.  ``None`` loads the most recent model for
                every target found.
            max_age_days: Maximum age in days.  Models older than
                this are ignored.  ``None`` accepts any age.

        Returns:
            Mapping ``{target: forecaster}`` of loaded model objects.
            Empty dict if no matching models were found.

        """
        model_dir = (
            get_cache_home(self.config.cache_home) / "models" / self.data_frame_name
        )
        if not model_dir.exists():
            return {}

        prefix = f"{self.data_frame_name}_"
        candidates: List[Path] = sorted(
            model_dir.glob(f"{prefix}*.joblib"),
            reverse=True,
        )

        if task_name is not None:
            candidates = [p for p in candidates if f"_{task_name}_" in p.name]

        if target is not None:
            candidates = [p for p in candidates if f"_{target}_" in p.name]

        if max_age_days is not None:
            now = datetime.now(timezone.utc)
            filtered = []
            for p in candidates:
                # Extract timestamp from filename:
                # <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.joblib
                stem = p.stem  # without .joblib
                ts_str = stem[-15:]  # YYYYMMDD_HHMMSS
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                    age_days = (now - ts).total_seconds() / 86400
                    if age_days <= max_age_days:
                        filtered.append(p)
                except ValueError:
                    continue
            candidates = filtered

        # For each target, keep only the most recent file.
        # Files are already sorted newest-first.
        loaded: Dict[str, Any] = {}
        for p in candidates:
            # Parse target from filename:
            # <data_frame_name>_<target>_<task_name>_<YYYYMMDD_HHMMSS>.joblib
            stem = p.stem
            # Remove the prefix (data_frame_name + "_")
            rest = stem[len(prefix) :]
            # Remove the timestamp suffix ("_YYYYMMDD_HHMMSS")
            rest_no_ts = rest[:-16]  # strip "_YYYYMMDD_HHMMSS"
            # rest_no_ts = "<target>_<task_name>"
            # Split on known task names to extract target
            parsed_target = None
            for tname in self._TASK_MODEL_NAMES:
                suffix = f"_{tname}"
                if rest_no_ts.endswith(suffix):
                    parsed_target = rest_no_ts[: -len(suffix)]
                    break
            if parsed_target is None:
                # Unknown task — use everything before the last underscore
                # group as the target.
                parts = rest_no_ts.rsplit("_", 1)
                parsed_target = parts[0] if len(parts) > 1 else rest_no_ts

            if parsed_target in loaded:
                continue  # already have a newer file for this target

            try:
                loaded[parsed_target] = _joblib_load(p)
                self.logger.info("  Loaded model from: %s", p)
            except Exception:
                self.logger.warning("  Failed to load model: %s", p)
                continue

        return loaded

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
        pkg = build_prediction_package(
            forecaster=forecaster,
            target=target,
            y_train=y_train,
            predict_size=self.config.predict_size,
            exog_train=exog_train,
            exog_future=exog_future,
            df_test=self.df_test,
        )
        pkg["forecaster"] = forecaster
        return pkg

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

    def agg_predictor(
        self,
        results: Dict[str, Dict[str, Any]],
        targets: List[str],
        weights: List[float],
    ) -> Dict[str, Any]:
        """Aggregate per-target prediction packages into a weighted forecast.

        Delegates to the module-level agg_predictor function.
        Available as an instance method so that subclasses can override the
        aggregation strategy when needed.

        Args:
            results: Mapping of target name to prediction package (as
                returned by
                build_prediction_package).
            targets: Ordered list of target names to include.
            weights: Per-target aggregation weights aligned with ``targets``.

        Returns:
            Aggregated prediction package dict.
        """
        return agg_predictor(results, targets, weights)

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
    ) -> Dict[str, Any]:
        """Aggregate results and optionally display the combined figure.

        Aggregation is always performed.  When ``agg_weights`` is not
        configured on the task, equal weights are used as the fallback so
        that every task unconditionally concludes with an aggregated forecast.
        The result is stored in :attr:`agg_results` keyed by ``task_name``.
        """
        if self.config.agg_weights is not None:
            active_weights = self.config.agg_weights[: len(self.config.targets)]
        else:
            n = len(self.config.targets)
            active_weights = [1.0 / n] * n
            self.logger.info(
                "No agg_weights configured — using equal weights (1/%d each).", n
            )

        agg_pkg = self.agg_predictor(
            results=results,
            targets=self.config.targets,
            weights=active_weights,
        )
        self.agg_results[task_name] = agg_pkg
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
        """Log a summary of the current pipeline configuration."""
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

    # ------------------------------------------------------------------
    # Abstract run
    # ------------------------------------------------------------------

    def run(self, show: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Execute the task-specific training / tuning pipeline.

        Subclasses **must** override this method.

        Args:
            show: If ``True``, display prediction figures.
            **kwargs: Task-specific arguments (e.g. ``search_space``).

        Returns:
            Aggregated prediction package for the task.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run(). "
            "Use LazyTask, OptunaTask, SpotOptimTask, PredictTask, or CleanTask."
        )
