# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for MultiTask in manager.multitask.

Covers:
- Import paths (direct and via manager package)
- Constructor defaults and custom arguments
- ConfigMulti delegation (capitalized args → config attributes)
- Pipeline guard methods (RuntimeError when steps called out of order)
- create_forecaster() factory
- Logger configuration
- Default search spaces
- prepare_data() with package demo data
- detect_outliers() execution
- impute() execution
- build_exogenous_features() execution
- run() dispatcher and run_task_lazy() integration
"""

import logging

import pandas as pd
import pytest

from spotforecast2.manager.multitask import MultiTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default() -> MultiTask:
    """Return a MultiTask with default parameters."""
    return MultiTask()


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


class TestMultiTaskImport:
    """Verify MultiTask is importable from expected paths."""

    def test_import_from_multitask(self):
        from spotforecast2.manager.multitask import MultiTask as MT

        assert MT is MultiTask

    def test_import_from_manager_package(self):
        from spotforecast2.manager import MultiTask as MT

        assert MT is MultiTask

    def test_is_class(self):
        assert isinstance(MultiTask, type)


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestMultiTaskDefaults:
    """Verify default argument values."""

    def test_task_default(self):
        assert _default().TASK == "lazy"

    def test_data_frame_name_default(self):
        assert _default().DATA_FRAME_NAME == "demo10"

    def test_data_source_default(self):
        assert _default().DATA_SOURCE == "demo10.csv"

    def test_data_test_default(self):
        assert _default().DATA_TEST == "demo11.csv"

    def test_cache_data_default(self):
        assert _default().CACHE_DATA is True

    def test_cache_home_default(self):
        assert _default().CACHE_HOME is None

    def test_agg_weights_default(self):
        assert _default().AGG_WEIGHTS is None

    def test_index_name_default(self):
        assert _default().INDEX_NAME == "DateTime"

    def test_number_folds_default(self):
        assert _default().NUMBER_FOLDS == 10

    def test_predict_size_default(self):
        assert _default().PREDICT_SIZE == 24

    def test_bounds_default(self):
        assert _default().BOUNDS is None

    def test_contamination_default(self):
        assert _default().CONTAMINATION == 0.03

    def test_imputation_method_default(self):
        assert _default().IMPUTATION_METHOD == "weighted"

    def test_use_exogenous_features_default(self):
        assert _default().USE_EXOGENOUS_FEATURES is True

    def test_n_trials_optuna_default(self):
        assert _default().N_TRIALS_OPTUNA == 15

    def test_n_trials_spotoptim_default(self):
        assert _default().N_TRIALS_SPOTOPTIM == 10

    def test_n_initial_spotoptim_default(self):
        assert _default().N_INITIAL_SPOTOPTIM == 5

    def test_train_size_default(self):
        assert _default().TRAIN_SIZE == pd.Timedelta(days=365)

    def test_delta_val_default(self):
        assert _default().DELTA_VAL == pd.Timedelta(days=70)


# ---------------------------------------------------------------------------
# Custom arguments
# ---------------------------------------------------------------------------


class TestMultiTaskCustomArgs:
    """Verify custom constructor arguments are stored correctly."""

    def test_custom_task(self):
        mt = MultiTask(TASK="training")
        assert mt.TASK == "training"

    def test_custom_predict_size(self):
        mt = MultiTask(PREDICT_SIZE=48)
        assert mt.PREDICT_SIZE == 48

    def test_custom_contamination(self):
        mt = MultiTask(CONTAMINATION=0.05)
        assert mt.CONTAMINATION == 0.05

    def test_custom_agg_weights(self):
        weights = [1.0, -1.0, 0.5]
        mt = MultiTask(AGG_WEIGHTS=weights)
        assert mt.AGG_WEIGHTS == weights

    def test_custom_n_trials_optuna(self):
        mt = MultiTask(N_TRIALS_OPTUNA=5)
        assert mt.N_TRIALS_OPTUNA == 5

    def test_custom_number_folds(self):
        mt = MultiTask(NUMBER_FOLDS=5)
        assert mt.NUMBER_FOLDS == 5
        assert mt.DELTA_VAL == pd.Timedelta(days=35)

    def test_custom_imputation_method(self):
        mt = MultiTask(IMPUTATION_METHOD="linear")
        assert mt.IMPUTATION_METHOD == "linear"


# ---------------------------------------------------------------------------
# ConfigMulti delegation
# ---------------------------------------------------------------------------


class TestMultiTaskConfig:
    """Verify capitalized args are correctly forwarded to ConfigMulti."""

    def test_config_is_config_multi(self):
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        assert isinstance(_default().config, ConfigMulti)

    def test_config_predict_size(self):
        mt = MultiTask(PREDICT_SIZE=48)
        assert mt.config.predict_size == 48

    def test_config_contamination(self):
        mt = MultiTask(CONTAMINATION=0.05)
        assert mt.config.contamination == 0.05

    def test_config_imputation_method(self):
        mt = MultiTask(IMPUTATION_METHOD="linear")
        assert mt.config.imputation_method == "linear"

    def test_config_use_exogenous_features(self):
        mt = MultiTask(USE_EXOGENOUS_FEATURES=False)
        assert mt.config.use_exogenous_features is False

    def test_config_task(self):
        mt = MultiTask(TASK="optuna")
        assert mt.config.task == "optuna"

    def test_config_n_trials_optuna(self):
        mt = MultiTask(N_TRIALS_OPTUNA=3)
        assert mt.config.n_trials_optuna == 3

    def test_config_n_trials_spotoptim(self):
        mt = MultiTask(N_TRIALS_SPOTOPTIM=7)
        assert mt.config.n_trials_spotoptim == 7

    def test_config_n_initial_spotoptim(self):
        mt = MultiTask(N_INITIAL_SPOTOPTIM=3)
        assert mt.config.n_initial_spotoptim == 3

    def test_config_agg_weights(self):
        weights = [1.0, -1.0]
        mt = MultiTask(AGG_WEIGHTS=weights)
        assert mt.config.agg_weights == weights

    def test_config_bounds(self):
        bounds = [(0, 100), (-50, 200)]
        mt = MultiTask(BOUNDS=bounds)
        assert mt.config.bounds == bounds

    def test_config_index_name(self):
        mt = MultiTask(INDEX_NAME="ts")
        assert mt.config.index_name == "ts"

    def test_config_data_source(self):
        mt = MultiTask(DATA_SOURCE="my_data.csv")
        assert mt.config.data_source == "my_data.csv"

    def test_config_overrides(self):
        """Extra kwargs go to ConfigMulti constructor."""
        mt = MultiTask(country_code="FR", random_state=42)
        assert mt.config.country_code == "FR"
        assert mt.config.random_state == 42


# ---------------------------------------------------------------------------
# Pipeline state initialisation
# ---------------------------------------------------------------------------


class TestMultiTaskPipelineState:
    """Verify initial pipeline state attributes."""

    def test_df_pipeline_initially_none(self):
        assert _default().df_pipeline is None

    def test_df_test_initially_none(self):
        assert _default().df_test is None

    def test_weight_func_initially_none(self):
        assert _default().weight_func is None

    def test_exogenous_features_initially_none(self):
        assert _default().exogenous_features is None

    def test_exog_feature_names_initially_empty(self):
        assert _default().exog_feature_names == []

    def test_data_with_exog_initially_none(self):
        assert _default().data_with_exog is None

    def test_exo_pred_initially_none(self):
        assert _default().exo_pred is None

    def test_results_initially_empty(self):
        assert _default().results == {}


# ---------------------------------------------------------------------------
# Pipeline guard methods
# ---------------------------------------------------------------------------


class TestMultiTaskGuards:
    """Verify RuntimeError raised when steps called out of order."""

    def test_detect_outliers_before_prepare(self):
        with pytest.raises(RuntimeError, match="prepare_data"):
            _default().detect_outliers()

    def test_impute_before_prepare(self):
        with pytest.raises(RuntimeError, match="prepare_data"):
            _default().impute()

    def test_build_exogenous_before_prepare(self):
        with pytest.raises(RuntimeError, match="prepare_data"):
            _default().build_exogenous_features()

    def test_plot_with_outliers_before_detect(self):
        with pytest.raises(RuntimeError, match="detect_outliers"):
            _default().plot_with_outliers()

    def test_run_before_prepare(self):
        with pytest.raises(RuntimeError, match="Pipeline data not prepared"):
            _default().run(show=False)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TestMultiTaskLogger:
    """Verify logger configuration."""

    def test_logger_is_logger_instance(self):
        assert isinstance(_default().logger, logging.Logger)

    def test_logger_name_includes_class(self):
        assert "MultiTask" in _default().logger.name

    def test_custom_log_level(self):
        mt = MultiTask(log_level=logging.DEBUG)
        assert mt.logger.level == logging.DEBUG

    def test_default_log_level(self):
        mt = MultiTask()
        assert mt.logger.level == logging.INFO


# ---------------------------------------------------------------------------
# create_forecaster()
# ---------------------------------------------------------------------------


class TestMultiTaskCreateForecaster:
    """Verify the forecaster factory method."""

    def test_returns_forecaster_recursive(self):
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        mt = MultiTask()
        forecaster = mt.create_forecaster()
        assert isinstance(forecaster, ForecasterRecursive)

    def test_lags_from_config(self):
        mt = MultiTask(lags_consider=[1, 2, 48])
        forecaster = mt.create_forecaster()
        # lags is a numpy array; the largest lag should be 48
        assert max(forecaster.lags) == 48

    def test_weight_func_none_initially(self):
        mt = MultiTask()
        forecaster = mt.create_forecaster()
        assert forecaster.weight_func is None


# ---------------------------------------------------------------------------
# run() dispatcher
# ---------------------------------------------------------------------------


class TestMultiTaskRunDispatcher:
    """Verify the run() method dispatches to the correct task."""

    def test_invalid_task_raises(self):
        mt = MultiTask(TASK="invalid_task")
        mt.df_pipeline = pd.DataFrame()  # bypass guard
        mt.config.end_train_ts = pd.Timestamp("2024-01-01", tz="UTC")
        with pytest.raises(ValueError, match="Unknown task"):
            mt.run(show=False)

    def test_valid_task_names(self):
        """Ensure all valid task names are recognised."""
        for task in ["lazy", "training", "optuna", "spotoptim"]:
            mt = MultiTask(TASK=task)
            assert mt.TASK == task


# ---------------------------------------------------------------------------
# Default search spaces
# ---------------------------------------------------------------------------


class TestMultiTaskSearchSpaces:
    """Verify the default search space methods."""

    def test_spotoptim_search_space_is_dict(self):
        space = MultiTask._default_spotoptim_search_space()
        assert isinstance(space, dict)

    def test_spotoptim_search_space_keys(self):
        space = MultiTask._default_spotoptim_search_space()
        expected = {
            "num_leaves",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "bagging_fraction",
            "feature_fraction",
            "reg_alpha",
            "reg_lambda",
            "lags",
        }
        assert expected == set(space.keys())

    def test_spotoptim_lags_is_list(self):
        space = MultiTask._default_spotoptim_search_space()
        assert isinstance(space["lags"], list)


# ---------------------------------------------------------------------------
# Integration: prepare_data with package demo data
# ---------------------------------------------------------------------------


class TestMultiTaskPrepareData:
    """Integration test: load demo10 and run prepare_data."""

    def test_prepare_data_loads_data(self):
        mt = MultiTask(DATA_FRAME_NAME="demo10")
        mt.prepare_data()
        assert mt.df_pipeline is not None
        assert mt.df_test is not None
        assert isinstance(mt.df_pipeline, pd.DataFrame)
        assert mt.df_pipeline.shape[0] > 0

    def test_prepare_data_sets_targets(self):
        mt = MultiTask(DATA_FRAME_NAME="demo10")
        mt.prepare_data()
        assert mt.config.targets is not None
        assert len(mt.config.targets) > 0

    def test_prepare_data_sets_date_ranges(self):
        mt = MultiTask(DATA_FRAME_NAME="demo10")
        mt.prepare_data()
        assert mt.config.data_start is not None
        assert mt.config.data_end is not None
        assert mt.config.cov_start is not None
        assert mt.config.cov_end is not None

    def test_prepare_data_returns_self(self):
        mt = MultiTask()
        result = mt.prepare_data()
        assert result is mt


# ---------------------------------------------------------------------------
# Integration: detect_outliers
# ---------------------------------------------------------------------------


class TestMultiTaskDetectOutliers:
    """Integration test: outlier detection after prepare_data."""

    def test_detect_outliers_runs(self):
        mt = MultiTask()
        mt.prepare_data()
        mt.detect_outliers()
        assert mt.df_pipeline_original is not None

    def test_detect_outliers_returns_self(self):
        mt = MultiTask()
        mt.prepare_data()
        result = mt.detect_outliers()
        assert result is mt

    def test_detect_outliers_preserves_original(self):
        mt = MultiTask()
        mt.prepare_data()
        shape_before = mt.df_pipeline.shape
        mt.detect_outliers()
        assert mt.df_pipeline_original.shape == shape_before


# ---------------------------------------------------------------------------
# Integration: impute
# ---------------------------------------------------------------------------


class TestMultiTaskImpute:
    """Integration test: imputation after prepare_data."""

    def test_impute_runs(self):
        mt = MultiTask()
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        assert mt.df_pipeline.notna().all().all()

    def test_impute_returns_self(self):
        mt = MultiTask()
        mt.prepare_data()
        mt.detect_outliers()
        result = mt.impute()
        assert result is mt


# ---------------------------------------------------------------------------
# Integration: build_exogenous_features
# ---------------------------------------------------------------------------


class TestMultiTaskExogenousFeatures:
    """Integration test: exogenous feature engineering."""

    def test_exog_disabled(self):
        mt = MultiTask(USE_EXOGENOUS_FEATURES=False)
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        mt.build_exogenous_features()
        assert mt.exogenous_features is None
        assert mt.exog_feature_names == []

    def test_exog_enabled(self):
        mt = MultiTask(USE_EXOGENOUS_FEATURES=True)
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        mt.build_exogenous_features()
        assert mt.exogenous_features is not None
        assert len(mt.exog_feature_names) > 0
        assert mt.data_with_exog is not None
        assert mt.exo_pred is not None

    def test_build_exog_returns_self(self):
        mt = MultiTask(USE_EXOGENOUS_FEATURES=False)
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        result = mt.build_exogenous_features()
        assert result is mt


# ---------------------------------------------------------------------------
# Integration: log_summary
# ---------------------------------------------------------------------------


class TestMultiTaskLogSummary:
    """Verify log_summary does not raise."""

    def test_log_summary_no_exog(self):
        mt = MultiTask(USE_EXOGENOUS_FEATURES=False)
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        mt.build_exogenous_features()
        mt.log_summary()  # should not raise

    def test_log_summary_with_exog(self):
        mt = MultiTask(USE_EXOGENOUS_FEATURES=True)
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        mt.build_exogenous_features()
        mt.log_summary()  # should not raise
