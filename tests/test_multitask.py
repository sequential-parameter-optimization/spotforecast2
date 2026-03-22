# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for the multitask package class hierarchy.

Covers:
- Import paths (direct, via manager package, and via multitask package)
- BaseTask shared logic (constructor, config, pipeline state, guards, logger)
- LazyTask, OptunaTask, SpotOptimTask subclass behaviour
- MultiTask backward-compatible dispatcher
- Inheritance hierarchy
- Pipeline methods (prepare_data, detect_outliers, impute, build_exogenous_features)
- create_forecaster() factory
- Default search spaces
- run() dispatcher and run_task_*() convenience methods
"""

import logging

import pandas as pd
import pytest

from spotforecast2.manager.multitask import (
    BaseTask,
    LazyTask,
    MultiTask,
    OptunaTask,
    SpotOptimTask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default() -> MultiTask:
    """Return a MultiTask with default parameters."""
    return MultiTask()


def _lazy() -> LazyTask:
    """Return a LazyTask with default parameters."""
    return LazyTask()


def _optuna() -> OptunaTask:
    """Return an OptunaTask with default parameters."""
    return OptunaTask()


def _spotoptim() -> SpotOptimTask:
    """Return a SpotOptimTask with default parameters."""
    return SpotOptimTask()


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


class TestImportPaths:
    """Verify all classes are importable from expected paths."""

    def test_import_multitask_from_multitask_package(self):
        from spotforecast2.manager.multitask import MultiTask as MT

        assert MT is MultiTask

    def test_import_multitask_from_manager_package(self):
        from spotforecast2.manager import MultiTask as MT

        assert MT is MultiTask

    def test_import_base_task(self):
        from spotforecast2.manager.multitask import BaseTask as BT

        assert BT is BaseTask

    def test_import_lazy_task(self):
        from spotforecast2.manager.multitask import LazyTask as LT

        assert LT is LazyTask

    def test_import_optuna_task(self):
        from spotforecast2.manager.multitask import OptunaTask as OT

        assert OT is OptunaTask

    def test_import_spotoptim_task(self):
        from spotforecast2.manager.multitask import SpotOptimTask as ST

        assert ST is SpotOptimTask

    def test_import_from_manager_all_classes(self):
        from spotforecast2.manager import (
            BaseTask as BT,
            LazyTask as LT,
            OptunaTask as OT,
            SpotOptimTask as ST,
        )

        assert BT is BaseTask
        assert LT is LazyTask
        assert OT is OptunaTask
        assert ST is SpotOptimTask

    def test_all_are_classes(self):
        for cls in [
            BaseTask,
            LazyTask,
            OptunaTask,
            SpotOptimTask,
            MultiTask,
        ]:
            assert isinstance(cls, type)


# ---------------------------------------------------------------------------
# Inheritance hierarchy
# ---------------------------------------------------------------------------


class TestInheritance:
    """Verify the class hierarchy."""

    def test_lazy_task_inherits_base(self):
        assert issubclass(LazyTask, BaseTask)

    def test_optuna_task_inherits_base(self):
        assert issubclass(OptunaTask, BaseTask)

    def test_spotoptim_task_inherits_base(self):
        assert issubclass(SpotOptimTask, BaseTask)

    def test_multitask_inherits_base(self):
        assert issubclass(MultiTask, BaseTask)

    def test_lazy_instance_is_base(self):
        assert isinstance(_lazy(), BaseTask)

    def test_optuna_instance_is_base(self):
        assert isinstance(_optuna(), BaseTask)

    def test_spotoptim_instance_is_base(self):
        assert isinstance(_spotoptim(), BaseTask)

    def test_multitask_instance_is_base(self):
        assert isinstance(_default(), BaseTask)

    def test_multitask_is_class(self):
        assert isinstance(MultiTask, type)


# ---------------------------------------------------------------------------
# Task name (_task_name class variable)
# ---------------------------------------------------------------------------


class TestTaskName:
    """Verify each class reports the correct task name."""

    def test_lazy_task_name(self):
        assert _lazy().TASK == "lazy"

    def test_optuna_task_name(self):
        assert _optuna().TASK == "optuna"

    def test_spotoptim_task_name(self):
        assert _spotoptim().TASK == "spotoptim"

    def test_multitask_default_task(self):
        assert _default().TASK == "lazy"

    def test_multitask_custom_task(self):
        mt = MultiTask(task="optuna")
        assert mt.TASK == "optuna"

    def test_multitask_optuna_task(self):
        mt = MultiTask(task="optuna")
        assert mt.TASK == "optuna"

    def test_multitask_spotoptim_task(self):
        mt = MultiTask(task="spotoptim")
        assert mt.TASK == "spotoptim"


# ---------------------------------------------------------------------------
# Constructor defaults (shared across all subclasses)
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify default argument values on all task classes."""

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_data_frame_name_default(self, factory):
        assert factory().data_frame_name == "demo10"

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_data_source_default(self, factory):
        assert factory().data_source == "demo10.csv"

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_data_test_default(self, factory):
        assert factory().data_test == "demo11.csv"

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_cache_data_default(self, factory):
        assert factory().cache_data is True

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_predict_size_default(self, factory):
        assert factory().predict_size == 24

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_contamination_default(self, factory):
        assert factory().contamination == 0.03

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_imputation_method_default(self, factory):
        assert factory().imputation_method == "weighted"

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_number_folds_default(self, factory):
        assert factory().number_folds == 10

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_train_size_default(self, factory):
        assert factory().TRAIN_SIZE == pd.Timedelta(days=365)

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_delta_val_default(self, factory):
        assert factory().DELTA_VAL == pd.Timedelta(days=70)

    def test_cache_home_default(self):
        assert _default().cache_home is None

    def test_agg_weights_default(self):
        assert _default().agg_weights is None

    def test_index_name_default(self):
        assert _default().index_name == "DateTime"

    def test_bounds_default(self):
        assert _default().bounds is None

    def test_use_exogenous_features_default(self):
        assert _default().use_exogenous_features is True

    def test_n_trials_optuna_default(self):
        assert _default().n_trials_optuna == 15

    def test_n_trials_spotoptim_default(self):
        assert _default().n_trials_spotoptim == 10

    def test_n_initial_spotoptim_default(self):
        assert _default().n_initial_spotoptim == 5


# ---------------------------------------------------------------------------
# Custom arguments
# ---------------------------------------------------------------------------


class TestCustomArgs:
    """Verify custom constructor arguments are stored correctly."""

    def test_custom_predict_size(self):
        task = LazyTask(predict_size=48)
        assert task.predict_size == 48

    def test_custom_contamination(self):
        task = LazyTask(contamination=0.05)
        assert task.contamination == 0.05

    def test_custom_agg_weights(self):
        weights = [1.0, -1.0, 0.5]
        task = OptunaTask(agg_weights=weights)
        assert task.agg_weights == weights

    def test_custom_n_trials_optuna(self):
        task = OptunaTask(n_trials_optuna=5)
        assert task.n_trials_optuna == 5

    def test_custom_n_trials_spotoptim(self):
        task = SpotOptimTask(n_trials_spotoptim=20)
        assert task.n_trials_spotoptim == 20

    def test_custom_number_folds(self):
        task = LazyTask(number_folds=5)
        assert task.number_folds == 5
        assert task.DELTA_VAL == pd.Timedelta(days=35)

    def test_custom_imputation_method(self):
        task = LazyTask(imputation_method="linear")
        assert task.imputation_method == "linear"

    def test_multitask_custom_task(self):
        mt = MultiTask(task="training")
        assert mt.TASK == "training"


# ---------------------------------------------------------------------------
# ConfigMulti delegation
# ---------------------------------------------------------------------------


class TestConfigDelegation:
    """Verify constructor args are correctly forwarded to ConfigMulti."""

    def test_config_is_config_multi(self):
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        assert isinstance(_lazy().config, ConfigMulti)

    def test_config_predict_size(self):
        task = LazyTask(predict_size=48)
        assert task.config.predict_size == 48

    def test_config_contamination(self):
        task = LazyTask(contamination=0.05)
        assert task.config.contamination == 0.05

    def test_config_imputation_method(self):
        task = OptunaTask(imputation_method="linear")
        assert task.config.imputation_method == "linear"

    def test_config_use_exogenous_features(self):
        task = SpotOptimTask(use_exogenous_features=False)
        assert task.config.use_exogenous_features is False

    def test_config_task_lazy(self):
        assert _lazy().config.task == "lazy"

    def test_config_task_optuna(self):
        assert _optuna().config.task == "optuna"

    def test_config_task_spotoptim(self):
        assert _spotoptim().config.task == "spotoptim"

    def test_config_task_multitask(self):
        mt = MultiTask(task="optuna")
        assert mt.config.task == "optuna"

    def test_config_n_trials_optuna(self):
        task = OptunaTask(n_trials_optuna=3)
        assert task.config.n_trials_optuna == 3

    def test_config_n_trials_spotoptim(self):
        task = SpotOptimTask(n_trials_spotoptim=7)
        assert task.config.n_trials_spotoptim == 7

    def test_config_n_initial_spotoptim(self):
        task = SpotOptimTask(n_initial_spotoptim=3)
        assert task.config.n_initial_spotoptim == 3

    def test_config_agg_weights(self):
        weights = [1.0, -1.0]
        task = LazyTask(agg_weights=weights)
        assert task.config.agg_weights == weights

    def test_config_bounds(self):
        bounds = [(0, 100), (-50, 200)]
        task = LazyTask(bounds=bounds)
        assert task.config.bounds == bounds

    def test_config_index_name(self):
        task = LazyTask(index_name="ts")
        assert task.config.index_name == "ts"

    def test_config_data_source(self):
        task = LazyTask(data_source="my_data.csv")
        assert task.config.data_source == "my_data.csv"

    def test_config_overrides(self):
        """Extra kwargs go to ConfigMulti constructor."""
        task = LazyTask(country_code="FR", random_state=42)
        assert task.config.country_code == "FR"
        assert task.config.random_state == 42

    def test_multitask_config_predict_size(self):
        mt = MultiTask(predict_size=48)
        assert mt.config.predict_size == 48


# ---------------------------------------------------------------------------
# Pipeline state initialisation
# ---------------------------------------------------------------------------


class TestPipelineState:
    """Verify initial pipeline state attributes for all task types."""

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_df_pipeline_initially_none(self, factory):
        assert factory().df_pipeline is None

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_df_test_initially_none(self, factory):
        assert factory().df_test is None

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_results_initially_empty(self, factory):
        assert factory().results == {}

    def test_weight_func_initially_none(self):
        assert _lazy().weight_func is None

    def test_exogenous_features_initially_none(self):
        assert _lazy().exogenous_features is None

    def test_exog_feature_names_initially_empty(self):
        assert _lazy().exog_feature_names == []

    def test_data_with_exog_initially_none(self):
        assert _lazy().data_with_exog is None

    def test_exo_pred_initially_none(self):
        assert _lazy().exo_pred is None


# ---------------------------------------------------------------------------
# Pipeline guard methods
# ---------------------------------------------------------------------------


class TestGuards:
    """Verify RuntimeError raised when steps called out of order."""

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_detect_outliers_before_prepare(self, factory):
        with pytest.raises(RuntimeError, match="prepare_data"):
            factory().detect_outliers()

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_impute_before_prepare(self, factory):
        with pytest.raises(RuntimeError, match="prepare_data"):
            factory().impute()

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_build_exogenous_before_prepare(self, factory):
        with pytest.raises(RuntimeError, match="prepare_data"):
            factory().build_exogenous_features()

    def test_plot_with_outliers_before_detect(self):
        with pytest.raises(RuntimeError, match="detect_outliers"):
            _lazy().plot_with_outliers()

    def test_run_before_prepare_multitask(self):
        with pytest.raises(RuntimeError, match="Pipeline data not prepared"):
            _default().run(show=False)

    def test_run_before_prepare_lazy(self):
        with pytest.raises(RuntimeError, match="Pipeline data not prepared"):
            _lazy().run(show=False)


# ---------------------------------------------------------------------------
# BaseTask.run() raises NotImplementedError
# ---------------------------------------------------------------------------


class TestBaseTaskRun:
    """Verify that BaseTask.run() is abstract-like."""

    def test_base_task_run_raises(self):
        task = LazyTask.__new__(BaseTask)
        BaseTask.__init__(task)
        with pytest.raises(NotImplementedError, match="must implement run"):
            task.run(show=False)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TestLogger:
    """Verify logger configuration across task types."""

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy, _optuna, _spotoptim],
        ids=["MultiTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_logger_is_logger_instance(self, factory):
        assert isinstance(factory().logger, logging.Logger)

    def test_logger_name_includes_class_multitask(self):
        assert "MultiTask" in _default().logger.name

    def test_logger_name_includes_class_lazy(self):
        assert "LazyTask" in _lazy().logger.name

    def test_logger_name_includes_class_optuna(self):
        assert "OptunaTask" in _optuna().logger.name

    def test_logger_name_includes_class_spotoptim(self):
        assert "SpotOptimTask" in _spotoptim().logger.name

    def test_custom_log_level(self):
        task = LazyTask(log_level=logging.DEBUG)
        assert task.logger.level == logging.DEBUG

    def test_default_log_level(self):
        assert _lazy().logger.level == logging.INFO


# ---------------------------------------------------------------------------
# create_forecaster()
# ---------------------------------------------------------------------------


class TestCreateForecaster:
    """Verify the forecaster factory method on all task types."""

    def test_returns_forecaster_recursive(self):
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        forecaster = _lazy().create_forecaster()
        assert isinstance(forecaster, ForecasterRecursive)

    def test_lags_from_config(self):
        task = LazyTask(lags_consider=[1, 2, 48])
        forecaster = task.create_forecaster()
        assert max(forecaster.lags) == 48

    def test_weight_func_none_initially(self):
        forecaster = _lazy().create_forecaster()
        assert forecaster.weight_func is None

    def test_multitask_create_forecaster(self):
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        forecaster = _default().create_forecaster()
        assert isinstance(forecaster, ForecasterRecursive)


# ---------------------------------------------------------------------------
# MultiTask run() dispatcher
# ---------------------------------------------------------------------------


class TestMultiTaskRunDispatcher:
    """Verify the MultiTask.run() method dispatches to the correct task."""

    def test_invalid_task_raises(self):
        mt = MultiTask(task="invalid_task")
        mt.df_pipeline = pd.DataFrame()
        mt.config.end_train_ts = pd.Timestamp("2024-01-01", tz="UTC")
        with pytest.raises(ValueError, match="Unknown task"):
            mt.run(show=False)

    def test_valid_task_names(self):
        for task in ["lazy", "training", "optuna", "spotoptim"]:
            mt = MultiTask(task=task)
            assert mt.TASK == task

    def test_has_run_task_lazy(self):
        assert hasattr(_default(), "run_task_lazy")

    def test_has_run_task_optuna(self):
        assert hasattr(_default(), "run_task_optuna")

    def test_has_run_task_spotoptim(self):
        assert hasattr(_default(), "run_task_spotoptim")


# ---------------------------------------------------------------------------
# Default search spaces
# ---------------------------------------------------------------------------


class TestSearchSpaces:
    """Verify the default search space methods."""

    def test_spotoptim_search_space_is_dict(self):
        space = SpotOptimTask._default_spotoptim_search_space()
        assert isinstance(space, dict)

    def test_spotoptim_search_space_keys(self):
        space = SpotOptimTask._default_spotoptim_search_space()
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
        space = SpotOptimTask._default_spotoptim_search_space()
        assert isinstance(space["lags"], list)

    def test_multitask_spotoptim_search_space_backward_compat(self):
        """MultiTask still exposes _default_spotoptim_search_space."""
        space = MultiTask._default_spotoptim_search_space()
        assert isinstance(space, dict)
        assert "lags" in space

    def test_multitask_optuna_search_space_backward_compat(self):
        """MultiTask still exposes _default_optuna_search_space (static)."""
        assert callable(MultiTask._default_optuna_search_space)

    def test_optuna_task_has_search_space(self):
        assert callable(OptunaTask._default_optuna_search_space)


# ---------------------------------------------------------------------------
# Integration: prepare_data with package demo data
# ---------------------------------------------------------------------------


class TestPrepareData:
    """Integration test: load demo10 and run prepare_data."""

    @pytest.mark.parametrize(
        "factory",
        [_default, _lazy],
        ids=["MultiTask", "LazyTask"],
    )
    def test_prepare_data_loads_data(self, factory):
        task = factory()
        task.prepare_data()
        assert task.df_pipeline is not None
        assert task.df_test is not None
        assert isinstance(task.df_pipeline, pd.DataFrame)
        assert task.df_pipeline.shape[0] > 0

    def test_prepare_data_sets_targets(self):
        task = LazyTask(data_frame_name="demo10")
        task.prepare_data()
        assert task.config.targets is not None
        assert len(task.config.targets) > 0

    def test_prepare_data_sets_date_ranges(self):
        task = LazyTask(data_frame_name="demo10")
        task.prepare_data()
        assert task.config.data_start is not None
        assert task.config.data_end is not None
        assert task.config.cov_start is not None
        assert task.config.cov_end is not None

    def test_prepare_data_returns_self(self):
        task = LazyTask()
        result = task.prepare_data()
        assert result is task

    def test_multitask_prepare_data_returns_self(self):
        mt = MultiTask()
        result = mt.prepare_data()
        assert result is mt


# ---------------------------------------------------------------------------
# Integration: detect_outliers
# ---------------------------------------------------------------------------


class TestDetectOutliers:
    """Integration test: outlier detection after prepare_data."""

    def test_detect_outliers_runs(self):
        task = LazyTask()
        task.prepare_data()
        task.detect_outliers()
        assert task.df_pipeline_original is not None

    def test_detect_outliers_returns_self(self):
        task = LazyTask()
        task.prepare_data()
        result = task.detect_outliers()
        assert result is task

    def test_detect_outliers_preserves_original(self):
        task = LazyTask()
        task.prepare_data()
        shape_before = task.df_pipeline.shape
        task.detect_outliers()
        assert task.df_pipeline_original.shape == shape_before


# ---------------------------------------------------------------------------
# Integration: impute
# ---------------------------------------------------------------------------


class TestImpute:
    """Integration test: imputation after prepare_data."""

    def test_impute_runs(self):
        task = LazyTask()
        task.prepare_data()
        task.detect_outliers()
        task.impute()
        assert task.df_pipeline.notna().all().all()

    def test_impute_returns_self(self):
        task = LazyTask()
        task.prepare_data()
        task.detect_outliers()
        result = task.impute()
        assert result is task


# ---------------------------------------------------------------------------
# Integration: build_exogenous_features
# ---------------------------------------------------------------------------


class TestExogenousFeatures:
    """Integration test: exogenous feature engineering."""

    def test_exog_disabled(self):
        task = LazyTask(use_exogenous_features=False)
        task.prepare_data()
        task.detect_outliers()
        task.impute()
        task.build_exogenous_features()
        assert task.exogenous_features is None
        assert task.exog_feature_names == []

    def test_exog_enabled(self):
        task = LazyTask(use_exogenous_features=True)
        task.prepare_data()
        task.detect_outliers()
        task.impute()
        task.build_exogenous_features()
        assert task.exogenous_features is not None
        assert len(task.exog_feature_names) > 0
        assert task.data_with_exog is not None
        assert task.exo_pred is not None

    def test_build_exog_returns_self(self):
        task = LazyTask(use_exogenous_features=False)
        task.prepare_data()
        task.detect_outliers()
        task.impute()
        result = task.build_exogenous_features()
        assert result is task


# ---------------------------------------------------------------------------
# Integration: log_summary
# ---------------------------------------------------------------------------


class TestLogSummary:
    """Verify log_summary does not raise."""

    def test_log_summary_no_exog(self):
        task = LazyTask(use_exogenous_features=False)
        task.prepare_data()
        task.detect_outliers()
        task.impute()
        task.build_exogenous_features()
        task.log_summary()

    def test_log_summary_with_exog(self):
        task = LazyTask(use_exogenous_features=True)
        task.prepare_data()
        task.detect_outliers()
        task.impute()
        task.build_exogenous_features()
        task.log_summary()


# ---------------------------------------------------------------------------
# Method chaining across task types
# ---------------------------------------------------------------------------


class TestMethodChaining:
    """Verify method chaining works identically across task types."""

    @pytest.mark.parametrize(
        "cls",
        [LazyTask, OptunaTask, SpotOptimTask, MultiTask],
        ids=["LazyTask", "OptunaTask", "SpotOptimTask", "MultiTask"],
    )
    def test_full_chain(self, cls):
        kwargs = {"use_exogenous_features": False}
        if cls is MultiTask:
            kwargs["task"] = "lazy"
        task = cls(**kwargs)
        result = (
            task.prepare_data().detect_outliers().impute().build_exogenous_features()
        )
        assert result is task
