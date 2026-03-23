# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for PredictTask — prediction-only using saved models.

Covers:
- Import paths (multitask package and manager package)
- PredictTask class attributes and inheritance
- PredictTask.run() halts with RuntimeError when no models are saved
- PredictTask.run() loads saved models and produces predictions
- PredictTask.run() filters by task_name
- PredictTask.run() filters by max_age_days
- PredictTask.run() raises when a target has no saved model
- MultiTask dispatcher recognizes "predict" task
- MultiTask.run_task_predict() method exists
- execute_predict() function accessible
- Round-trip: train via LazyTask → save → predict via PredictTask
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2.manager.multitask import (
    BaseTask,
    LazyTask,
    MultiTask,
    PredictTask,
)
from spotforecast2.manager.multitask.predict import execute_predict
from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

_DEMO_CSV = str(get_package_data_home() / "demo10.csv")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    """Load the package demo10 CSV once for the whole module."""
    return fetch_data(filename=_DEMO_CSV)


@pytest.fixture()
def predict_task(tmp_path: Path) -> PredictTask:
    """Return a PredictTask with cache_home set to a temp directory."""
    return PredictTask(data_frame_name="test_data", cache_home=tmp_path)


@pytest.fixture()
def lazy_task(tmp_path: Path) -> LazyTask:
    """Return a LazyTask with matching configuration for saving models."""
    return LazyTask(data_frame_name="test_data", cache_home=tmp_path)


@pytest.fixture()
def mock_forecasters() -> Dict[str, Any]:
    """Return a dict of trivially fitted sklearn models as stand-ins."""
    m1 = LinearRegression()
    m1.fit(np.arange(10).reshape(-1, 1), np.arange(10))
    m2 = LinearRegression()
    m2.fit(np.arange(10).reshape(-1, 1), np.arange(10) * 2)
    return {"target_0": m1, "target_1": m2}


@pytest.fixture()
def single_forecaster() -> Dict[str, Any]:
    """Return a dict with a single fitted model."""
    m = LinearRegression()
    m.fit(np.arange(10).reshape(-1, 1), np.arange(10))
    return {"target_0": m}


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


class TestImports:
    """Verify PredictTask is importable from the expected locations."""

    def test_import_from_multitask(self):
        from spotforecast2.manager.multitask import PredictTask  # noqa: F811

        assert PredictTask is not None

    def test_import_from_manager(self):
        from spotforecast2.manager import PredictTask  # noqa: F811

        assert PredictTask is not None

    def test_import_execute_predict(self):
        from spotforecast2.manager.multitask.predict import (
            execute_predict,
        )  # noqa: F811

        assert callable(execute_predict)


# ---------------------------------------------------------------------------
# Class attributes and inheritance
# ---------------------------------------------------------------------------


class TestClassAttributes:
    """Verify PredictTask class-level attributes and hierarchy."""

    def test_inherits_from_base_task(self):
        assert issubclass(PredictTask, BaseTask)

    def test_task_name_attribute(self):
        assert PredictTask._task_name == "predict"

    def test_instance_task_attribute(self, predict_task):
        assert predict_task.TASK == "predict"

    def test_instance_data_frame_name(self, predict_task):
        assert predict_task.data_frame_name == "test_data"

    def test_has_run_method(self):
        assert hasattr(PredictTask, "run")
        assert callable(getattr(PredictTask, "run"))

    def test_inherits_save_models(self):
        assert hasattr(PredictTask, "save_models")

    def test_inherits_load_models(self):
        assert hasattr(PredictTask, "load_models")

    def test_inherits_prepare_data(self):
        assert hasattr(PredictTask, "prepare_data")


# ---------------------------------------------------------------------------
# run() — no saved models → RuntimeError
# ---------------------------------------------------------------------------


class TestRunNoModels:
    """Verify PredictTask.run() raises when no models exist."""

    def test_run_raises_no_models_empty_cache(self, predict_task, demo_df):
        """When the models directory does not exist at all."""
        predict_task.prepare_data(demo_data=demo_df)
        predict_task.detect_outliers()
        predict_task.impute()
        with pytest.raises(RuntimeError, match="No saved models found"):
            predict_task.run(show=False)

    def test_run_raises_no_models_empty_dir(self, tmp_path, demo_df):
        """When the models directory exists but contains no matching files."""
        model_dir = tmp_path / "models" / "test_data"
        model_dir.mkdir(parents=True)
        task = PredictTask(data_frame_name="test_data", cache_home=tmp_path)
        task.prepare_data(demo_data=demo_df)
        task.detect_outliers()
        task.impute()
        with pytest.raises(RuntimeError, match="No saved models found"):
            task.run(show=False)


# ---------------------------------------------------------------------------
# run() — missing target → RuntimeError
# ---------------------------------------------------------------------------


class TestRunMissingTarget:
    """Verify error when a saved model exists for some targets but not all."""

    def test_run_raises_for_missing_target(self, tmp_path, single_forecaster, demo_df):
        """Save model for target_0 but the task needs more targets."""
        task = PredictTask(data_frame_name="test_data", cache_home=tmp_path)
        task.prepare_data(demo_data=demo_df)
        task.detect_outliers()
        task.impute()

        # Only save for target_0, but the demo10 dataset has multiple targets
        task.save_models(task_name="lazy", forecasters=single_forecaster)

        targets = task.config.targets
        if len(targets) > 1:
            # At least one target should be missing
            with pytest.raises(RuntimeError, match="No saved model found for target"):
                task.run(show=False)


# ---------------------------------------------------------------------------
# run() — successful prediction with saved models
# ---------------------------------------------------------------------------


class TestRunSuccess:
    """Verify PredictTask.run() succeeds when models are available."""

    def test_full_round_trip(self, tmp_path, demo_df):
        """Train with LazyTask, save models, predict with PredictTask."""
        # 1. Train and save
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        # 2. Predict using saved models
        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        result = pred.run(show=False)

        assert result is not None
        assert "predict" in pred.results
        for target in pred.config.targets:
            pkg = pred.results["predict"][target]
            assert "future_pred" in pkg
            assert len(pkg["future_pred"]) == 24

    def test_results_stored_under_predict_key(self, tmp_path, demo_df):
        """Verify per-target results are stored in results['predict']."""
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        pred.run(show=False)

        assert "predict" in pred.results
        assert isinstance(pred.results["predict"], dict)
        assert len(pred.results["predict"]) == len(pred.config.targets)

    def test_agg_results_stored(self, tmp_path, demo_df):
        """Verify aggregated results are stored."""
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        pred.run(show=False)

        assert "task 5: Predict (loaded models)" in pred.agg_results


# ---------------------------------------------------------------------------
# run() — task_name filter
# ---------------------------------------------------------------------------


class TestRunTaskNameFilter:
    """Verify PredictTask.run() can filter by source task_name."""

    def test_filter_by_task_name(self, tmp_path, demo_df):
        """Save under 'lazy', load with task_name='lazy'."""
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        result = pred.run(show=False, task_name="lazy")
        assert result is not None

    def test_filter_by_nonexistent_task_name(self, tmp_path, demo_df):
        """Save under 'lazy' but filter for 'optuna' → no models."""
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        with pytest.raises(RuntimeError, match="No saved models found"):
            pred.run(show=False, task_name="optuna")


# ---------------------------------------------------------------------------
# run() — max_age_days filter
# ---------------------------------------------------------------------------


class TestRunMaxAgeFilter:
    """Verify PredictTask.run() respects max_age_days."""

    def test_recent_models_accepted(self, tmp_path, demo_df):
        """Models saved just now should be within any reasonable age."""
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        result = pred.run(show=False, max_age_days=1.0)
        assert result is not None

    def test_expired_models_rejected(self, tmp_path, demo_df):
        """Manually create an old model file to test expiry filtering."""
        from joblib import dump

        model_dir = tmp_path / "models" / "demo10"
        model_dir.mkdir(parents=True)

        # Create a PredictTask to discover the target names
        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()

        # Save old-timestamped model files for all targets
        for target in pred.config.targets:
            m = LinearRegression()
            m.fit(np.arange(10).reshape(-1, 1), np.arange(10))
            old_ts = "20240101_000000"
            filename = f"demo10_{target}_lazy_{old_ts}.joblib"
            dump(m, model_dir / filename, compress=3)

        with pytest.raises(RuntimeError, match="No saved models found"):
            pred.run(show=False, max_age_days=1.0)


# ---------------------------------------------------------------------------
# MultiTask dispatcher integration
# ---------------------------------------------------------------------------


class TestMultiTaskDispatcher:
    """Verify MultiTask recognizes and dispatches the 'predict' task."""

    def test_multitask_has_run_task_predict(self):
        assert hasattr(MultiTask, "run_task_predict")
        assert callable(getattr(MultiTask, "run_task_predict"))

    def test_multitask_dispatch_predict_no_models(self, tmp_path):
        """MultiTask.run(task='predict') should raise when no models exist."""
        mt = MultiTask(
            task="predict",
            data_frame_name="test_data",
            data_source=_DEMO_CSV,
            cache_home=tmp_path,
        )
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        with pytest.raises(RuntimeError, match="No saved models found"):
            mt.run(show=False)

    def test_multitask_dispatches_predict_explicitly(self, tmp_path):
        """MultiTask.run(task='predict') dispatches to predict logic."""
        mt = MultiTask(
            data_frame_name="test_data",
            data_source=_DEMO_CSV,
            cache_home=tmp_path,
        )
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        with pytest.raises(RuntimeError, match="No saved models found"):
            mt.run(task="predict", show=False)

    def test_multitask_run_task_predict_method(self, tmp_path):
        """MultiTask.run_task_predict() directly invokes predict logic."""
        mt = MultiTask(
            data_frame_name="test_data",
            data_source=_DEMO_CSV,
            cache_home=tmp_path,
        )
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        with pytest.raises(RuntimeError, match="No saved models found"):
            mt.run_task_predict(show=False)

    def test_multitask_full_round_trip(self, tmp_path):
        """Train via 'lazy' then predict via 'predict' using MultiTask."""
        mt = MultiTask(
            data_frame_name="demo10",
            data_source=_DEMO_CSV,
            cache_home=tmp_path,
            predict_size=24,
        )
        mt.prepare_data()
        mt.detect_outliers()
        mt.impute()
        mt.build_exogenous_features()

        # Train
        mt.run(task="lazy", show=False)
        mt.save_models(task_name="lazy")

        # Predict
        result = mt.run(task="predict", show=False)
        assert result is not None
        assert "predict" in mt.results


# ---------------------------------------------------------------------------
# execute_predict function
# ---------------------------------------------------------------------------


class TestExecutePredict:
    """Verify the module-level execute_predict function."""

    def test_execute_predict_is_callable(self):
        assert callable(execute_predict)

    def test_execute_predict_no_models(self, predict_task, demo_df):
        predict_task.prepare_data(demo_data=demo_df)
        predict_task.detect_outliers()
        predict_task.impute()
        with pytest.raises(RuntimeError, match="No saved models found"):
            execute_predict(predict_task, show=False)


# ---------------------------------------------------------------------------
# PredictTask constructor arguments
# ---------------------------------------------------------------------------


class TestConstructorArgs:
    """Verify PredictTask passes arguments through to BaseTask."""

    def test_default_data_frame_name(self):
        task = PredictTask()
        assert task.data_frame_name == "demo10"

    def test_custom_predict_size(self, tmp_path):
        task = PredictTask(predict_size=48, cache_home=tmp_path)
        assert task.config.predict_size == 48

    def test_custom_cache_home(self, tmp_path):
        task = PredictTask(cache_home=tmp_path)
        assert task.cache_home == tmp_path

    def test_custom_data_frame_name(self, tmp_path):
        task = PredictTask(data_frame_name="my_dataset", cache_home=tmp_path)
        assert task.data_frame_name == "my_dataset"


# ---------------------------------------------------------------------------
# Prediction package structure
# ---------------------------------------------------------------------------


class TestPredictionPackage:
    """Verify the prediction package returned by PredictTask contains
    the expected keys with correct types."""

    def test_package_keys(self, tmp_path, demo_df):
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        pred.run(show=False)

        for target in pred.config.targets:
            pkg = pred.results["predict"][target]
            assert "train_actual" in pkg
            assert "train_pred" in pkg
            assert "future_pred" in pkg
            assert "metrics_train" in pkg
            assert "validation_passed" in pkg

    def test_future_pred_length(self, tmp_path, demo_df):
        lazy = LazyTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        lazy.prepare_data(demo_data=demo_df)
        lazy.detect_outliers()
        lazy.impute()
        lazy.build_exogenous_features()
        lazy.run(show=False)
        lazy.save_models(task_name="lazy")

        pred = PredictTask(
            data_frame_name="demo10",
            cache_home=tmp_path,
            predict_size=24,
        )
        pred.prepare_data(demo_data=demo_df)
        pred.detect_outliers()
        pred.impute()
        pred.build_exogenous_features()
        pred.run(show=False)

        for target in pred.config.targets:
            assert len(pred.results["predict"][target]["future_pred"]) == 24


# ---------------------------------------------------------------------------
# Data isolation between datasets
# ---------------------------------------------------------------------------


class TestDataIsolation:
    """Verify models from one data_frame_name are not loaded by another."""

    def test_different_data_frame_names_isolated(self, tmp_path, single_forecaster):
        task_a = PredictTask(data_frame_name="dataset_a", cache_home=tmp_path)
        task_b = PredictTask(data_frame_name="dataset_b", cache_home=tmp_path)

        # Save under dataset_a
        task_a.save_models(task_name="lazy", forecasters=single_forecaster)

        # dataset_b should find nothing
        loaded = task_b.load_models(task_name="lazy")
        assert loaded == {}
