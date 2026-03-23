# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for BaseTask.save_models() and BaseTask.load_models().

Covers:
- save_models writes .joblib files with correct naming convention
- save_models validates task_name against allowed values
- save_models raises when no results are available
- save_models accepts explicit forecasters dict
- load_models retrieves the most recently saved model
- load_models filters by task_name
- load_models filters by target
- load_models filters by max_age_days
- Round-trip: save then load returns a functional model
- Multiple saves: most recent wins
- Missing models directory returns empty dict
- Method availability on all task subclasses
- Idempotency of save/load cycle
"""

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2.manager.multitask import (
    BaseTask,
    LazyTask,
    OptunaTask,
    SpotOptimTask,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task(tmp_path: Path) -> LazyTask:
    """Return a LazyTask with cache_home set to a temporary directory."""
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
# save_models — file creation and naming
# ---------------------------------------------------------------------------


class TestSaveModels:
    """Verify save_models writes correct joblib files."""

    def test_save_creates_files(self, task, mock_forecasters):
        paths = task.save_models(task_name="lazy", forecasters=mock_forecasters)
        assert len(paths) == 2
        for target, path in paths.items():
            assert path.exists()
            assert path.suffix == ".joblib"

    def test_save_file_in_models_dir(self, task, mock_forecasters):
        paths = task.save_models(task_name="lazy", forecasters=mock_forecasters)
        for path in paths.values():
            assert path.parent.name == "test_data"
            assert path.parent.parent.name == "models"

    def test_save_filename_format(self, task, single_forecaster):
        paths = task.save_models(task_name="lazy", forecasters=single_forecaster)
        path = paths["target_0"]
        assert path.name.startswith("test_data_target_0_lazy_")
        assert path.name.endswith(".joblib")

    def test_save_filename_contains_task_name(self, task, single_forecaster):
        for tname in ("lazy", "optuna", "spotoptim"):
            paths = task.save_models(task_name=tname, forecasters=single_forecaster)
            path = paths["target_0"]
            assert f"_{tname}_" in path.name

    def test_save_returns_dict_of_paths(self, task, mock_forecasters):
        result = task.save_models(task_name="lazy", forecasters=mock_forecasters)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, Path)


# ---------------------------------------------------------------------------
# save_models — validation
# ---------------------------------------------------------------------------


class TestSaveModelsValidation:
    """Verify save_models raises on invalid inputs."""

    def test_invalid_task_name_raises_value_error(self, task, single_forecaster):
        with pytest.raises(ValueError, match="Unknown task_name"):
            task.save_models(task_name="invalid", forecasters=single_forecaster)

    def test_no_results_raises_runtime_error(self, task):
        with pytest.raises(RuntimeError, match="No results for task"):
            task.save_models(task_name="lazy")

    def test_missing_forecaster_key_raises_runtime_error(self, task):
        task.results["lazy"] = {"target_0": {"predictions": [1, 2, 3]}}
        with pytest.raises(RuntimeError, match="does not contain a 'forecaster' key"):
            task.save_models(task_name="lazy")


# ---------------------------------------------------------------------------
# save_models — from results dict
# ---------------------------------------------------------------------------


class TestSaveModelsFromResults:
    """Verify save_models can extract forecasters from self.results."""

    def test_save_from_results(self, task, single_forecaster):
        m = single_forecaster["target_0"]
        task.results["lazy"] = {"target_0": {"forecaster": m, "predictions": []}}
        paths = task.save_models(task_name="lazy")
        assert "target_0" in paths
        assert paths["target_0"].exists()

    def test_save_from_results_multiple_targets(self, task, mock_forecasters):
        task.results["lazy"] = {
            t: {"forecaster": f, "predictions": []} for t, f in mock_forecasters.items()
        }
        paths = task.save_models(task_name="lazy")
        assert len(paths) == 2
        for p in paths.values():
            assert p.exists()


# ---------------------------------------------------------------------------
# load_models — basic retrieval
# ---------------------------------------------------------------------------


class TestLoadModels:
    """Verify load_models retrieves saved models."""

    def test_load_returns_empty_when_no_dir(self, tmp_path):
        task = LazyTask(
            data_frame_name="test_data",
            cache_home=tmp_path / "nonexistent_subdir",
        )
        result = task.load_models()
        assert result == {}

    def test_load_returns_empty_when_no_files(self, task):
        result = task.load_models()
        assert result == {}

    def test_round_trip(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        loaded = task.load_models()
        assert "target_0" in loaded
        # Verify it's a functional model
        pred = loaded["target_0"].predict(np.array([[5.0]]))
        assert len(pred) == 1

    def test_round_trip_multiple_targets(self, task, mock_forecasters):
        task.save_models(task_name="lazy", forecasters=mock_forecasters)
        loaded = task.load_models()
        assert set(loaded.keys()) == {"target_0", "target_1"}

    def test_loaded_model_preserves_coefficients(self, task, single_forecaster):
        original = single_forecaster["target_0"]
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        loaded = task.load_models()
        np.testing.assert_array_almost_equal(original.coef_, loaded["target_0"].coef_)
        np.testing.assert_array_almost_equal(
            original.intercept_, loaded["target_0"].intercept_
        )


# ---------------------------------------------------------------------------
# load_models — filtering
# ---------------------------------------------------------------------------


class TestLoadModelsFiltering:
    """Verify load_models correctly filters by task_name, target, max_age."""

    def test_filter_by_task_name(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        task.save_models(task_name="optuna", forecasters=single_forecaster)

        lazy_models = task.load_models(task_name="lazy")
        assert "target_0" in lazy_models

        optuna_models = task.load_models(task_name="optuna")
        assert "target_0" in optuna_models

    def test_filter_by_task_name_excludes_other_tasks(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        spotoptim_models = task.load_models(task_name="spotoptim")
        assert spotoptim_models == {}

    def test_filter_by_target(self, task, mock_forecasters):
        task.save_models(task_name="lazy", forecasters=mock_forecasters)
        loaded = task.load_models(target="target_0")
        assert "target_0" in loaded
        assert "target_1" not in loaded

    def test_filter_by_target_no_match(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        loaded = task.load_models(target="nonexistent_target")
        assert loaded == {}

    def test_filter_by_max_age(self, task, single_forecaster):
        task.save_models(task_name="lazy", forecasters=single_forecaster)
        loaded = task.load_models(max_age_days=1.0)
        assert "target_0" in loaded

    def test_filter_by_max_age_expired(self, task, tmp_path):
        """Manually create an old model file to test expiry."""
        model_dir = tmp_path / "models" / "test_data"
        model_dir.mkdir(parents=True)
        old_ts = "20240101_000000"
        filename = f"test_data_target_0_lazy_{old_ts}.joblib"
        filepath = model_dir / filename
        from joblib import dump

        m = LinearRegression()
        m.fit(np.arange(5).reshape(-1, 1), np.arange(5))
        dump(m, filepath, compress=3)

        loaded = task.load_models(max_age_days=1.0)
        assert loaded == {}

    def test_combined_filters(self, task, mock_forecasters):
        task.save_models(task_name="lazy", forecasters=mock_forecasters)
        task.save_models(task_name="optuna", forecasters=mock_forecasters)
        loaded = task.load_models(task_name="lazy", target="target_1")
        assert "target_1" in loaded
        assert "target_0" not in loaded


# ---------------------------------------------------------------------------
# load_models — most recent wins
# ---------------------------------------------------------------------------


class TestLoadModelsMostRecent:
    """Verify that when multiple snapshots exist, the newest is loaded."""

    def test_most_recent_model_wins(self, task):
        m1 = LinearRegression()
        m1.fit(np.arange(10).reshape(-1, 1), np.arange(10))
        task.save_models(task_name="lazy", forecasters={"target_0": m1})

        time.sleep(1.1)  # Ensure different timestamp

        m2 = LinearRegression()
        m2.fit(np.arange(10).reshape(-1, 1), np.arange(10) * 3)
        task.save_models(task_name="lazy", forecasters={"target_0": m2})

        loaded = task.load_models(task_name="lazy")
        # The second model has coef ~3, the first ~1
        np.testing.assert_array_almost_equal(loaded["target_0"].coef_, m2.coef_)


# ---------------------------------------------------------------------------
# Method availability on all task subclasses
# ---------------------------------------------------------------------------


class TestMethodAvailability:
    """Verify save_models and load_models are available on all task classes."""

    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask, OptunaTask, SpotOptimTask],
        ids=["BaseTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_has_save_models(self, cls):
        assert hasattr(cls, "save_models")
        assert callable(getattr(cls, "save_models"))

    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask, OptunaTask, SpotOptimTask],
        ids=["BaseTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_has_load_models(self, cls):
        assert hasattr(cls, "load_models")
        assert callable(getattr(cls, "load_models"))


# ---------------------------------------------------------------------------
# Task-name mapping
# ---------------------------------------------------------------------------


class TestTaskModelNames:
    """Verify the _TASK_MODEL_NAMES class attribute."""

    def test_task_model_names_is_dict(self):
        assert isinstance(BaseTask._TASK_MODEL_NAMES, dict)

    def test_task_model_names_contains_lazy(self):
        assert "lazy" in BaseTask._TASK_MODEL_NAMES

    def test_task_model_names_contains_optuna(self):
        assert "optuna" in BaseTask._TASK_MODEL_NAMES

    def test_task_model_names_contains_spotoptim(self):
        assert "spotoptim" in BaseTask._TASK_MODEL_NAMES

    @pytest.mark.parametrize("task_name", ["lazy", "optuna", "spotoptim"])
    def test_all_valid_task_names_accepted(self, task, single_forecaster, task_name):
        paths = task.save_models(task_name=task_name, forecasters=single_forecaster)
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# Cross-task save/load isolation
# ---------------------------------------------------------------------------


class TestCrossTaskIsolation:
    """Verify models from different tasks do not interfere."""

    def test_different_tasks_produce_different_files(self, task, single_forecaster):
        paths_lazy = task.save_models(task_name="lazy", forecasters=single_forecaster)
        paths_optuna = task.save_models(
            task_name="optuna", forecasters=single_forecaster
        )
        assert paths_lazy["target_0"] != paths_optuna["target_0"]

    def test_load_with_task_filter_returns_correct_task(self, task):
        m_lazy = LinearRegression()
        m_lazy.fit(np.arange(10).reshape(-1, 1), np.arange(10))
        m_optuna = LinearRegression()
        m_optuna.fit(np.arange(10).reshape(-1, 1), np.arange(10) * 5)

        task.save_models(task_name="lazy", forecasters={"target_0": m_lazy})
        task.save_models(task_name="optuna", forecasters={"target_0": m_optuna})

        lazy_loaded = task.load_models(task_name="lazy")
        optuna_loaded = task.load_models(task_name="optuna")

        np.testing.assert_array_almost_equal(
            lazy_loaded["target_0"].coef_, m_lazy.coef_
        )
        np.testing.assert_array_almost_equal(
            optuna_loaded["target_0"].coef_, m_optuna.coef_
        )


# ---------------------------------------------------------------------------
# Different data_frame_name isolation
# ---------------------------------------------------------------------------


class TestDataFrameNameIsolation:
    """Verify models from different data_frame_names do not collude."""

    def test_different_data_frame_names_isolated(self, tmp_path, single_forecaster):
        task_a = LazyTask(data_frame_name="dataset_a", cache_home=tmp_path)
        task_b = LazyTask(data_frame_name="dataset_b", cache_home=tmp_path)

        task_a.save_models(task_name="lazy", forecasters=single_forecaster)

        loaded_b = task_b.load_models(task_name="lazy")
        assert loaded_b == {}

        loaded_a = task_a.load_models(task_name="lazy")
        assert "target_0" in loaded_a
