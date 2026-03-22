# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for tuning-result persistence (save / load).

Covers:
- save_tuning_results writes a valid JSON file with correct structure
- load_tuning_results retrieves the most recent matching result
- Filtering by task_name
- Filtering by max_age_days
- Round-trip: save then load returns identical data
- Multiple saves: most recent wins
- Missing tuning_results directory returns None
- Corrupt JSON files are skipped gracefully
- numpy arrays in best_lags are serialized correctly
- LazyTask.run() with use_tuned_params loads cached results
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from spotforecast2.manager.multitask import BaseTask, LazyTask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task(tmp_path: Path) -> LazyTask:
    """Return a LazyTask with cache_home set to a temporary directory."""
    return LazyTask(data_frame_name="test_data", cache_home=tmp_path)


@pytest.fixture()
def sample_params() -> Dict[str, Any]:
    """Sample best_params for testing."""
    return {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
    }


@pytest.fixture()
def sample_lags():
    """Sample best_lags for testing."""
    return [1, 2, 24, 48]


# ---------------------------------------------------------------------------
# save_tuning_results
# ---------------------------------------------------------------------------


class TestSaveTuningResults:
    """Verify save_tuning_results writes correct JSON files."""

    def test_save_creates_file(self, task, sample_params, sample_lags):
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_file_in_tuning_results_dir(self, task, sample_params, sample_lags):
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        assert path.parent.name == "tuning_results"

    def test_save_filename_format(self, task, sample_params, sample_lags):
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        assert path.name.startswith("test_data_target_0_optuna_")
        assert path.name.endswith(".json")

    def test_save_json_structure(self, task, sample_params, sample_lags):
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        with open(path) as f:
            data = json.load(f)

        assert data["data_frame_name"] == "test_data"
        assert data["target"] == "target_0"
        assert data["task_name"] == "optuna"
        assert "timestamp" in data
        assert data["best_params"] == sample_params
        assert data["best_lags"] == sample_lags

    def test_save_timestamp_format(self, task, sample_params, sample_lags):
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        with open(path) as f:
            data = json.load(f)

        # Should parse without error
        ts = datetime.strptime(data["timestamp"], "%Y%m%d_%H%M%S")
        assert ts.year >= 2026

    def test_save_numpy_lags(self, task, sample_params):
        """numpy arrays should be serialized to plain lists."""
        np_lags = np.array([1, 2, 24, 48])
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=np_lags,
        )
        with open(path) as f:
            data = json.load(f)

        assert data["best_lags"] == [1, 2, 24, 48]
        assert isinstance(data["best_lags"], list)

    def test_save_int_lags(self, task, sample_params):
        """Integer lags should be preserved."""
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=48,
        )
        with open(path) as f:
            data = json.load(f)

        assert data["best_lags"] == 48

    def test_save_returns_path(self, task, sample_params, sample_lags):
        result = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        assert isinstance(result, Path)

    def test_save_spotoptim(self, task, sample_params, sample_lags):
        path = task.save_tuning_results(
            target="target_0",
            task_name="spotoptim",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        assert "spotoptim" in path.name


# ---------------------------------------------------------------------------
# load_tuning_results
# ---------------------------------------------------------------------------


class TestLoadTuningResults:
    """Verify load_tuning_results retrieves correct cached results."""

    def test_load_returns_none_when_empty(self, task):
        result = task.load_tuning_results(target="target_0")
        assert result is None

    def test_load_returns_none_when_no_dir(self, tmp_path):
        """No tuning_results directory at all."""
        task = LazyTask(
            data_frame_name="test_data",
            cache_home=tmp_path / "nonexistent_subdir",
        )
        result = task.load_tuning_results(target="target_0")
        assert result is None

    def test_round_trip(self, task, sample_params, sample_lags):
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        result = task.load_tuning_results(target="target_0")
        assert result is not None
        assert result["best_params"] == sample_params
        assert result["best_lags"] == sample_lags
        assert result["task_name"] == "optuna"
        assert result["target"] == "target_0"

    def test_load_most_recent(self, task, sample_params):
        """When multiple files exist, the most recent is returned."""
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params={"n_estimators": 100},
            best_lags=24,
        )
        time.sleep(1.1)  # Ensure different timestamp
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params={"n_estimators": 200},
            best_lags=48,
        )
        result = task.load_tuning_results(target="target_0")
        assert result["best_params"]["n_estimators"] == 200
        assert result["best_lags"] == 48

    def test_load_filter_by_task_name(self, task, sample_params, sample_lags):
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        task.save_tuning_results(
            target="target_0",
            task_name="spotoptim",
            best_params={"n_estimators": 300},
            best_lags=72,
        )

        optuna_result = task.load_tuning_results(target="target_0", task_name="optuna")
        assert optuna_result["task_name"] == "optuna"

        spotoptim_result = task.load_tuning_results(
            target="target_0", task_name="spotoptim"
        )
        assert spotoptim_result["task_name"] == "spotoptim"

    def test_load_filter_by_max_age(self, task, sample_params, sample_lags):
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        # With a very small max_age_days the result should still be found
        result = task.load_tuning_results(target="target_0", max_age_days=1.0)
        assert result is not None

    def test_load_expired_returns_none(
        self, task, sample_params, sample_lags, tmp_path
    ):
        """Manually create an old file to test expiry."""
        tuning_dir = tmp_path / "tuning_results"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        old_ts = "20240101_000000"
        filename = f"test_data_target_0_optuna_{old_ts}.json"
        payload = {
            "data_frame_name": "test_data",
            "target": "target_0",
            "task_name": "optuna",
            "timestamp": old_ts,
            "best_params": sample_params,
            "best_lags": sample_lags,
        }
        with open(tuning_dir / filename, "w") as f:
            json.dump(payload, f)

        result = task.load_tuning_results(target="target_0", max_age_days=1.0)
        assert result is None

    def test_load_skips_corrupt_json(self, task, sample_params, sample_lags, tmp_path):
        """Corrupt JSON files are skipped gracefully."""
        tuning_dir = tmp_path / "tuning_results"
        tuning_dir.mkdir(parents=True, exist_ok=True)

        # Write a corrupt file
        corrupt_file = tuning_dir / "test_data_target_0_optuna_29991231_235959.json"
        corrupt_file.write_text("NOT VALID JSON {{{")

        # Save a valid one
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        result = task.load_tuning_results(target="target_0")
        assert result is not None
        assert result["best_params"] == sample_params

    def test_load_wrong_target_returns_none(self, task, sample_params, sample_lags):
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        result = task.load_tuning_results(target="target_99")
        assert result is None

    def test_load_wrong_data_frame_name(self, tmp_path, sample_params, sample_lags):
        task_a = LazyTask(data_frame_name="dataset_A", cache_home=tmp_path)
        task_a.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=sample_lags,
        )
        task_b = LazyTask(data_frame_name="dataset_B", cache_home=tmp_path)
        result = task_b.load_tuning_results(target="target_0")
        assert result is None


# ---------------------------------------------------------------------------
# LazyTask integration with tuned params
# ---------------------------------------------------------------------------


class TestLazyTaskTunedParams:
    """Verify LazyTask.run() applies cached tuning results."""

    def test_run_signature_has_use_tuned_params(self):
        import inspect

        sig = inspect.signature(LazyTask.run)
        assert "use_tuned_params" in sig.parameters
        assert "max_age_days" in sig.parameters

    def test_run_defaults_use_tuned_params_true(self):
        import inspect

        sig = inspect.signature(LazyTask.run)
        assert sig.parameters["use_tuned_params"].default is True

    def test_run_defaults_max_age_days_none(self):
        import inspect

        sig = inspect.signature(LazyTask.run)
        assert sig.parameters["max_age_days"].default is None


# ---------------------------------------------------------------------------
# BaseTask method availability
# ---------------------------------------------------------------------------


class TestMethodAvailability:
    """Verify save/load methods exist on all task classes."""

    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask],
        ids=["BaseTask", "LazyTask"],
    )
    def test_has_save_tuning_results(self, cls):
        assert hasattr(cls, "save_tuning_results")
        assert callable(getattr(cls, "save_tuning_results"))

    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask],
        ids=["BaseTask", "LazyTask"],
    )
    def test_has_load_tuning_results(self, cls):
        assert hasattr(cls, "load_tuning_results")
        assert callable(getattr(cls, "load_tuning_results"))

    def test_optuna_task_has_save(self):
        from spotforecast2.manager.multitask import OptunaTask

        assert hasattr(OptunaTask, "save_tuning_results")

    def test_spotoptim_task_has_save(self):
        from spotforecast2.manager.multitask import SpotOptimTask

        assert hasattr(SpotOptimTask, "save_tuning_results")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_save_creates_tuning_results_dir(self, tmp_path):
        """The tuning_results subdirectory is created automatically."""
        task = LazyTask(data_frame_name="test_data", cache_home=tmp_path)
        tuning_dir = tmp_path / "tuning_results"
        assert not tuning_dir.exists()
        task.save_tuning_results(
            target="t", task_name="optuna", best_params={}, best_lags=24
        )
        assert tuning_dir.exists()

    def test_save_empty_params(self, task):
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params={},
            best_lags=24,
        )
        with open(path) as f:
            data = json.load(f)
        assert data["best_params"] == {}

    def test_save_nested_list_lags(self, task, sample_params):
        nested_lags = [1, 2, [24, 48]]
        path = task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params=sample_params,
            best_lags=nested_lags,
        )
        with open(path) as f:
            data = json.load(f)
        assert data["best_lags"] == [1, 2, [24, 48]]

    def test_multiple_targets_independent(self, task, sample_params):
        task.save_tuning_results(
            target="target_0",
            task_name="optuna",
            best_params={"lr": 0.01},
            best_lags=24,
        )
        task.save_tuning_results(
            target="target_1",
            task_name="optuna",
            best_params={"lr": 0.05},
            best_lags=48,
        )

        r0 = task.load_tuning_results(target="target_0")
        r1 = task.load_tuning_results(target="target_1")

        assert r0["best_params"]["lr"] == 0.01
        assert r1["best_params"]["lr"] == 0.05
        assert r0["best_lags"] == 24
        assert r1["best_lags"] == 48
