# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for CleanTask and execute_clean."""

import importlib
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# Helpers
# ===========================================================================


def _make_cache_dir(tmp_path: Path) -> Path:
    """Create a cache directory with typical pipeline subdirectories."""
    cache = tmp_path / "sf2_cache"
    cache.mkdir()
    (cache / "models").mkdir()
    (cache / "tuning_results").mkdir()
    (cache / "unified_pipeline").mkdir()
    return cache


def _make_task(cache_home: Path) -> "CleanTask":  # noqa: F821 – resolved at runtime
    from spotforecast2.manager.multitask.clean import CleanTask

    return CleanTask(cache_home=cache_home)


def _make_mock_task(cache_home: Path) -> MagicMock:
    """Create a MagicMock task to avoid BaseTask.__init__ side-effects."""
    task = MagicMock()
    task.config.cache_home = cache_home
    return task


# ===========================================================================
# 1. execute_clean – directory does not exist
# ===========================================================================


class TestExecuteCleanMissingDir:
    """execute_clean must return status 'empty' when cache does not exist."""

    def test_missing_dir_returns_empty_status(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        missing = tmp_path / "nonexistent_cache"
        task = _make_mock_task(missing)
        result = execute_clean(task)

        assert result["status"] == "empty"

    def test_missing_dir_deleted_items_is_empty_list(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        missing = tmp_path / "nonexistent_cache"
        task = _make_mock_task(missing)
        result = execute_clean(task)

        assert result["deleted_items"] == []

    def test_missing_dir_returns_correct_cache_dir(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        missing = tmp_path / "nonexistent_cache"
        task = _make_mock_task(missing)
        result = execute_clean(task)

        assert result["cache_dir"] == missing

    def test_missing_dir_does_not_raise(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        missing = tmp_path / "nonexistent_cache"
        task = _make_mock_task(missing)
        # Must not raise any exception
        execute_clean(task)


# ===========================================================================
# 2. execute_clean – successful removal
# ===========================================================================


class TestExecuteCleanSuccess:
    """execute_clean must delete the cache directory and return 'success'."""

    def test_success_status(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        result = execute_clean(task)

        assert result["status"] == "success"

    def test_directory_removed_after_clean(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        execute_clean(task)

        assert not cache.exists()

    def test_deleted_items_contains_subdirs(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        result = execute_clean(task)

        assert "models" in result["deleted_items"]
        assert "tuning_results" in result["deleted_items"]
        assert "unified_pipeline" in result["deleted_items"]

    def test_prints_success(self, tmp_path, capsys):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        execute_clean(task)

        captured = capsys.readouterr()
        assert "success" in captured.out

    def test_returns_correct_cache_dir(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        result = execute_clean(task)

        assert result["cache_dir"] == cache

    def test_cache_home_override_parameter(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        # task configured with one path, override points to a different dir
        other = tmp_path / "other_cache"
        other.mkdir()
        (other / "models").mkdir()

        task = _make_task(tmp_path / "unused_cache")
        result = execute_clean(task, cache_home=other)

        assert result["status"] == "success"
        assert not other.exists()


# ===========================================================================
# 3. execute_clean – dry_run mode
# ===========================================================================


class TestExecuteCleanDryRun:
    """execute_clean with dry_run=True must not delete anything."""

    def test_dry_run_returns_dry_run_status(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        result = execute_clean(task, dry_run=True)

        assert result["status"] == "dry_run"

    def test_dry_run_does_not_remove_directory(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        execute_clean(task, dry_run=True)

        assert cache.exists()

    def test_dry_run_deleted_items_lists_contents(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)
        result = execute_clean(task, dry_run=True)

        assert "models" in result["deleted_items"]
        assert "tuning_results" in result["deleted_items"]

    def test_dry_run_missing_dir_returns_empty(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        missing = tmp_path / "nonexistent"
        # Use a mock task to avoid BaseTask.__init__ creating the directory
        task = _make_mock_task(missing)
        result = execute_clean(task, dry_run=True)

        assert result["status"] == "empty"
        assert result["deleted_items"] == []


# ===========================================================================
# 4. execute_clean – error handling
# ===========================================================================


class TestExecuteCleanErrors:
    """execute_clean must raise RuntimeError on OS-level failures."""

    def test_oserror_raises_runtime_error(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)

        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            with pytest.raises(RuntimeError, match="Could not clean cache directory"):
                execute_clean(task)

    def test_runtime_error_message_contains_path(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)

        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            with pytest.raises(RuntimeError) as exc_info:
                execute_clean(task)

        assert str(cache) in str(exc_info.value)

    def test_runtime_error_chained_from_oserror(self, tmp_path):
        from spotforecast2.manager.multitask.clean import execute_clean

        cache = _make_cache_dir(tmp_path)
        task = _make_task(cache)

        with patch("shutil.rmtree", side_effect=OSError("mock error")):
            with pytest.raises(RuntimeError) as exc_info:
                execute_clean(task)

        assert isinstance(exc_info.value.__cause__, OSError)


# ===========================================================================
# 5. CleanTask class properties
# ===========================================================================


class TestCleanTaskClass:
    """CleanTask must follow the expected class conventions."""

    def test_task_name_is_clean(self):
        from spotforecast2.manager.multitask.clean import CleanTask

        assert CleanTask._task_name == "clean"

    def test_inherits_from_base_task(self):
        from spotforecast2.manager.multitask.base import BaseTask
        from spotforecast2.manager.multitask.clean import CleanTask

        assert issubclass(CleanTask, BaseTask)

    def test_has_run_method(self):
        from spotforecast2.manager.multitask.clean import CleanTask

        assert hasattr(CleanTask, "run")
        assert callable(CleanTask.run)

    def test_auto_save_models_default_true(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        task = CleanTask(cache_home=tmp_path)
        assert task.auto_save_models is True

    def test_auto_save_models_can_be_disabled(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        task = CleanTask(cache_home=tmp_path, auto_save_models=False)
        assert task.auto_save_models is False


# ===========================================================================
# 6. CleanTask.run() delegation to execute_clean
# ===========================================================================


class TestCleanTaskRun:
    """CleanTask.run() must delegate to execute_clean correctly."""

    def test_run_returns_success_when_cache_exists(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        cache = _make_cache_dir(tmp_path)
        task = CleanTask(cache_home=cache)
        result = task.run()

        assert result["status"] == "success"

    def test_run_deletes_cache_dir(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        cache = _make_cache_dir(tmp_path)
        task = CleanTask(cache_home=cache)
        task.run()

        assert not cache.exists()

    def test_run_dry_run_does_not_delete(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        cache = _make_cache_dir(tmp_path)
        task = CleanTask(cache_home=cache)
        result = task.run(dry_run=True)

        assert result["status"] == "dry_run"
        assert cache.exists()

    def test_run_accepts_show_kwarg_without_error(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        # BaseTask.__init__ creates the dir (with logging/ subdir); remove it entirely
        cache = tmp_path / "does_not_exist"
        task = CleanTask(cache_home=cache)
        shutil.rmtree(cache)  # remove the dir (and logging/ subdir) that init created
        result = task.run(show=True)
        assert result["status"] == "empty"

    def test_run_cache_home_override(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        cache = _make_cache_dir(tmp_path)
        # Task configured to a different (non-existent) path
        task = CleanTask(cache_home=tmp_path / "other")
        result = task.run(cache_home=cache)

        assert result["status"] == "success"
        assert not cache.exists()

    def test_run_returns_empty_for_missing_dir(self, tmp_path):
        from spotforecast2.manager.multitask.clean import CleanTask

        # BaseTask.__init__ creates the dir (with logging/ subdir); remove it entirely
        cache = tmp_path / "no_cache_here"
        task = CleanTask(cache_home=cache)
        shutil.rmtree(cache)  # remove the dir (and logging/ subdir) that init created
        result = task.run()

        assert result["status"] == "empty"


# ===========================================================================
# 7. CleanTask imports
# ===========================================================================


class TestCleanTaskImports:
    """CleanTask must be importable from all expected public paths."""

    def test_importable_from_multitask(self):
        mod = importlib.import_module("spotforecast2.manager.multitask")
        assert hasattr(mod, "CleanTask")

    def test_importable_from_manager(self):
        mod = importlib.import_module("spotforecast2.manager")
        assert hasattr(mod, "CleanTask")

    def test_in_multitask_all(self):
        mod = importlib.import_module("spotforecast2.manager.multitask")
        assert "CleanTask" in mod.__all__

    def test_in_manager_all(self):
        mod = importlib.import_module("spotforecast2.manager")
        assert "CleanTask" in mod.__all__


# ===========================================================================
# 8. MultiTask dispatcher for "clean"
# ===========================================================================


class TestMultiTaskCleanDispatch:
    """MultiTask must accept 'clean' as a valid task mode."""

    def test_multitask_has_run_task_clean(self):
        from spotforecast2.manager.multitask.multi import MultiTask

        mt = MultiTask.__new__(MultiTask)
        assert hasattr(mt, "run_task_clean")
        assert callable(mt.run_task_clean)

    def test_run_task_clean_returns_empty_for_missing_cache(self, tmp_path):
        from spotforecast2.manager.multitask.multi import MultiTask

        # MultiTask.__init__ creates the dir (with logging/ subdir); remove it entirely
        cache = tmp_path / "no_cache"
        mt = MultiTask(cache_home=cache, predict_size=24)
        shutil.rmtree(cache)  # remove the dir (and logging/ subdir) that init created
        result = mt.run_task_clean()

        assert result["status"] == "empty"

    def test_run_task_clean_dry_run(self, tmp_path):
        from spotforecast2.manager.multitask.multi import MultiTask

        cache = _make_cache_dir(tmp_path)
        mt = MultiTask(cache_home=cache, predict_size=24)
        result = mt.run_task_clean(dry_run=True)

        assert result["status"] == "dry_run"
        assert cache.exists()

    def test_run_dispatches_clean_via_run_method(self, tmp_path):
        from spotforecast2.manager.multitask.multi import MultiTask

        # MultiTask.__init__ creates the dir (with logging/ subdir); remove it entirely
        cache = tmp_path / "no_cache"
        mt = MultiTask(task="clean", cache_home=cache, predict_size=24)
        shutil.rmtree(cache)  # remove the dir (and logging/ subdir) that init created
        result = mt.run(show=False)

        assert result["status"] == "empty"
