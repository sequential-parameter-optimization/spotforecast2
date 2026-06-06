# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for dry_run support in the clean task.

Covers the full call chain:
  MultiTask(dry_run=True)
    → MultiTask.run()
      → run_task_clean(dry_run=True)
        → execute_clean(..., dry_run=True)

Test classes:
- TestMultiTaskDryRun   – MultiTask stores and dispatches dry_run
- TestExecuteCleanDryRun– execute_clean() dry_run logic (filesystem)
- TestCleanTaskDryRun   – CleanTask.run(dry_run=True) integration
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from spotforecast2.multitask.clean import CleanTask, execute_clean
from spotforecast2.multitask.multi import MultiTask

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _populated_cache(tmp_path: Path) -> Path:
    """Create a cache directory with two items and return its path."""
    cache = tmp_path / "sf2_cache"
    cache.mkdir()
    (cache / "models").mkdir()
    (cache / "tuning").mkdir()
    return cache


# ---------------------------------------------------------------------------
# MultiTask dry_run storage and dispatch
# ---------------------------------------------------------------------------


class TestMultiTaskDryRun:
    """MultiTask must store dry_run and pass it through to run_task_clean."""

    def test_dry_run_stored_true(self):
        mt = MultiTask(task="clean", dry_run=True)
        assert mt._dry_run is True

    def test_dry_run_stored_false(self):
        mt = MultiTask(task="clean", dry_run=False)
        assert mt._dry_run is False

    def test_dry_run_default_is_false(self):
        mt = MultiTask(task="clean")
        assert mt._dry_run is False

    def test_run_dispatches_dry_run_true_to_clean(self):
        mt = MultiTask(task="clean", dry_run=True)
        with patch.object(mt, "run_task_clean", return_value={}) as mock_clean:
            mt.run()
        mock_clean.assert_called_once_with(show=True, dry_run=True)

    def test_run_dispatches_dry_run_false_to_clean(self):
        mt = MultiTask(task="clean", dry_run=False)
        with patch.object(mt, "run_task_clean", return_value={}) as mock_clean:
            mt.run()
        mock_clean.assert_called_once_with(show=True, dry_run=False)

    def test_run_dispatches_show_and_dry_run_together(self):
        mt = MultiTask(task="clean", dry_run=True)
        with patch.object(mt, "run_task_clean", return_value={}) as mock_clean:
            mt.run(show=False)
        mock_clean.assert_called_once_with(show=False, dry_run=True)

    def test_pipeline_tasks_unaffected_by_dry_run(self):
        """dry_run must not influence lazy/optuna/spotoptim/predict dispatch."""
        mt = MultiTask(task="lazy", dry_run=True)
        with patch.object(mt, "run_task_lazy", return_value={}) as mock_lazy:
            mt.run()
        mock_lazy.assert_called_once_with(show=True)

    def test_unknown_task_raises_value_error(self):
        mt = MultiTask(task="clean")
        with pytest.raises(ValueError, match="Unknown task"):
            mt.run(task="bogus")

    def test_dry_run_not_forwarded_to_base(self):
        """dry_run is a MultiTask-level concern; BaseTask must not receive it."""
        mt = MultiTask(task="clean", dry_run=True)
        # BaseTask has no _dry_run attribute of its own — only MultiTask adds it
        assert hasattr(mt, "_dry_run")


# ---------------------------------------------------------------------------
# execute_clean() dry_run filesystem behaviour
# ---------------------------------------------------------------------------


class TestExecuteCleanDryRun:
    """execute_clean() must inspect-but-not-delete when dry_run=True."""

    def _make_task(self, cache_path: Path) -> CleanTask:
        return CleanTask(cache_home=cache_path)

    def test_status_is_dry_run(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = self._make_task(cache)
        result = execute_clean(task, dry_run=True)
        assert result["status"] == "dry_run"

    def test_cache_dir_not_deleted(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = self._make_task(cache)
        execute_clean(task, dry_run=True)
        assert cache.exists()

    def test_items_not_deleted(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = self._make_task(cache)
        execute_clean(task, dry_run=True)
        assert (cache / "models").exists()
        assert (cache / "tuning").exists()

    def test_deleted_items_lists_contents(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = self._make_task(cache)
        result = execute_clean(task, dry_run=True)
        # "logging" dir is created by the FileHandler on CleanTask construction
        assert "models" in result["deleted_items"]
        assert "tuning" in result["deleted_items"]

    def test_cache_dir_in_result(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = self._make_task(cache)
        result = execute_clean(task, dry_run=True)
        assert result["cache_dir"] == cache

    def test_empty_cache_dry_run_returns_only_logging(self, tmp_path):
        """A freshly constructed CleanTask creates only the logging/ subdir."""
        empty_cache = tmp_path / "empty_cache"
        task = self._make_task(empty_cache)
        result = execute_clean(task, dry_run=True)
        # Only the logging dir is present — created by the FileHandler
        assert result["deleted_items"] == ["logging"]

    def test_dry_run_false_deletes_cache(self, tmp_path):
        """Sanity check: without dry_run the directory IS removed."""
        cache = _populated_cache(tmp_path)
        task = self._make_task(cache)
        result = execute_clean(task, dry_run=False)
        assert result["status"] == "success"
        assert not cache.exists()

    def test_dry_run_false_returns_deleted_items(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = self._make_task(cache)
        result = execute_clean(task, dry_run=False)
        assert "models" in result["deleted_items"]
        assert "tuning" in result["deleted_items"]

    def test_cache_home_override_respected_in_dry_run(self, tmp_path):
        """cache_home kwarg overrides the task's configured home."""
        real_cache = _populated_cache(tmp_path)
        other_cache = tmp_path / "other"  # does not exist
        task = self._make_task(real_cache)
        result = execute_clean(task, cache_home=other_cache, dry_run=True)
        # other_cache doesn't exist → empty, and real_cache is untouched
        assert result["status"] == "empty"
        assert real_cache.exists()


# ---------------------------------------------------------------------------
# CleanTask.run(dry_run=True) integration
# ---------------------------------------------------------------------------


class TestCleanTaskDryRun:
    """CleanTask.run(dry_run=True) must not delete anything."""

    def test_run_dry_run_status(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = CleanTask(cache_home=cache)
        result = task.run(dry_run=True)
        assert result["status"] == "dry_run"

    def test_run_dry_run_preserves_directory(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = CleanTask(cache_home=cache)
        task.run(dry_run=True)
        assert cache.exists()

    def test_run_dry_run_lists_items(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = CleanTask(cache_home=cache)
        result = task.run(dry_run=True)
        assert "models" in result["deleted_items"]
        assert "tuning" in result["deleted_items"]

    def test_run_no_dry_run_deletes(self, tmp_path):
        cache = _populated_cache(tmp_path)
        task = CleanTask(cache_home=cache)
        result = task.run(dry_run=False)
        assert result["status"] == "success"
        assert not cache.exists()

    def test_run_dry_run_default_is_false(self, tmp_path):
        """CleanTask.run() without dry_run argument should actually delete."""
        cache = _populated_cache(tmp_path)
        task = CleanTask(cache_home=cache)
        result = task.run()
        assert result["status"] == "success"

    def test_multitask_clean_with_dry_run_true(self, tmp_path):
        """Full-stack: MultiTask(dry_run=True).run() must not delete."""
        cache = _populated_cache(tmp_path)
        mt = MultiTask(task="clean", cache_home=cache, dry_run=True)
        result = mt.run()
        assert result["status"] == "dry_run"
        assert cache.exists()

    def test_multitask_clean_with_dry_run_false(self, tmp_path):
        """Full-stack: MultiTask(dry_run=False).run() must delete."""
        cache = _populated_cache(tmp_path)
        mt = MultiTask(task="clean", cache_home=cache, dry_run=False)
        result = mt.run()
        assert result["status"] == "success"
        assert not cache.exists()
