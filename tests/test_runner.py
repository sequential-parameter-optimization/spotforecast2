# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for spotforecast2.manager.multitask.runner.run().

Covers:
- Import paths (from runner module, from multitask package, from manager package)
- Function signature (parameters and defaults)
- ValueError on unknown task
- clean task: MultiTask constructed with task="clean", run() called, empty DataFrame returned
- pipeline tasks (lazy, optuna, spotoptim, predict): full pipeline sequence called
- Default bounds used when bounds=None
- Custom bounds forwarded to MultiTask
- agg_weights forwarded for pipeline tasks but not for clean
- **kwargs forwarded to MultiTask
- Return type is always pd.DataFrame
- Return column is "forecast" for pipeline tasks
- Return is empty DataFrame for clean task
"""

import inspect
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spotforecast2.manager.multitask.runner import (
    _ALL_TASKS,
    _DEFAULT_AGG_WEIGHTS,
    _DEFAULT_BOUNDS,
    run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FUTURE_PRED = pd.Series(
    [1.0, 2.0, 3.0],
    index=pd.date_range("2025-01-02", periods=3, freq="h", tz="UTC"),
    name="forecast",
)

_DUMMY_DF = pd.DataFrame({"DateTime": ["2025-01-01"], "target": [0.0]})


def _mock_mt(future_pred: pd.Series = _FUTURE_PRED) -> MagicMock:
    """Return a MultiTask mock whose run() returns a minimal agg package."""
    mt = MagicMock()
    mt.run.return_value = {"future_pred": future_pred}
    return mt


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


class TestImportPaths:
    def test_importable_from_runner_module(self):
        from spotforecast2.manager.multitask.runner import run as _run

        assert callable(_run)

    def test_importable_from_multitask_package(self):
        from spotforecast2.manager.multitask import run as _run

        assert callable(_run)

    def test_importable_from_manager_package(self):
        from spotforecast2.manager import run as _run

        assert callable(_run)

    def test_same_object_across_import_paths(self):
        from spotforecast2.manager import run as r1
        from spotforecast2.manager.multitask import run as r2
        from spotforecast2.manager.multitask.runner import run as r3

        assert r1 is r2 is r3


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


class TestSignature:
    def _sig(self):
        return inspect.signature(run)

    def test_has_dataframe_param(self):
        assert "dataframe" in self._sig().parameters

    def test_has_task_param(self):
        assert "task" in self._sig().parameters

    def test_has_bounds_param(self):
        assert "bounds" in self._sig().parameters

    def test_has_data_frame_name_param(self):
        assert "data_frame_name" in self._sig().parameters

    def test_has_kwargs(self):
        p = self._sig().parameters
        assert any(v.kind == inspect.Parameter.VAR_KEYWORD for v in p.values())

    def test_task_default_is_lazy(self):
        assert self._sig().parameters["task"].default == "lazy"

    def test_bounds_default_is_none(self):
        assert self._sig().parameters["bounds"].default is None

    def test_data_frame_name_default(self):
        assert self._sig().parameters["data_frame_name"].default == "demo10"


# ---------------------------------------------------------------------------
# ValueError on unknown task
# ---------------------------------------------------------------------------


class TestUnknownTask:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown task"):
            run(_DUMMY_DF, task="bogus")

    def test_error_message_contains_task_name(self):
        with pytest.raises(ValueError, match="bogus"):
            run(_DUMMY_DF, task="bogus")

    def test_all_valid_tasks_do_not_raise(self):
        for task in _ALL_TASKS:
            with patch(
                "spotforecast2.manager.multitask.runner.MultiTask",
                return_value=_mock_mt(),
            ):
                # Should not raise ValueError
                run(_DUMMY_DF, task=task)


# ---------------------------------------------------------------------------
# clean task
# ---------------------------------------------------------------------------


class TestCleanTask:
    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_multitask_constructed_with_clean(self, MockMT):
        MockMT.return_value = _mock_mt()
        run(_DUMMY_DF, task="clean", data_frame_name="mydata")
        MockMT.assert_called_once_with(
            task="clean",
            dataframe=_DUMMY_DF,
            data_frame_name="mydata",
        )

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_run_called_once(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="clean")
        mt.run.assert_called_once()

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_prepare_data_not_called(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="clean")
        mt.prepare_data.assert_not_called()

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_returns_empty_dataframe(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(_DUMMY_DF, task="clean")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_kwargs_forwarded_to_multitask(self, MockMT):
        MockMT.return_value = _mock_mt()
        run(_DUMMY_DF, task="clean", cache_home="/tmp/cache")
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == "/tmp/cache"

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_agg_weights_not_forwarded_for_clean(self, MockMT):
        MockMT.return_value = _mock_mt()
        run(_DUMMY_DF, task="clean")
        _, kwargs = MockMT.call_args
        assert "agg_weights" not in kwargs

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_bounds_not_forwarded_for_clean(self, MockMT):
        MockMT.return_value = _mock_mt()
        run(_DUMMY_DF, task="clean")
        _, kwargs = MockMT.call_args
        assert "bounds" not in kwargs


# ---------------------------------------------------------------------------
# Pipeline tasks (lazy, optuna, spotoptim, predict)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["lazy", "optuna", "spotoptim", "predict"])
class TestPipelineTasks:
    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_multitask_constructed_with_correct_task(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task)
        _, kwargs = MockMT.call_args
        assert kwargs["task"] == task

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_dataframe_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task)
        _, kwargs = MockMT.call_args
        assert kwargs["dataframe"] is _DUMMY_DF

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_default_agg_weights_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task)
        _, kwargs = MockMT.call_args
        assert kwargs["agg_weights"] == _DEFAULT_AGG_WEIGHTS

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_default_bounds_used_when_none(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, bounds=None)
        _, kwargs = MockMT.call_args
        assert kwargs["bounds"] == _DEFAULT_BOUNDS

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_custom_bounds_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        custom_bounds = [(0, 100)] * 11
        run(_DUMMY_DF, task=task, bounds=custom_bounds)
        _, kwargs = MockMT.call_args
        assert kwargs["bounds"] == custom_bounds

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_pipeline_sequence_called(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task)
        mt.prepare_data.assert_called_once()
        mt.detect_outliers.assert_called_once()
        mt.impute.assert_called_once()
        mt.build_exogenous_features.assert_called_once()

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_run_called_with_show_false(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task)
        mt.run.assert_called_once_with(show=False)

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_returns_dataframe(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        result = run(_DUMMY_DF, task=task)
        assert isinstance(result, pd.DataFrame)

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_return_column_is_forecast(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        result = run(_DUMMY_DF, task=task)
        assert "forecast" in result.columns

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_return_values_match_future_pred(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        result = run(_DUMMY_DF, task=task)
        pd.testing.assert_series_equal(
            result["forecast"],
            _FUTURE_PRED.rename("forecast"),
            check_names=False,
        )

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_kwargs_forwarded_to_multitask(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, predict_size=48, train_days=180)
        _, kwargs = MockMT.call_args
        assert kwargs["predict_size"] == 48
        assert kwargs["train_days"] == 180

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_data_frame_name_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, data_frame_name="mydata")
        _, kwargs = MockMT.call_args
        assert kwargs["data_frame_name"] == "mydata"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_bounds_has_11_entries(self):
        assert len(_DEFAULT_BOUNDS) == 11

    def test_default_bounds_are_tuples(self):
        for entry in _DEFAULT_BOUNDS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2

    def test_default_agg_weights_has_11_entries(self):
        assert len(_DEFAULT_AGG_WEIGHTS) == 11

    def test_all_tasks_contains_expected(self):
        assert _ALL_TASKS == {"lazy", "optuna", "spotoptim", "predict", "clean"}
