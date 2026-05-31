# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for ``spotforecast2.multitask.runner.run``.

After the config-object refactor the public surface is:

    run(config: Optional[PipelineConfig] = None, *, task, dataframe,
        project_name, cache_home, plot_with_outliers, show, show_progress,
        dry_run, log_level, **overrides)

The runner no longer silently injects 11-target demo10 ``bounds`` /
``agg_weights``; callers opt in via ``make_demo10_config()``.
``project_name`` is mapped onto ``config.data_frame_name`` before the
``MultiTask`` is constructed.
"""

import inspect
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from spotforecast2_safe.configurator.config_multi import ConfigMulti

from spotforecast2.multitask.runner import (
    _ALL_TASKS,
    _DEMO10_AGG_WEIGHTS,
    _DEMO10_BOUNDS,
    make_demo10_config,
    run,
)

_FUTURE_PRED = pd.Series(
    [1.0, 2.0, 3.0],
    index=pd.date_range("2025-01-02", periods=3, freq="h", tz="UTC"),
    name="forecast",
)

_DUMMY_DF = pd.DataFrame({"DateTime": ["2025-01-01"], "target": [0.0]})


def _mock_mt(future_pred: pd.Series = _FUTURE_PRED) -> MagicMock:
    mt = MagicMock()
    mt.run.return_value = {"future_pred": future_pred}
    return mt


class TestImportPaths:
    def test_importable_from_runner_module(self):
        from spotforecast2.multitask.runner import run as _run

        assert callable(_run)

    def test_importable_from_multitask_package(self):
        from spotforecast2.multitask import run as _run

        assert callable(_run)

    def test_same_object_across_import_paths(self):
        from spotforecast2.multitask import run as r1
        from spotforecast2.multitask.runner import run as r2

        assert r1 is r2

    def test_make_demo10_config_exported(self):
        from spotforecast2.multitask import make_demo10_config as _mk

        assert callable(_mk)


class TestSignature:
    def _sig(self):
        return inspect.signature(run)

    def test_has_config_param(self):
        assert "config" in self._sig().parameters

    def test_has_dataframe_param(self):
        assert "dataframe" in self._sig().parameters

    def test_has_task_param(self):
        assert "task" in self._sig().parameters

    def test_has_project_name_param(self):
        assert "project_name" in self._sig().parameters

    def test_has_overrides(self):
        p = self._sig().parameters
        assert any(v.kind == inspect.Parameter.VAR_KEYWORD for v in p.values())

    def test_task_default_is_lazy(self):
        assert self._sig().parameters["task"].default == "lazy"

    def test_config_default_is_none(self):
        assert self._sig().parameters["config"].default is None

    def test_no_legacy_bounds_param(self):
        # bounds now lives on config; runner doesn't carry it directly
        assert "bounds" not in self._sig().parameters

    def test_no_legacy_agg_weights_param(self):
        assert "agg_weights" not in self._sig().parameters

    def test_no_legacy_train_days_param(self):
        assert "train_days" not in self._sig().parameters


class TestUnknownTask:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown task"):
            run(task="bogus", dataframe=_DUMMY_DF)

    def test_error_message_contains_task_name(self):
        with pytest.raises(ValueError, match="bogus"):
            run(task="bogus", dataframe=_DUMMY_DF)

    def test_all_valid_tasks_do_not_raise(self):
        for task in _ALL_TASKS:
            with patch(
                "spotforecast2.multitask.runner.MultiTask",
                return_value=_mock_mt(),
            ):
                run(task=task, dataframe=_DUMMY_DF)


class TestCleanTask:
    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_multitask_constructed_with_clean(self, MockMT):
        from spotforecast2_safe.data.fetch_data import get_cache_home

        MockMT.return_value = _mock_mt()
        run(task="clean", project_name="mydata")
        args, kwargs = MockMT.call_args
        # config is positional
        assert isinstance(args[0], ConfigMulti)
        assert kwargs["task"] == "clean"
        assert kwargs["cache_home"] == get_cache_home()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_project_name_lands_on_config(self, MockMT):
        MockMT.return_value = _mock_mt()
        run(task="clean", project_name="mydata")
        args, _ = MockMT.call_args
        assert args[0].data_frame_name == "mydata"

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_run_called_once(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="clean")
        mt.run.assert_called_once()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_prepare_data_not_called(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="clean")
        mt.prepare_data.assert_not_called()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_returns_empty_dataframe(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(task="clean")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_cache_home_forwarded(self, MockMT):
        MockMT.return_value = _mock_mt()
        run(task="clean", cache_home="/tmp/cache")
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == "/tmp/cache"


@pytest.mark.parametrize("task", ["lazy", "optuna", "spotoptim", "predict"])
class TestPipelineTasks:
    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_multitask_constructed_with_correct_task(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        run(task=task, dataframe=_DUMMY_DF)
        _, kwargs = MockMT.call_args
        assert kwargs["task"] == task

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_config_is_positional(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        run(task=task, dataframe=_DUMMY_DF)
        args, _ = MockMT.call_args
        assert isinstance(args[0], ConfigMulti)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_dataframe_forwarded(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        run(task=task, dataframe=_DUMMY_DF)
        _, kwargs = MockMT.call_args
        assert kwargs["dataframe"] is _DUMMY_DF

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_project_name_lands_on_config(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        run(task=task, dataframe=_DUMMY_DF, project_name="mydata")
        args, _ = MockMT.call_args
        assert args[0].data_frame_name == "mydata"

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_caller_supplied_config_passes_through(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        cfg = make_demo10_config(predict_size=48)
        run(cfg, task=task, dataframe=_DUMMY_DF)
        args, _ = MockMT.call_args
        # The same config object is forwarded; demo10 presets survive.
        assert args[0] is cfg
        assert cfg.bounds == _DEMO10_BOUNDS
        assert cfg.agg_weights == _DEMO10_AGG_WEIGHTS

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_no_silent_demo10_injection(self, MockMT, task):
        """A default ``ConfigMulti`` does NOT inherit demo10 bounds/weights."""
        MockMT.return_value = _mock_mt()
        run(task=task, dataframe=_DUMMY_DF)
        args, _ = MockMT.call_args
        assert args[0].bounds is None
        assert args[0].agg_weights is None

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_pipeline_sequence_called(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task=task, dataframe=_DUMMY_DF)
        mt.prepare_data.assert_called_once()
        mt.detect_outliers.assert_called_once()
        mt.impute.assert_called_once()
        mt.build_exogenous_features.assert_called_once()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_run_called_with_show_false(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task=task, dataframe=_DUMMY_DF)
        mt.run.assert_called_once_with(show=False)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_returns_dataframe(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        result = run(task=task, dataframe=_DUMMY_DF)
        assert isinstance(result, pd.DataFrame)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_return_column_is_forecast(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        result = run(task=task, dataframe=_DUMMY_DF)
        assert "forecast" in result.columns

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_return_values_match_future_pred(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        result = run(task=task, dataframe=_DUMMY_DF)
        pd.testing.assert_series_equal(
            result["forecast"],
            _FUTURE_PRED.rename("forecast"),
            check_names=False,
        )

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_overrides_forwarded_to_multitask(self, MockMT, task):
        MockMT.return_value = _mock_mt()
        run(task=task, dataframe=_DUMMY_DF, predict_size=48)
        _, kwargs = MockMT.call_args
        # **overrides flows through MultiTask kwargs verbatim.
        assert kwargs["predict_size"] == 48


class TestConstants:
    def test_default_bounds_has_11_entries(self):
        assert len(_DEMO10_BOUNDS) == 11

    def test_default_bounds_are_tuples(self):
        for entry in _DEMO10_BOUNDS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2

    def test_default_agg_weights_has_11_entries(self):
        assert len(_DEMO10_AGG_WEIGHTS) == 11

    def test_all_tasks_contains_expected(self):
        assert _ALL_TASKS == {
            "lazy",
            "defaults",
            "optuna",
            "spotoptim",
            "predict",
            "clean",
        }


class TestMakeDemo10Config:
    def test_returns_config_multi(self):
        cfg = make_demo10_config()
        assert isinstance(cfg, ConfigMulti)

    def test_carries_demo10_bounds(self):
        cfg = make_demo10_config()
        assert cfg.bounds == _DEMO10_BOUNDS

    def test_carries_demo10_agg_weights(self):
        cfg = make_demo10_config()
        assert cfg.agg_weights == _DEMO10_AGG_WEIGHTS

    def test_overrides_applied(self):
        cfg = make_demo10_config(predict_size=48, number_folds=5)
        assert cfg.predict_size == 48
        assert cfg.number_folds == 5

    def test_overrides_do_not_clobber_bounds(self):
        cfg = make_demo10_config(predict_size=48)
        assert cfg.bounds == _DEMO10_BOUNDS
