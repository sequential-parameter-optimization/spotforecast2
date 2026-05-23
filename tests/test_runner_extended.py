# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Additional pytest tests for ``spotforecast2.multitask.runner.run``.

Complementary to ``test_runner.py``.  Covers behaviours that are not
about the construction-arg shape but about runner-level side effects:

- ``show`` is forwarded to ``mt.run()``.
- ``plot_with_outliers`` controls the optional pipeline step.
- ``cache_home`` auto-resolves via ``get_cache_home()`` and is forwarded.
- ``**overrides`` are threaded through to the ``MultiTask`` kwargs.
- The returned DataFrame shape / type / column / index / values match
  ``future_pred``.

After the config-object refactor, callers pass a ``ConfigMulti`` (or
``make_demo10_config()`` for the 11-target demo presets) as the positional
``config`` argument.  ``dataframe`` is a keyword-only argument.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spotforecast2.multitask.runner import _DEMO10_AGG_WEIGHTS, make_demo10_config, run

_FUTURE_PRED = pd.Series(
    [10.0, 20.0, 30.0],
    index=pd.date_range("2025-06-01", periods=3, freq="h", tz="UTC"),
    name="future_pred",
)

_DUMMY_DF = pd.DataFrame({"DateTime": ["2025-01-01"], "target": [0.0]})


def _mock_mt(future_pred: pd.Series = _FUTURE_PRED) -> MagicMock:
    mt = MagicMock()
    mt.run.return_value = {"future_pred": future_pred}
    return mt


class TestShowParameter:
    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_show_true_passed_to_run(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF, show=True)
        mt.run.assert_called_once_with(show=True)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_show_false_passed_to_run(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF, show=False)
        mt.run.assert_called_once_with(show=False)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_show_default_is_false(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF)
        mt.run.assert_called_once_with(show=False)


class TestPlotWithOutliers:
    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_true_calls_method(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF, plot_with_outliers=True)
        mt.plot_with_outliers.assert_called_once()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_false_does_not_call_method(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF, plot_with_outliers=False)
        mt.plot_with_outliers.assert_not_called()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_default_is_off(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF)
        mt.plot_with_outliers.assert_not_called()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_clean_task_never_called(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="clean", plot_with_outliers=True)
        mt.plot_with_outliers.assert_not_called()


class TestCacheHomeBehavior:
    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_none_auto_resolves_for_pipeline_task(self, MockMT):
        from spotforecast2_safe.data.fetch_data import get_cache_home

        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF, cache_home=None)
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == get_cache_home()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_none_auto_resolves_for_clean_task(self, MockMT):
        from spotforecast2_safe.data.fetch_data import get_cache_home

        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="clean", cache_home=None)
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == get_cache_home()

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_explicit_forwarded_for_pipeline_task(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF, cache_home="/my/cache")
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == "/my/cache"

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_explicit_forwarded_for_clean_task(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="clean", cache_home="/my/cache")
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == "/my/cache"

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_no_warning_when_cache_home_none(self, MockMT, capsys):
        MockMT.return_value = _mock_mt()
        run(task="lazy", dataframe=_DUMMY_DF, cache_home=None)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestDemo10OptIn:
    """The demo10 presets are no longer auto-injected; callers opt in."""

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_default_config_does_not_carry_demo10_weights(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task="lazy", dataframe=_DUMMY_DF)
        args, _ = MockMT.call_args
        assert args[0].agg_weights is None

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_make_demo10_config_carries_weights(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        cfg = make_demo10_config()
        run(cfg, task="lazy", dataframe=_DUMMY_DF)
        args, _ = MockMT.call_args
        assert args[0].agg_weights == _DEMO10_AGG_WEIGHTS


@pytest.mark.parametrize("task", ["lazy", "optuna", "predict"])
class TestOverrideForwarding:
    """``**overrides`` flow through to ``MultiTask`` kwargs verbatim."""

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_n_trials_optuna_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task=task, dataframe=_DUMMY_DF, n_trials_optuna=50)
        _, kwargs = MockMT.call_args
        assert kwargs["n_trials_optuna"] == 50

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_train_size_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task=task, dataframe=_DUMMY_DF, train_size=pd.Timedelta(days=180))
        _, kwargs = MockMT.call_args
        assert kwargs["train_size"] == pd.Timedelta(days=180)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_show_progress_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task=task, dataframe=_DUMMY_DF, show_progress=True)
        _, kwargs = MockMT.call_args
        assert kwargs["show_progress"] is True

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_log_level_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(task=task, dataframe=_DUMMY_DF, log_level=10)
        _, kwargs = MockMT.call_args
        assert kwargs["log_level"] == 10


class TestReturnValueProperties:
    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_return_has_exactly_one_column(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(task="lazy", dataframe=_DUMMY_DF)
        assert len(result.columns) == 1

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_return_column_name_is_forecast(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(task="lazy", dataframe=_DUMMY_DF)
        assert result.columns[0] == "forecast"

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_return_index_matches_future_pred(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(task="lazy", dataframe=_DUMMY_DF)
        pd.testing.assert_index_equal(result.index, _FUTURE_PRED.index)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_return_values_match_future_pred(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(task="lazy", dataframe=_DUMMY_DF)
        assert list(result["forecast"]) == list(_FUTURE_PRED.values)

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_forecast_column_is_numeric(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(task="lazy", dataframe=_DUMMY_DF)
        assert pd.api.types.is_numeric_dtype(result["forecast"])

    @patch("spotforecast2.multitask.runner.MultiTask")
    def test_clean_returns_empty_dataframe(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(task="clean")
        assert isinstance(result, pd.DataFrame)
        assert result.empty
