# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Additional pytest tests for spotforecast2.manager.multitask.runner.run().

Covers behaviors complementary to test_runner.py:

- show=True/False forwarded to mt.run()
- plot_with_outliers=True/False controls mt.plot_with_outliers() call
- cache_data=True with no explicit cache_home triggers warning and
  auto-resolves to get_cache_home()
- cache_data=False leaves cache_home as None without printing a warning
- Explicit cache_home is forwarded as-is and suppresses the warning
- Custom agg_weights forwarded to MultiTask constructor
- Scalar parameters n_trials_optuna, train_days, val_days, show_progress,
  verbose, and log_level forwarded to MultiTask constructor
- Returned DataFrame has exactly one column named "forecast"
- Returned DataFrame index matches the future_pred Series index
- Returned forecast values match the underlying future_pred values
- clean task returns an empty DataFrame of type pd.DataFrame
- forecast column dtype is numeric
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spotforecast2.manager.multitask.runner import _DEFAULT_AGG_WEIGHTS, run

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_FUTURE_PRED = pd.Series(
    [10.0, 20.0, 30.0],
    index=pd.date_range("2025-06-01", periods=3, freq="h", tz="UTC"),
    name="future_pred",
)

_DUMMY_DF = pd.DataFrame({"DateTime": ["2025-01-01"], "target": [0.0]})


def _mock_mt(future_pred: pd.Series = _FUTURE_PRED) -> MagicMock:
    """Return a MultiTask mock whose run() returns a minimal agg package.

    Args:
        future_pred: Series to embed in the mocked run() return value.

    Returns:
        Configured MagicMock for MultiTask.
    """
    mt = MagicMock()
    mt.run.return_value = {"future_pred": future_pred}
    return mt


# ---------------------------------------------------------------------------
# show parameter
# ---------------------------------------------------------------------------


class TestShowParameter:
    """Tests that the show flag is forwarded correctly to mt.run()."""

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_show_true_passed_to_run(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", show=True)
        mt.run.assert_called_once_with(show=True)

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_show_false_passed_to_run(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", show=False)
        mt.run.assert_called_once_with(show=False)

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_show_default_is_false(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy")
        mt.run.assert_called_once_with(show=False)


# ---------------------------------------------------------------------------
# plot_with_outliers parameter
# ---------------------------------------------------------------------------


class TestPlotWithOutliers:
    """Tests that plot_with_outliers controls mt.plot_with_outliers() calls."""

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_plot_with_outliers_true_calls_method(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", plot_with_outliers=True)
        mt.plot_with_outliers.assert_called_once()

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_plot_with_outliers_false_does_not_call_method(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", plot_with_outliers=False)
        mt.plot_with_outliers.assert_not_called()

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_plot_with_outliers_default_is_off(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy")
        mt.plot_with_outliers.assert_not_called()

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_plot_with_outliers_clean_task_never_called(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        # clean task uses a different code path; plot_with_outliers must be ignored
        run(_DUMMY_DF, task="clean", plot_with_outliers=True)
        mt.plot_with_outliers.assert_not_called()


# ---------------------------------------------------------------------------
# cache_data / cache_home interaction
# ---------------------------------------------------------------------------


class TestCacheDataBehavior:
    """Tests auto-resolution and forwarding of cache_home."""

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_cache_data_true_no_home_uses_get_cache_home(self, MockMT):
        from spotforecast2_safe.data.fetch_data import get_cache_home

        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", cache_data=True, cache_home=None)
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == get_cache_home()

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_cache_data_true_no_home_prints_warning(self, MockMT, capsys):
        MockMT.return_value = _mock_mt()
        run(_DUMMY_DF, task="lazy", cache_data=True, cache_home=None)
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_cache_data_false_no_warning_printed(self, MockMT, capsys):
        MockMT.return_value = _mock_mt()
        run(_DUMMY_DF, task="lazy", cache_data=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_cache_data_false_cache_home_none_forwarded(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", cache_data=False, cache_home=None)
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] is None

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_explicit_cache_home_forwarded_as_is(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", cache_data=True, cache_home="/my/cache")
        _, kwargs = MockMT.call_args
        assert kwargs["cache_home"] == "/my/cache"

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_explicit_cache_home_suppresses_warning(self, MockMT, capsys):
        MockMT.return_value = _mock_mt()
        run(_DUMMY_DF, task="lazy", cache_data=True, cache_home="/my/cache")
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Custom agg_weights
# ---------------------------------------------------------------------------


class TestAggWeights:
    """Tests custom and default agg_weights forwarding."""

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_custom_agg_weights_forwarded(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        custom_weights = [2.0] * 11
        run(_DUMMY_DF, task="lazy", agg_weights=custom_weights)
        _, kwargs = MockMT.call_args
        assert kwargs["agg_weights"] == custom_weights

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_none_agg_weights_uses_default(self, MockMT):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task="lazy", agg_weights=None)
        _, kwargs = MockMT.call_args
        assert kwargs["agg_weights"] == _DEFAULT_AGG_WEIGHTS


# ---------------------------------------------------------------------------
# Scalar parameter forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["lazy", "optuna", "predict"])
class TestScalarParameterForwarding:
    """Tests that scalar constructor params are forwarded for pipeline tasks."""

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_n_trials_optuna_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, n_trials_optuna=50)
        _, kwargs = MockMT.call_args
        assert kwargs["n_trials_optuna"] == 50

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_train_days_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, train_days=180)
        _, kwargs = MockMT.call_args
        assert kwargs["train_days"] == 180

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_val_days_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, val_days=14)
        _, kwargs = MockMT.call_args
        assert kwargs["val_days"] == 14

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_show_progress_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, show_progress=True)
        _, kwargs = MockMT.call_args
        assert kwargs["show_progress"] is True

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_verbose_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, verbose=True)
        _, kwargs = MockMT.call_args
        assert kwargs["verbose"] is True

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_log_level_forwarded(self, MockMT, task):
        mt = _mock_mt()
        MockMT.return_value = mt
        run(_DUMMY_DF, task=task, log_level=10)
        _, kwargs = MockMT.call_args
        assert kwargs["log_level"] == 10


# ---------------------------------------------------------------------------
# Return value properties
# ---------------------------------------------------------------------------


class TestReturnValueProperties:
    """Tests the shape, type, and content of the returned DataFrame."""

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_return_has_exactly_one_column(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(_DUMMY_DF, task="lazy")
        assert len(result.columns) == 1

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_return_column_name_is_forecast(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(_DUMMY_DF, task="lazy")
        assert result.columns[0] == "forecast"

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_return_index_matches_future_pred(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(_DUMMY_DF, task="lazy")
        pd.testing.assert_index_equal(result.index, _FUTURE_PRED.index)

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_return_values_match_future_pred(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(_DUMMY_DF, task="lazy")
        assert list(result["forecast"]) == list(_FUTURE_PRED.values)

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_forecast_column_is_numeric(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(_DUMMY_DF, task="lazy")
        assert pd.api.types.is_numeric_dtype(result["forecast"])

    @patch("spotforecast2.manager.multitask.runner.MultiTask")
    def test_clean_returns_empty_dataframe(self, MockMT):
        MockMT.return_value = _mock_mt()
        result = run(_DUMMY_DF, task="clean")
        assert isinstance(result, pd.DataFrame)
        assert result.empty
