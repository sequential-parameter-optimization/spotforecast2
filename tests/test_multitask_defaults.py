# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``defaults`` task (ADR-002 Step 2).

``DefaultsStrategy`` and ``DefaultsTask`` route ``task="defaults"`` requests
through ``BaseTask._run_strategy``.  The behaviour contract that distinguishes
``defaults`` from ``lazy``:

- ``defaults`` never reads the tuning-result cache;
- ``defaults`` never writes a tuning-result JSON (``save_tuning_results``
  is the search-task path);
- when ``auto_save_models=True``, ``defaults`` saves models under
  ``task_name="defaults"`` so ``PredictTask(task_name="defaults")`` can
  load them.
"""

from unittest.mock import MagicMock, patch

from spotforecast2.multitask import DefaultsTask
from spotforecast2.multitask.defaults import execute_defaults
from spotforecast2.multitask.strategies import DefaultsStrategy


def test_defaults_task_exposes_task_name():
    task = DefaultsTask(predict_size=24)
    assert task.TASK == "defaults"


def test_defaults_strategy_returns_forecaster_unchanged():
    """The strategy is a no-op pre-fit step."""
    sentinel = object()
    strategy = DefaultsStrategy()
    result = strategy.prepare_forecaster(
        task=MagicMock(),
        target="any",
        forecaster=sentinel,
        y_train=None,
        exog_train=None,
    )
    assert result is sentinel


def test_execute_defaults_delegates_to_run_strategy_with_defaults_key():
    """``execute_defaults`` builds a ``DefaultsStrategy`` and calls
    ``_run_strategy`` with ``results_key="defaults"``."""
    task = MagicMock()
    execute_defaults(task, show=False)

    task._run_strategy.assert_called_once()
    kwargs = task._run_strategy.call_args.kwargs
    assert kwargs["results_key"] == "defaults"
    assert kwargs["show"] is False
    strategy = task._run_strategy.call_args.args[0]
    assert isinstance(strategy, DefaultsStrategy)


def test_defaults_does_not_read_tuning_cache():
    """End-to-end through ``_run_strategy``: the strategy hook is invoked
    once per target without ever calling ``load_tuning_results`` (that path
    belongs to ``LazyStrategy`` with ``use_tuned_params=True``)."""
    task = DefaultsTask(predict_size=24, auto_save_models=False)
    task.config.targets = ["t1"]
    strategy = DefaultsStrategy()

    with patch.object(task, "_ensure_pipeline_ready"), patch.object(
        task,
        "_get_target_data",
        return_value=(MagicMock(), MagicMock(), MagicMock()),
    ), patch.object(task, "create_forecaster"), patch.object(
        task, "_train_and_predict_target", return_value={"future_pred": MagicMock()}
    ), patch.object(task, "_aggregate_and_show", return_value={}), patch.object(
        task, "load_tuning_results"
    ) as load_tuning_results:
        task._run_strategy(
            strategy, task_name="t", results_key="defaults", show=False
        )

    load_tuning_results.assert_not_called()


def test_defaults_does_not_write_tuning_results():
    """End-to-end through ``_run_strategy``: no JSON tuning-result file is
    written for the defaults task (that path belongs to Optuna/SpotOptim)."""
    task = DefaultsTask(predict_size=24, auto_save_models=False)
    task.config.targets = ["t1"]
    strategy = DefaultsStrategy()

    with patch.object(task, "_ensure_pipeline_ready"), patch.object(
        task,
        "_get_target_data",
        return_value=(MagicMock(), MagicMock(), MagicMock()),
    ), patch.object(task, "create_forecaster"), patch.object(
        task, "_train_and_predict_target", return_value={"future_pred": MagicMock()}
    ), patch.object(task, "_aggregate_and_show", return_value={}), patch.object(
        task, "save_tuning_results"
    ) as save_tuning_results:
        task._run_strategy(
            strategy, task_name="t", results_key="defaults", show=False
        )

    save_tuning_results.assert_not_called()


def test_defaults_saves_models_under_defaults_key_when_auto_save():
    """``auto_save_models=True`` causes ``_run_strategy`` to call
    ``save_models(task_name="defaults")`` so ``PredictTask`` can find them."""
    task = DefaultsTask(predict_size=24, auto_save_models=True)
    task.config.targets = ["t1"]
    strategy = DefaultsStrategy()

    with patch.object(task, "_ensure_pipeline_ready"), patch.object(
        task,
        "_get_target_data",
        return_value=(MagicMock(), MagicMock(), MagicMock()),
    ), patch.object(task, "create_forecaster"), patch.object(
        task, "_train_and_predict_target", return_value={"future_pred": MagicMock()}
    ), patch.object(task, "_aggregate_and_show", return_value={}), patch.object(
        task, "save_models"
    ) as save_models:
        task._run_strategy(
            strategy, task_name="t", results_key="defaults", show=False
        )

    save_models.assert_called_once_with(task_name="defaults")


def test_multitask_dispatcher_routes_defaults_task():
    """``MultiTask(task="defaults").run()`` reaches ``run_task_defaults``."""
    from spotforecast2.multitask import MultiTask

    mt = MultiTask(task="defaults", predict_size=24)
    with patch.object(mt, "run_task_defaults") as run_task_defaults:
        run_task_defaults.return_value = {}
        mt.run(show=False)
    run_task_defaults.assert_called_once_with(show=False)


def test_runner_accepts_task_defaults():
    """``runner._PIPELINE_TASKS`` includes ``"defaults"`` so
    ``run(task="defaults")`` does not raise on the task-name validation."""
    from spotforecast2.multitask.runner import _PIPELINE_TASKS

    assert "defaults" in _PIPELINE_TASKS
