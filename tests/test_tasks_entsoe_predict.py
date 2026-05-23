# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``spotforecast2-entsoe predict`` CLI subcommand."""

from unittest.mock import patch

from spotforecast2_safe.configurator import ConfigEntsoe

from spotforecast2.tasks.task_entsoe import (
    entsoe_lgbm_factory,
    entsoe_xgb_factory,
    entsoe_data_loader,
    main,
)


def _predict_call(model: str):
    """Invoke ``main()`` with ``predict <model>`` while mocking ``run``.

    Returns ``(positional_args, keyword_args)`` of the single ``run`` call.
    """
    with patch("spotforecast2.tasks.task_entsoe.run") as mock_run:
        with patch("sys.argv", ["spotforecast2-entsoe", "predict", model]):
            main()
    assert mock_run.call_count == 1
    return mock_run.call_args.args, mock_run.call_args.kwargs


def test_predict_lgbm_dispatches_to_run_with_predict_task():
    args, kwargs = _predict_call("lgbm")
    config = args[0]
    assert isinstance(config, ConfigEntsoe)
    assert kwargs["task"] == "predict"
    assert kwargs["project_name"] == "entsoe-lgbm"
    assert config.targets == ["Actual Load"]
    assert config.data_loader is entsoe_data_loader
    assert config.forecaster_factory is entsoe_lgbm_factory
    assert config.index_name == "Time (UTC)"


def test_predict_xgb_uses_xgb_project_and_factory():
    args, kwargs = _predict_call("xgb")
    config = args[0]
    assert kwargs["task"] == "predict"
    assert kwargs["project_name"] == "entsoe-xgb"
    assert config.forecaster_factory is entsoe_xgb_factory


def test_predict_default_model_is_lgbm():
    with patch("spotforecast2.tasks.task_entsoe.run") as mock_run:
        with patch("sys.argv", ["spotforecast2-entsoe", "predict"]):
            main()
    args = mock_run.call_args.args
    kwargs = mock_run.call_args.kwargs
    assert kwargs["project_name"] == "entsoe-lgbm"
    assert args[0].forecaster_factory is entsoe_lgbm_factory
