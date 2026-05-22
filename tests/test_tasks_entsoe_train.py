# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``spotforecast2-entsoe train`` CLI subcommand.

Mocks ``spotforecast2.multitask.runner.run`` so the tests run offline and
without needing the merged ENTSO-E CSV.  They verify only that the CLI
dispatches to ``run(...)`` with the correct kwargs for each model choice
(LightGBM / XGBoost).
"""

from unittest.mock import patch

from spotforecast2_safe.configurator import ConfigEntsoe

from spotforecast2.tasks.task_entsoe import (
    entsoe_lgbm_factory,
    entsoe_xgb_factory,
    entsoe_data_loader,
    main,
)


def _train_call_kwargs(model: str) -> dict:
    """Invoke ``main()`` with ``train <model>`` while mocking ``run``."""
    with patch("spotforecast2.tasks.task_entsoe.run") as mock_run:
        with patch("sys.argv", ["spotforecast2-entsoe", "train", model]):
            main()
    assert mock_run.call_count == 1
    return mock_run.call_args.kwargs


def test_train_lgbm_dispatches_to_run_with_defaults_task():
    kwargs = _train_call_kwargs("lgbm")
    assert kwargs["task"] == "defaults"
    assert kwargs["config_cls"] is ConfigEntsoe
    assert kwargs["project_name"] == "entsoe-lgbm"
    assert kwargs["targets"] == ["Actual Load"]
    assert kwargs["agg_weights"] == [1.0]
    assert kwargs["bounds"] == [(-1e9, 1e9)]
    assert kwargs["data_loader"] is entsoe_data_loader
    assert kwargs["forecaster_factory"] is entsoe_lgbm_factory
    assert kwargs["index_name"] == "Time (UTC)"
    assert kwargs["show"] is False


def test_train_xgb_dispatches_to_run_with_xgb_factory():
    kwargs = _train_call_kwargs("xgb")
    assert kwargs["task"] == "defaults"
    assert kwargs["project_name"] == "entsoe-xgb"
    assert kwargs["forecaster_factory"] is entsoe_xgb_factory


def test_train_default_model_is_lgbm():
    """Omitting the positional ``model`` arg falls back to LightGBM."""
    with patch("spotforecast2.tasks.task_entsoe.run") as mock_run:
        with patch("sys.argv", ["spotforecast2-entsoe", "train"]):
            main()
    assert mock_run.call_args.kwargs["project_name"] == "entsoe-lgbm"
    assert mock_run.call_args.kwargs["forecaster_factory"] is entsoe_lgbm_factory


def test_train_show_flag_forwards_to_run():
    with patch("spotforecast2.tasks.task_entsoe.run") as mock_run:
        with patch("sys.argv", ["spotforecast2-entsoe", "train", "lgbm", "--show"]):
            main()
    assert mock_run.call_args.kwargs["show"] is True
