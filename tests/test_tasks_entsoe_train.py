# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``spotforecast2-entsoe train`` CLI subcommand.

Mocks ``spotforecast2.tasks.task_entsoe._run_entsoe_pipeline`` so the tests
run offline and without needing the merged ENTSO-E CSV.  The CLI invokes
``_run_entsoe_pipeline(config, task="defaults", project_name=..., show=...)``
— the config carries every pipeline parameter (targets, agg_weights, bounds,
data_loader, forecaster_factory, index_name, …).
"""

from unittest.mock import patch

from spotforecast2_safe.configurator import ConfigEntsoe

from spotforecast2.tasks.task_entsoe import (
    entsoe_data_loader,
    entsoe_lgbm_factory,
    entsoe_xgb_factory,
    main,
)


def _train_call(model: str):
    """Invoke ``main()`` with ``train <model> --force`` while mocking the helper.

    Returns ``(positional_args, keyword_args)`` of the single
    ``_run_entsoe_pipeline`` call.  ``--force`` bypasses the cadence gate so
    the dispatch test does not depend on the state of the user's model cache.
    """
    with patch("spotforecast2.tasks.task_entsoe._run_entsoe_pipeline") as mock_pipeline:
        with patch("sys.argv", ["spotforecast2-entsoe", "train", model, "--force"]):
            main()
    assert mock_pipeline.call_count == 1
    return mock_pipeline.call_args.args, mock_pipeline.call_args.kwargs


def test_train_lgbm_dispatches_to_pipeline_with_defaults_task():
    args, kwargs = _train_call("lgbm")
    config = args[0]
    assert isinstance(config, ConfigEntsoe)
    assert kwargs["task"] == "defaults"
    assert kwargs["project_name"] == "entsoe-lgbm"
    assert config.targets == ["Actual Load"]
    assert config.agg_weights == [1.0]
    assert config.bounds == [(-1e9, 1e9)]
    assert config.data_loader is entsoe_data_loader
    assert config.forecaster_factory is entsoe_lgbm_factory
    assert config.index_name == "Time (UTC)"
    assert kwargs["show"] is False


def test_train_xgb_dispatches_to_pipeline_with_xgb_factory():
    args, kwargs = _train_call("xgb")
    config = args[0]
    assert kwargs["task"] == "defaults"
    assert kwargs["project_name"] == "entsoe-xgb"
    assert config.forecaster_factory is entsoe_xgb_factory


def test_train_default_model_is_lgbm():
    """Omitting the positional ``model`` arg falls back to LightGBM."""
    with patch("spotforecast2.tasks.task_entsoe._run_entsoe_pipeline") as mock_pipeline:
        with patch("sys.argv", ["spotforecast2-entsoe", "train", "--force"]):
            main()
    args = mock_pipeline.call_args.args
    kwargs = mock_pipeline.call_args.kwargs
    assert kwargs["project_name"] == "entsoe-lgbm"
    assert args[0].forecaster_factory is entsoe_lgbm_factory


def test_train_show_flag_forwards_to_pipeline():
    with patch("spotforecast2.tasks.task_entsoe._run_entsoe_pipeline") as mock_pipeline:
        with patch(
            "sys.argv",
            ["spotforecast2-entsoe", "train", "lgbm", "--show", "--force"],
        ):
            main()
    assert mock_pipeline.call_args.kwargs["show"] is True
