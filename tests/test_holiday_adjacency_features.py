# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``include_holiday_adjacency_features`` flag wiring.

Guards three behaviours:

1. When ``include_holiday_adjacency_features=True``, ``build_exogenous_features``
   calls ``get_holiday_adjacency_features`` with the same arguments as the
   sibling ``get_holiday_features`` call and concatenates the result.
2. When ``include_holiday_adjacency_features=False`` (default), the function is
   never called — existing behaviour is byte-identical.
3. The flag is forwarded to ``select_exogenous_features`` in both the True and
   False branches.

All heavy feature builders are patched at their import site in
``spotforecast2.multitask.base`` so no network / leaf library is touched.
"""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.configurator.config_multi import ConfigMulti

from spotforecast2.multitask import MultiTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(include_holiday_adjacency_features: bool = False) -> MultiTask:
    cfg = ConfigMulti(
        include_holiday_adjacency_features=include_holiday_adjacency_features,
    )
    mt = MultiTask(cfg)
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    mt.df_pipeline = pd.DataFrame({"target_0": np.arange(72.0)}, index=idx)
    mt.run_state.targets = ["target_0"]
    mt.run_state.data_start = idx[0]
    mt.run_state.cov_end = idx[-1]
    return mt


def _frame(index: pd.DatetimeIndex, columns: list) -> pd.DataFrame:
    return pd.DataFrame({c: 0.0 for c in columns}, index=index)


def _patch_base_builders(stack: ExitStack, idx: pd.DatetimeIndex) -> None:
    """Patch all heavy builders except get_holiday_adjacency_features.

    ``select_exogenous_features`` gets a pass-through default so no test
    depends on the installed spotforecast2-safe signature; individual
    tests re-patch it when they need to assert call kwargs.
    """
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.get_weather_features",
            return_value=(
                _frame(idx, ["temperature_2m"]),
                _frame(idx, ["temperature_2m"]),
            ),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.get_calendar_features",
            return_value=_frame(idx, ["hour"]),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.get_day_night_features",
            return_value=_frame(idx, ["is_daylight"]),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.get_holiday_features",
            return_value=_frame(idx, ["is_holiday"]),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.apply_cyclical_encoding",
            side_effect=lambda data, drop_original: data,
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.create_interaction_features",
            side_effect=lambda exogenous_features, weather_aligned, **_: exogenous_features,
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.merge_data_and_covariates",
            side_effect=lambda data, exogenous_features, **_: (
                data,
                exogenous_features,
                exogenous_features,
            ),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.select_exogenous_features",
            side_effect=lambda exogenous_features, **_: list(
                exogenous_features.columns
            ),
        )
    )


# ---------------------------------------------------------------------------
# Tests: flag=True
# ---------------------------------------------------------------------------


class TestAdjacencyFeaturesEnabled:
    def test_get_holiday_adjacency_features_is_called(self):
        """When the flag is True, the adjacency builder must be invoked."""
        mt = _make_task(include_holiday_adjacency_features=True)
        idx = mt.df_pipeline.index
        with ExitStack() as stack:
            _patch_base_builders(stack, idx)
            mock_adj = stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_adjacency_features",
                    return_value=_frame(
                        idx, ["is_brueckentag", "is_before_holiday", "is_after_holiday"]
                    ),
                )
            )
            mt.build_exogenous_features()

        mock_adj.assert_called_once()

    def test_adjacency_columns_present_after_concat(self):
        """Adjacency columns must appear in ``mt.exogenous_features`` after the build."""
        mt = _make_task(include_holiday_adjacency_features=True)
        idx = mt.df_pipeline.index
        with ExitStack() as stack:
            _patch_base_builders(stack, idx)
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_adjacency_features",
                    return_value=_frame(
                        idx, ["is_brueckentag", "is_before_holiday", "is_after_holiday"]
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.select_exogenous_features",
                    side_effect=lambda exogenous_features, **_: list(
                        exogenous_features.columns
                    ),
                )
            )
            mt.build_exogenous_features()

        for col in ("is_brueckentag", "is_before_holiday", "is_after_holiday"):
            assert col in mt.exogenous_features.columns, f"Missing column: {col}"

    def test_adjacency_args_match_holiday_args(self):
        """The adjacency builder must receive the same location/time args as get_holiday_features."""
        mt = _make_task(include_holiday_adjacency_features=True)
        idx = mt.df_pipeline.index
        with ExitStack() as stack:
            _patch_base_builders(stack, idx)
            mock_holiday = stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_features",
                    return_value=_frame(idx, ["is_holiday"]),
                )
            )
            mock_adj = stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_adjacency_features",
                    return_value=_frame(
                        idx, ["is_brueckentag", "is_before_holiday", "is_after_holiday"]
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.select_exogenous_features",
                    side_effect=lambda exogenous_features, **_: list(
                        exogenous_features.columns
                    ),
                )
            )
            mt.build_exogenous_features()

        holiday_kw = mock_holiday.call_args.kwargs
        adj_kw = mock_adj.call_args.kwargs
        for key in (
            "start",
            "cov_end",
            "forecast_horizon",
            "tz",
            "freq",
            "country_code",
            "state",
        ):
            assert adj_kw.get(key) == holiday_kw.get(key), (
                f"Argument '{key}' differs: adjacency={adj_kw.get(key)!r}, "
                f"holiday={holiday_kw.get(key)!r}"
            )

    def test_select_exogenous_features_receives_flag_true(self):
        """``select_exogenous_features`` must be called with the flag set to True."""
        mt = _make_task(include_holiday_adjacency_features=True)
        idx = mt.df_pipeline.index
        with ExitStack() as stack:
            _patch_base_builders(stack, idx)
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_adjacency_features",
                    return_value=_frame(
                        idx, ["is_brueckentag", "is_before_holiday", "is_after_holiday"]
                    ),
                )
            )
            mock_select = stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.select_exogenous_features",
                    return_value=[],
                )
            )
            mt.build_exogenous_features()

        kw = mock_select.call_args.kwargs
        assert kw.get("include_holiday_adjacency_features") is True


# ---------------------------------------------------------------------------
# Tests: flag=False (default)
# ---------------------------------------------------------------------------


class TestAdjacencyFeaturesDisabled:
    def test_get_holiday_adjacency_features_not_called(self):
        """When the flag is False, the adjacency builder must not be invoked."""
        mt = _make_task(include_holiday_adjacency_features=False)
        idx = mt.df_pipeline.index
        with ExitStack() as stack:
            _patch_base_builders(stack, idx)
            mock_adj = stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_adjacency_features",
                    return_value=_frame(
                        idx, ["is_brueckentag", "is_before_holiday", "is_after_holiday"]
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.select_exogenous_features",
                    return_value=[],
                )
            )
            mt.build_exogenous_features()

        mock_adj.assert_not_called()

    def test_select_exogenous_features_receives_flag_false(self):
        """``select_exogenous_features`` must be called with the flag set to False."""
        mt = _make_task(include_holiday_adjacency_features=False)
        idx = mt.df_pipeline.index
        with ExitStack() as stack:
            _patch_base_builders(stack, idx)
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_adjacency_features",
                    return_value=_frame(
                        idx, ["is_brueckentag", "is_before_holiday", "is_after_holiday"]
                    ),
                )
            )
            mock_select = stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.select_exogenous_features",
                    return_value=[],
                )
            )
            mt.build_exogenous_features()

        kw = mock_select.call_args.kwargs
        assert kw.get("include_holiday_adjacency_features") is False


# ---------------------------------------------------------------------------
# Test: ConfigMulti default
# ---------------------------------------------------------------------------


def test_config_multi_default_is_false():
    """ConfigMulti must default include_holiday_adjacency_features to False."""
    cfg = ConfigMulti()
    assert cfg.include_holiday_adjacency_features is False


def test_config_entsoe_default_is_false():
    """ConfigEntsoe must default include_holiday_adjacency_features to False."""
    cfg = ConfigEntsoe()
    assert cfg.include_holiday_adjacency_features is False


def test_config_multi_flag_propagates_to_task():
    """A ConfigMulti with the flag True must be reflected on task.config."""
    cfg = ConfigMulti(include_holiday_adjacency_features=True)
    mt = MultiTask(cfg)
    assert mt.config.include_holiday_adjacency_features is True
