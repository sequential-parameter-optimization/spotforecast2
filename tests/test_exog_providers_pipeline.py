# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for the exogenous-provider injection in ``build_exogenous_features``.

``BaseTask.build_exogenous_features`` appends the columns produced by the
sf2-safe exogenous-provider registry (covid_infection_rate + the ENTSO-E
day-ahead forecast / renewable / net-load / price) and adds their names to
``exog_feature_names`` after the pattern-based selection. These tests exercise
that block end-to-end:

- the covid provider (bundled, offline) is appended *and* selected even though
  ``select_exogenous_features`` would not match it by name;
- a provider whose data is missing is skipped or re-raised per
  ``config.on_exog_provider_failure`` (fail-safe);
- with no provider flags the exogenous frame is unchanged.

All heavy feature builders are patched at their import site in
``spotforecast2.multitask.base`` so no network is touched. ``SPOTFORECAST2_DATA``
is pointed at an empty temp dir so the ENTSO-E providers (which read interim
CSVs) fail deterministically; the covid provider reads package data and is
unaffected.
"""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.preprocessing.exog_providers import ExogProviderError

from spotforecast2.multitask import MultiTask


def _make_task(**config_kwargs) -> MultiTask:
    cfg = ConfigMulti(**config_kwargs)
    mt = MultiTask(cfg)
    idx = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    mt.df_pipeline = pd.DataFrame({"target_0": np.arange(96.0)}, index=idx)
    mt.run_state.targets = ["target_0"]
    mt.run_state.data_start = idx[0]
    mt.run_state.data_end = idx[-1]
    mt.run_state.cov_end = idx[-1]
    return mt


def _frame(index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: 0.0 for c in columns}, index=index)


def _patch_builders(stack: ExitStack, idx: pd.DatetimeIndex) -> None:
    """Patch the heavy builders, but leave ``select_exogenous_features`` real."""
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
            side_effect=lambda exogenous_features, weather_aligned, **kwargs: (
                exogenous_features
            ),
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


@pytest.fixture(autouse=True)
def _empty_data_home(tmp_path, monkeypatch):
    # No interim CSVs here, so the ENTSO-E providers fail deterministically.
    monkeypatch.setenv("SPOTFORECAST2_DATA", str(tmp_path))


def test_covid_provider_appended_and_selected():
    mt = _make_task(include_covid_infection_rate=True, on_exog_provider_failure="skip")
    idx = mt.df_pipeline.index
    with ExitStack() as stack:
        _patch_builders(stack, idx)
        mt.build_exogenous_features()

    # The injection added the column ...
    assert "covid_infection_rate" in mt.exogenous_features.columns
    # ... and it survives selection even though select_exogenous_features does
    # not match it by name (it is appended afterwards).
    assert "covid_infection_rate" in mt.exog_feature_names
    assert not mt.exogenous_features["covid_infection_rate"].isna().any()


def test_missing_provider_skipped_but_others_kept():
    mt = _make_task(
        include_covid_infection_rate=True,
        include_entsoe_forecast_load=True,  # no interim CSV -> fails
        on_exog_provider_failure="skip",
    )
    idx = mt.df_pipeline.index
    with ExitStack() as stack:
        _patch_builders(stack, idx)
        mt.build_exogenous_features()  # must not raise

    assert "covid_infection_rate" in mt.exog_feature_names
    assert "entsoe_forecasted_load" not in mt.exogenous_features.columns
    assert "entsoe_forecasted_load" not in mt.exog_feature_names


def test_missing_provider_raises_when_configured():
    mt = _make_task(
        include_entsoe_forecast_load=True,  # no interim CSV -> fails
        on_exog_provider_failure="raise",
    )
    idx = mt.df_pipeline.index
    with ExitStack() as stack:
        _patch_builders(stack, idx)
        with pytest.raises(ExogProviderError):
            mt.build_exogenous_features()


def test_no_provider_flags_leaves_exog_unchanged():
    mt = _make_task()  # all provider flags default False
    idx = mt.df_pipeline.index
    with ExitStack() as stack:
        _patch_builders(stack, idx)
        mt.build_exogenous_features()

    provider_cols = [
        c
        for c in mt.exogenous_features.columns
        if c == "covid_infection_rate" or c.startswith("entsoe_")
    ]
    assert provider_cols == []
