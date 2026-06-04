# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for opt-in fail-safe handling of Open-Meteo failures in BaseTask.

Covers the two-branch behaviour added by
``config.on_weather_failure`` and the matching
``except WeatherFetchError`` block in
``BaseTask.build_exogenous_features``:

- Default policy (``"raise"``): a ``WeatherFetchError`` raised by
  ``get_weather_features`` propagates and aborts the pipeline.
- Opt-in policy (``"skip"``): the same failure is caught, a warning
  is logged, and ``self.weather_aligned`` becomes an empty DataFrame
  so the rest of the pipeline can continue without weather features.

The mock replaces ``spotforecast2_safe.multitask.base.get_weather_features``
directly — no network is touched.  After the pipeline collapse the
implementation lives in the safe package; the patch target was updated
accordingly.
"""

import logging
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import pytest
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.weather import WeatherFetchError

from spotforecast2.multitask import MultiTask


def _make_task_ready(on_weather_failure: str = "raise") -> MultiTask:
    """Build a MultiTask with ``df_pipeline`` already populated.

    The test bypasses ``prepare_data`` by assigning a synthetic hourly
    index directly.  Calendar / day-night / holiday feature builders are
    mocked at the call site (see ``_mock_sibling_features``) so the test
    only exercises the weather-failure branch we are validating.
    """
    cfg = ConfigMulti(on_weather_failure=on_weather_failure)
    mt = MultiTask(cfg)
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    mt.df_pipeline = pd.DataFrame({"target_0": range(48)}, index=idx)
    mt.config.targets = ["target_0"]
    mt.config.data_start = idx[0]
    mt.config.cov_end = idx[-1]
    return mt


def _mock_sibling_features(stack: ExitStack, index: pd.DatetimeIndex) -> None:
    """Mock the non-weather feature builders + downstream transforms.

    The test exercises only the weather try/except contract — it should
    not depend on the calendar/day-night/holiday helpers' synthetic-data
    assumptions, nor on the downstream cyclical-encoding / interaction /
    selection / merge chain.  All of those are stubbed here.
    """
    empty = pd.DataFrame(index=index)
    stack.enter_context(
        patch("spotforecast2_safe.multitask.base.get_calendar_features", return_value=empty)
    )
    stack.enter_context(
        patch("spotforecast2_safe.multitask.base.get_day_night_features", return_value=empty)
    )
    stack.enter_context(
        patch("spotforecast2_safe.multitask.base.get_holiday_features", return_value=empty)
    )
    # Downstream transforms — return shapes the caller assigns through.
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
            "spotforecast2_safe.multitask.base.select_exogenous_features",
            return_value=[],
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2_safe.multitask.base.merge_data_and_covariates",
            return_value=(empty, empty, empty),
        )
    )


class TestRaiseByDefault:
    def test_weather_failure_propagates(self):
        mt = _make_task_ready()
        with patch(
            "spotforecast2_safe.multitask.base.get_weather_features",
            side_effect=WeatherFetchError("simulated outage"),
        ):
            with pytest.raises(WeatherFetchError, match="simulated outage"):
                mt.build_exogenous_features()


class TestSkipWhenConfigured:
    def test_weather_failure_swallowed(self):
        mt = _make_task_ready(on_weather_failure="skip")
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_weather_features",
                    side_effect=WeatherFetchError("simulated outage"),
                )
            )
            _mock_sibling_features(stack, mt.df_pipeline.index)
            # Should NOT raise.
            result = mt.build_exogenous_features()
        assert result is mt

    def test_weather_aligned_is_empty(self):
        mt = _make_task_ready(on_weather_failure="skip")
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_weather_features",
                    side_effect=WeatherFetchError("simulated outage"),
                )
            )
            _mock_sibling_features(stack, mt.df_pipeline.index)
            mt.build_exogenous_features()
        assert mt.weather_aligned is not None
        assert mt.weather_aligned.empty

    def test_no_weather_columns_selected(self):
        mt = _make_task_ready(on_weather_failure="skip")
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_weather_features",
                    side_effect=WeatherFetchError("simulated outage"),
                )
            )
            _mock_sibling_features(stack, mt.df_pipeline.index)
            mt.build_exogenous_features()
        # exog_feature_names is built from the merged feature matrix; with
        # weather absent, the typical Open-Meteo column names cannot appear.
        weather_token_columns = [
            name
            for name in mt.exog_feature_names
            if "temperature" in name
            or "wind_" in name
            or "precipitation" in name
            or "humidity" in name
        ]
        assert weather_token_columns == []


class TestWarningLogged:
    def test_warning_logged_on_skip(self, caplog):
        mt = _make_task_ready(on_weather_failure="skip")
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "spotforecast2_safe.multitask.base.get_weather_features",
                        side_effect=WeatherFetchError("simulated outage"),
                    )
                )
                _mock_sibling_features(stack, mt.df_pipeline.index)
                mt.build_exogenous_features()
        assert any(
            rec.levelno == logging.WARNING
            and "Open-Meteo fetch failed" in rec.message
            and "simulated outage" in rec.message
            for rec in caplog.records
        )
