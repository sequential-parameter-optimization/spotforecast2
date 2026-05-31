# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Failure-mode taxonomy tests for ``BaseTask.build_exogenous_features``.

Documented in the ENTSO-E manuscript as
``@sec-exo-failure-modes`` (``Publications/bart26b_spotforecast2_entsoe/
index.qmd``).  Each test here corresponds 1:1 with a row in
``@tbl-exo-failure-scenarios``:

- S1: Happy path — all four feature groups built and merged.
- S2: Open-Meteo down, ``on_weather_failure="raise"`` (default).
- S3: Open-Meteo down, ``on_weather_failure="skip"``.
- S4: Invalid ``country_code`` — holidays raise; not caught by
  ``on_weather_failure``.
- S5: Invalid coordinates — day-night raises; not caught.
- S7: Master toggle ``use_exogenous_features=False`` short-circuits.

S6 (train-with-weather → predict-without-weather LightGBM schema
mismatch) lives in
``test_exogenous_schema_mismatch_integration.py`` because it requires a
fit-and-predict round-trip rather than a single
``build_exogenous_features`` call.

S2 / S3 mirror — and intentionally do NOT replace —
``test_multitask_weather_failure.py``.  Keeping every scenario in one
module makes the manuscript's S1–S7 table executable in one pytest run.

All builders are patched at their import site in
``spotforecast2.multitask.base`` so no leaf library (``astral``,
``holidays``, ``requests``) is touched by these tests.
"""

import logging
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import pytest
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.weather import WeatherFetchError

from spotforecast2.multitask import MultiTask

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_task_ready(
    on_weather_failure: str = "raise",
    use_exogenous_features: bool = True,
) -> MultiTask:
    """Build a ``MultiTask`` with ``df_pipeline`` already populated.

    Mirrors the helper in ``test_multitask_weather_failure.py`` but
    exposes ``use_exogenous_features`` so S7 can flip it.
    """
    cfg = ConfigMulti(
        on_weather_failure=on_weather_failure,
        use_exogenous_features=use_exogenous_features,
    )
    mt = MultiTask(cfg)
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    mt.df_pipeline = pd.DataFrame({"target_0": range(48)}, index=idx)
    mt.config.targets = ["target_0"]
    mt.config.data_start = idx[0]
    mt.config.cov_end = idx[-1]
    return mt


def _stub_downstream(stack: ExitStack) -> None:
    """Stub the post-concat transforms / merge so each test isolates one branch."""
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.apply_cyclical_encoding",
            side_effect=lambda data, drop_original: data,
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.create_interaction_features",
            side_effect=lambda exogenous_features, weather_aligned, **_: exogenous_features,
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.select_exogenous_features",
            side_effect=lambda exogenous_features, **_: list(
                exogenous_features.columns
            ),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.merge_data_and_covariates",
            side_effect=lambda data, exogenous_features, **_: (
                data,
                exogenous_features,
                exogenous_features,
            ),
        )
    )


def _frame(index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    """Distinct dataframe per feature group so we can verify their columns survive."""
    return pd.DataFrame({c: 0 for c in columns}, index=index)


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


class TestScenarioS1HappyPath:
    """S1 — all four feature groups are built and concatenated."""

    def test_all_four_groups_present_in_exog_columns(self):
        mt = _make_task_ready()
        idx = mt.df_pipeline.index
        weather_cols = ["temperature_2m", "wind_speed_10m"]
        calendar_cols = ["month", "hour"]
        day_night_cols = ["sunrise_hour", "is_daylight"]
        holiday_cols = ["is_holiday"]

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_weather_features",
                    return_value=(
                        _frame(idx, weather_cols),
                        _frame(idx, weather_cols),
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_calendar_features",
                    return_value=_frame(idx, calendar_cols),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_day_night_features",
                    return_value=_frame(idx, day_night_cols),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_holiday_features",
                    return_value=_frame(idx, holiday_cols),
                )
            )
            _stub_downstream(stack)
            mt.build_exogenous_features()

        cols = set(mt.exogenous_features.columns)
        assert set(weather_cols).issubset(cols)
        assert set(calendar_cols).issubset(cols)
        assert set(day_night_cols).issubset(cols)
        assert set(holiday_cols).issubset(cols)
        assert mt.weather_aligned is not None and not mt.weather_aligned.empty


class TestScenarioS2WeatherRaiseDefault:
    """S2 — ``on_weather_failure="raise"`` (default) propagates."""

    def test_weather_fetch_error_propagates(self):
        mt = _make_task_ready(on_weather_failure="raise")
        with patch(
            "spotforecast2.multitask.base.get_weather_features",
            side_effect=WeatherFetchError("simulated outage"),
        ):
            with pytest.raises(WeatherFetchError, match="simulated outage"):
                mt.build_exogenous_features()


class TestScenarioS3WeatherSkip:
    """S3 — ``on_weather_failure="skip"`` swallows the WeatherFetchError."""

    def _run_with_skip(self, mt: MultiTask) -> None:
        idx = mt.df_pipeline.index
        empty = pd.DataFrame(index=idx)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_weather_features",
                    side_effect=WeatherFetchError("simulated outage"),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_calendar_features",
                    return_value=empty,
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_day_night_features",
                    return_value=empty,
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_holiday_features",
                    return_value=empty,
                )
            )
            _stub_downstream(stack)
            mt.build_exogenous_features()

    def test_pipeline_continues_without_weather(self):
        mt = _make_task_ready(on_weather_failure="skip")
        self._run_with_skip(mt)
        assert mt.weather_aligned is not None and mt.weather_aligned.empty

    def test_warning_logged_on_skip(self, caplog):
        mt = _make_task_ready(on_weather_failure="skip")
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            self._run_with_skip(mt)
        assert any(
            rec.levelno == logging.WARNING
            and "Open-Meteo fetch failed" in rec.message
            and "simulated outage" in rec.message
            for rec in caplog.records
        )

    def test_no_weather_columns_in_exog(self):
        mt = _make_task_ready(on_weather_failure="skip")
        self._run_with_skip(mt)
        weather_tokens = ("temperature", "wind_", "precipitation", "humidity")
        leaked = [
            name
            for name in mt.exog_feature_names
            if any(tok in name for tok in weather_tokens)
        ]
        assert leaked == []


class TestScenarioS4InvalidCountryCode:
    """S4 — holiday builder raising is *not* caught by ``on_weather_failure``."""

    def _build_with_holiday_failure(self, on_weather_failure: str) -> None:
        mt = _make_task_ready(on_weather_failure=on_weather_failure)
        idx = mt.df_pipeline.index
        empty = pd.DataFrame(index=idx)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_weather_features",
                    return_value=(empty, empty),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_calendar_features",
                    return_value=empty,
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_day_night_features",
                    return_value=empty,
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_holiday_features",
                    side_effect=NotImplementedError(
                        "Country ZZ is not available in the holidays library"
                    ),
                )
            )
            _stub_downstream(stack)
            mt.build_exogenous_features()

    def test_invalid_country_raises_by_default(self):
        with pytest.raises(NotImplementedError, match="Country ZZ"):
            self._build_with_holiday_failure(on_weather_failure="raise")

    def test_invalid_country_still_raises_under_skip(self):
        # Regression guard: on_weather_failure="skip" must NOT swallow
        # holiday-builder failures — only WeatherFetchError is in scope.
        with pytest.raises(NotImplementedError, match="Country ZZ"):
            self._build_with_holiday_failure(on_weather_failure="skip")


class TestScenarioS5InvalidCoordinates:
    """S5 — day-night builder raising is *not* caught by ``on_weather_failure``."""

    def _build_with_day_night_failure(self, on_weather_failure: str) -> None:
        mt = _make_task_ready(on_weather_failure=on_weather_failure)
        idx = mt.df_pipeline.index
        empty = pd.DataFrame(index=idx)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_weather_features",
                    return_value=(empty, empty),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_calendar_features",
                    return_value=empty,
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_day_night_features",
                    side_effect=ValueError("latitude out of range"),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_holiday_features",
                    return_value=empty,
                )
            )
            _stub_downstream(stack)
            mt.build_exogenous_features()

    def test_invalid_coords_raises_by_default(self):
        with pytest.raises(ValueError, match="latitude out of range"):
            self._build_with_day_night_failure(on_weather_failure="raise")

    def test_invalid_coords_still_raises_under_skip(self):
        with pytest.raises(ValueError, match="latitude out of range"):
            self._build_with_day_night_failure(on_weather_failure="skip")


class TestScenarioS7MasterToggleOff:
    """S7 — ``use_exogenous_features=False`` short-circuits the whole step."""

    def test_no_builder_is_called(self):
        mt = _make_task_ready(use_exogenous_features=False)
        with ExitStack() as stack:
            mock_weather = stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_weather_features",
                    side_effect=RuntimeError("must not be called"),
                )
            )
            mock_calendar = stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_calendar_features",
                    side_effect=RuntimeError("must not be called"),
                )
            )
            mock_day_night = stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_day_night_features",
                    side_effect=RuntimeError("must not be called"),
                )
            )
            mock_holiday = stack.enter_context(
                patch(
                    "spotforecast2.multitask.base.get_holiday_features",
                    side_effect=RuntimeError("must not be called"),
                )
            )
            result = mt.build_exogenous_features()

        assert result is mt
        assert not mock_weather.called
        assert not mock_calendar.called
        assert not mock_day_night.called
        assert not mock_holiday.called
