# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration test for Scenario S6 — train-with-weather → predict-without.

Documented in the ENTSO-E manuscript as the schema-mismatch case in
``@sec-on-weather-failure`` / ``@tbl-exo-failure-scenarios``.  A model
trained while Open-Meteo is reachable carries a wide exogenous schema
(weather columns + window aggregates + interactions).  If that same
forecaster is later asked to ``predict`` after
``on_weather_failure="skip"`` has dropped the weather columns, LightGBM
raises::

    The number of features in data (X) is not the same as it was in
    training data (Y).

This module verifies *both* halves of that round-trip:

1.  The LightGBM error message itself, by fitting a small
    ``LGBMRegressor`` on N features and predicting on M ≠ N features.
2.  The pipeline-level cause: that
    ``BaseTask.build_exogenous_features`` emits a *narrower* exogenous
    column set when ``on_weather_failure="skip"`` swallows the failure
    versus when weather is reachable — which is the asymmetry that
    triggers (1) inside a real save → load → predict workflow.
"""

import logging
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.weather import WeatherFetchError

from spotforecast2.multitask import MultiTask

# ---------------------------------------------------------------------------
# Helpers — minimal MultiTask shared with the orchestration test module.
# ---------------------------------------------------------------------------


def _make_task(on_weather_failure: str = "raise") -> MultiTask:
    cfg = ConfigMulti(on_weather_failure=on_weather_failure)
    mt = MultiTask(cfg)
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    mt.df_pipeline = pd.DataFrame({"target_0": range(48)}, index=idx)
    mt.config.targets = ["target_0"]
    mt.config.data_start = idx[0]
    mt.config.cov_end = idx[-1]
    return mt


def _stub_downstream(stack: ExitStack) -> None:
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
            side_effect=lambda exogenous_features, **_: list(
                exogenous_features.columns
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


def _frame(index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: 0.0 for c in columns}, index=index)


# ---------------------------------------------------------------------------
# S6a — the bare LightGBM schema-mismatch error verbatim from the qmd.
# ---------------------------------------------------------------------------


class TestS6LightGBMRaisesOnFeatureCountMismatch:
    """Reproduces the feature-count schema mismatch referenced in the qmd.

    Fitting on 36 features and predicting on 51 features raises a
    ``ValueError``.  Recent LightGBM versions delegate input validation
    to the scikit-learn wrapper, so the literal wording differs between
    versions:

    - sklearn (current): ``"X has 51 features, but LGBMRegressor is
      expecting 36 features as input."``
    - older LightGBM (referenced in ``@sec-on-weather-failure``):
      ``"The number of features in data (51) is not the same as it was
      in training data (36)."``

    The regex below matches either phrasing so the test stays robust
    across upstream version bumps.
    """

    def test_predict_with_wider_exog_raises_feature_count_error(self):
        lgb = pytest.importorskip("lightgbm")
        rng = np.random.default_rng(seed=42)

        n_train_features = 36
        n_predict_features = 51
        x_train = rng.standard_normal((200, n_train_features))
        y_train = rng.standard_normal(200)
        x_predict = rng.standard_normal((24, n_predict_features))

        model = lgb.LGBMRegressor(n_estimators=20, verbose=-1)
        model.fit(x_train, y_train)

        with pytest.raises(
            ValueError,
            match=(
                r"(number of features in data.*not the same as.*training data"
                r"|X has \d+ features.*expecting \d+ features)"
            ),
        ):
            model.predict(x_predict)


# ---------------------------------------------------------------------------
# S6b — the *pipeline-level* cause of S6: the exogenous schema narrows
#       when on_weather_failure="skip" swallows the WeatherFetchError.
# ---------------------------------------------------------------------------


class TestS6PipelineSchemaAsymmetry:
    """Demonstrates the asymmetric column set across the round-trip.

    The qmd's schema-mismatch trap is a consequence of two consecutive
    ``build_exogenous_features`` calls returning incompatible feature
    sets.  Here we run that step twice on the same data — once with a
    reachable Open-Meteo, once with a simulated outage under ``"skip"``
    — and assert the resulting column inventories differ in exactly the
    weather columns.
    """

    def test_skip_yields_strictly_narrower_exog_columns(self, caplog):
        weather_cols = ["temperature_2m", "wind_speed_10m", "precipitation"]
        calendar_cols = ["month", "hour"]
        day_night_cols = ["sunrise_hour", "is_daylight"]
        holiday_cols = ["is_holiday"]

        # ---- Train-time: Open-Meteo reachable -----------------------------
        mt_train = _make_task(on_weather_failure="raise")
        idx_train = mt_train.df_pipeline.index
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_weather_features",
                    return_value=(
                        _frame(idx_train, weather_cols),
                        _frame(idx_train, weather_cols),
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_calendar_features",
                    return_value=_frame(idx_train, calendar_cols),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_day_night_features",
                    return_value=_frame(idx_train, day_night_cols),
                )
            )
            stack.enter_context(
                patch(
                    "spotforecast2_safe.multitask.base.get_holiday_features",
                    return_value=_frame(idx_train, holiday_cols),
                )
            )
            _stub_downstream(stack)
            mt_train.build_exogenous_features()
        train_cols = set(mt_train.exogenous_features.columns)

        # ---- Predict-time: Open-Meteo unreachable, skip in effect ---------
        mt_predict = _make_task(on_weather_failure="skip")
        idx_predict = mt_predict.df_pipeline.index
        with caplog.at_level(logging.WARNING, logger=mt_predict.logger.name):
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "spotforecast2_safe.multitask.base.get_weather_features",
                        side_effect=WeatherFetchError("simulated outage"),
                    )
                )
                stack.enter_context(
                    patch(
                        "spotforecast2_safe.multitask.base.get_calendar_features",
                        return_value=_frame(idx_predict, calendar_cols),
                    )
                )
                stack.enter_context(
                    patch(
                        "spotforecast2_safe.multitask.base.get_day_night_features",
                        return_value=_frame(idx_predict, day_night_cols),
                    )
                )
                stack.enter_context(
                    patch(
                        "spotforecast2_safe.multitask.base.get_holiday_features",
                        return_value=_frame(idx_predict, holiday_cols),
                    )
                )
                _stub_downstream(stack)
                mt_predict.build_exogenous_features()
        predict_cols = set(mt_predict.exogenous_features.columns)

        missing_at_predict = train_cols - predict_cols
        assert missing_at_predict == set(weather_cols), (
            "Predict-time exogenous schema should be missing exactly the "
            "weather columns when on_weather_failure='skip' swallows the "
            f"outage. Got missing={missing_at_predict}."
        )
        assert predict_cols.issubset(train_cols)
        # Non-weather columns must survive the skip — that is the whole
        # point of the graceful-degradation policy.
        for col in calendar_cols + day_night_cols + holiday_cols:
            assert col in predict_cols
        assert any("Open-Meteo fetch failed" in rec.message for rec in caplog.records)
