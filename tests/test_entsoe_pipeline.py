# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end tests for the ENTSO-E integration driven directly through MultiTask.

Exercises each ENTSO-E task variant (defaults, lazy, optuna, spotoptim, predict)
against an in-memory synthetic single-target DataFrame supplied via
config.data_loader.  No network calls — use_exogenous_features=False skips
the Open-Meteo weather fetch.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from spotforecast2_safe.configurator import ConfigEntsoe

from spotforecast2.multitask.multi import MultiTask
from spotforecast2.tasks.task_entsoe import entsoe_lgbm_factory


def _synthetic_entsoe_df(n_days: int = 30) -> pd.DataFrame:
    """Single-target ENTSO-E-shaped DataFrame (hourly, ``Time (UTC)`` index).

    Generates a deterministic seasonal load curve with a daily sine bump and
    weekly amplitude variation — realistic enough for the recursive
    forecaster to fit a model without producing degenerate residuals.
    """
    n = n_days * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "Time (UTC)"
    rng = np.random.default_rng(0)
    hours = np.arange(n) % 24
    days = np.arange(n) // 24
    load = (
        50_000
        + 10_000 * np.sin(hours * 2 * np.pi / 24)
        + 5_000 * np.sin(days * 2 * np.pi / 7)
        + rng.normal(0, 500, n)
    )
    return pd.DataFrame({"Actual Load": load}, index=idx)


def _make_loader(df: pd.DataFrame):
    """Wrap a fixed DataFrame as a ``config.data_loader`` callable."""

    def loader(_config):
        return df.copy()

    return loader


@pytest.fixture
def entsoe_config(tmp_path: Path) -> ConfigEntsoe:
    """Pre-wired ``ConfigEntsoe`` used by every pipeline call in this module."""
    df = _synthetic_entsoe_df(n_days=30)
    cfg = ConfigEntsoe(
        targets=["Actual Load"],
        agg_weights=[1.0],
        bounds=[(-1e9, 1e9)],
        index_name="Time (UTC)",
        use_exogenous_features=False,
        use_outlier_detection=False,
        predict_size=12,
    )
    cfg.data_loader = _make_loader(df)
    cfg.forecaster_factory = entsoe_lgbm_factory
    cfg.cache_home = str(tmp_path)
    return cfg


def _assert_forecast_shape(forecast: pd.DataFrame, predict_size: int) -> None:
    """Every pipeline result must yield a single-column ``forecast`` DataFrame."""
    assert isinstance(forecast, pd.DataFrame)
    assert list(forecast.columns) == ["forecast"]
    assert len(forecast) == predict_size
    assert forecast["forecast"].notna().all()


def _drive(cfg: ConfigEntsoe, task: str, cache_home: str, **overrides) -> dict:
    """Construct a MultiTask, wire project name, run all five pipeline steps.

    Returns the raw result dict from ``mt.run(show=False)``.
    """
    cfg.data_frame_name = "test-entsoe"
    if overrides:
        cfg.set_params(**overrides)
    mt = MultiTask(cfg, task=task, cache_home=cache_home, log_level=40)
    mt.prepare_data()
    mt.detect_outliers()
    mt.impute()
    mt.build_exogenous_features()
    return mt.run(show=False)


def test_entsoe_defaults_round_trip(entsoe_config, tmp_path):
    """``task="defaults"`` fits with factory defaults and returns a forecast."""
    result = _drive(entsoe_config, "defaults", str(tmp_path))
    forecast = result["future_pred"].to_frame("forecast")
    _assert_forecast_shape(forecast, entsoe_config.predict_size)


def test_entsoe_lazy_round_trip(entsoe_config, tmp_path):
    """``task="lazy"`` works for ENTSO-E with no prior tuning in cache."""
    result = _drive(entsoe_config, "lazy", str(tmp_path))
    forecast = result["future_pred"].to_frame("forecast")
    _assert_forecast_shape(forecast, entsoe_config.predict_size)


def test_entsoe_optuna_round_trip(entsoe_config, tmp_path):
    """``task="optuna"`` runs Bayesian tuning then trains."""
    result = _drive(entsoe_config, "optuna", str(tmp_path), n_trials_optuna=2)
    forecast = result["future_pred"].to_frame("forecast")
    _assert_forecast_shape(forecast, entsoe_config.predict_size)


def test_entsoe_spotoptim_round_trip(entsoe_config, tmp_path):
    """``task="spotoptim"`` runs surrogate-model tuning then trains."""
    result = _drive(
        entsoe_config,
        "spotoptim",
        str(tmp_path),
        n_trials_spotoptim=3,
        n_initial_spotoptim=2,
    )
    forecast = result["future_pred"].to_frame("forecast")
    _assert_forecast_shape(forecast, entsoe_config.predict_size)


def test_entsoe_predict_after_defaults(entsoe_config, tmp_path):
    """``task="predict"`` reads models saved by a prior ``task="defaults"`` run."""
    # Training step: must write models into cache before predict can load them.
    _drive(entsoe_config, "defaults", str(tmp_path))
    # Predict step: same cache_home and project_name as the training run.
    result = _drive(entsoe_config, "predict", str(tmp_path))
    forecast = result["future_pred"].to_frame("forecast")
    _assert_forecast_shape(forecast, entsoe_config.predict_size)
