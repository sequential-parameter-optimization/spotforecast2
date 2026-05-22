# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end tests for the ENTSO-E integration under the unified `run()` entry point.

ADR-002 Step 6.  Exercises each ENTSO-E task variant (`lazy`, `defaults`,
`optuna`, `spotoptim`, `predict`) against an in-memory synthetic
single-target DataFrame supplied via `config.data_loader`.  No network
calls — `use_exogenous_features=False` skips the Open-Meteo weather fetch
that would otherwise need the internet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.configurator import ConfigEntsoe

from spotforecast2.multitask.runner import run
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
def entsoe_config_kwargs(tmp_path: Path) -> dict:
    """Common kwargs threaded into every ``run(...)`` call in this module."""
    df = _synthetic_entsoe_df(n_days=30)
    return {
        "config_cls": ConfigEntsoe,
        "project_name": "test-entsoe",
        "cache_home": str(tmp_path),
        "targets": ["Actual Load"],
        "agg_weights": [1.0],
        "bounds": [(-1e9, 1e9)],
        "data_loader": _make_loader(df),
        "forecaster_factory": entsoe_lgbm_factory,
        "index_name": "Time (UTC)",
        "use_exogenous_features": False,
        "use_outlier_detection": False,
        "predict_size": 12,
        "show": False,
    }


def _assert_forecast_shape(forecast: pd.DataFrame, predict_size: int) -> None:
    """Every run(...) returns a single-column ``forecast`` DataFrame."""
    assert isinstance(forecast, pd.DataFrame)
    assert list(forecast.columns) == ["forecast"]
    assert len(forecast) == predict_size
    assert forecast["forecast"].notna().all()


def test_run_entsoe_defaults_round_trip(entsoe_config_kwargs):
    """`task="defaults"` fits with factory defaults and returns a forecast."""
    forecast = run(task="defaults", **entsoe_config_kwargs)
    _assert_forecast_shape(forecast, entsoe_config_kwargs["predict_size"])


def test_run_entsoe_lazy_round_trip(entsoe_config_kwargs):
    """`task="lazy"` works for ENTSO-E with no prior tuning in cache."""
    forecast = run(task="lazy", **entsoe_config_kwargs)
    _assert_forecast_shape(forecast, entsoe_config_kwargs["predict_size"])


def test_run_entsoe_optuna_round_trip(entsoe_config_kwargs):
    """`task="optuna"` runs Bayesian tuning then trains."""
    forecast = run(
        task="optuna",
        n_trials_optuna=2,
        **entsoe_config_kwargs,
    )
    _assert_forecast_shape(forecast, entsoe_config_kwargs["predict_size"])


def test_run_entsoe_spotoptim_round_trip(entsoe_config_kwargs):
    """`task="spotoptim"` runs surrogate-model tuning then trains."""
    forecast = run(
        task="spotoptim",
        n_trials_spotoptim=3,
        n_initial_spotoptim=2,
        **entsoe_config_kwargs,
    )
    _assert_forecast_shape(forecast, entsoe_config_kwargs["predict_size"])


def test_run_entsoe_predict_after_defaults(entsoe_config_kwargs):
    """`task="predict"` reads models saved by a prior `task="defaults"` run."""
    run(task="defaults", **entsoe_config_kwargs)
    forecast = run(task="predict", **entsoe_config_kwargs)
    _assert_forecast_shape(forecast, entsoe_config_kwargs["predict_size"])
