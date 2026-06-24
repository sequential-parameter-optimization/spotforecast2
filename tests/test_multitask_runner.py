# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the spotforecast2 multitask runner (``run``).

The full package binds its own ``MultiTask`` and the wider task set to the
``run_with`` seam exposed by ``spotforecast2-safe``.  These tests cover the
behaviour that is specific to ``spotforecast2`` — namely that the auto-tuning
tasks ``"optuna"`` and ``"spotoptim"`` are *accepted* here (the safe runner
rejects them) — as well as the shared runner contract (clean returns an empty
DataFrame; end-to-end runs return a single-column ``forecast`` DataFrame;
``project_name`` and ``**overrides`` are forwarded to the config).
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2.multitask import run as run_from_package
from spotforecast2.multitask.runner import PIPELINE_TASKS, run
from spotforecast2_safe.configurator.config_multi import ConfigMulti
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.preprocessing import RollingFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_df(n_weeks: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * 7 * n_weeks
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    idx.name = "DateTime"
    return pd.DataFrame({"a": rng.normal(100, 10, n)}, index=idx)


def _fast_factory(config, *, weight_func=None, target=None):
    from lightgbm import LGBMRegressor

    return ForecasterRecursive(
        estimator=LGBMRegressor(
            n_estimators=5,
            random_state=config.random_state,
            verbose=-1,
        ),
        lags=6,
        window_features=RollingFeatures(stats=["mean"], window_sizes=6),
        weight_func=weight_func,
    )


def _minimal_cfg(**kwargs) -> ConfigMulti:
    defaults = dict(
        predict_size=6,
        use_exogenous_features=False,
        use_outlier_detection=False,
        auto_save_models=False,
        number_folds=2,
        forecaster_factory=_fast_factory,
    )
    defaults.update(kwargs)
    return ConfigMulti(**defaults)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_run_reexported_from_package(self):
        """``run`` exported via the package __init__ is the runner function."""
        assert run_from_package is run

    def test_pipeline_tasks_include_autotuning(self):
        """The full package advertises the auto-tuning tasks."""
        assert {"optuna", "spotoptim"} <= PIPELINE_TASKS
        assert "clean" not in PIPELINE_TASKS  # clean is not a pipeline task


# ---------------------------------------------------------------------------
# Task availability — auto-tuning tasks ACCEPTED here (unlike safe runner)
# ---------------------------------------------------------------------------


class TestRunTaskAvailability:
    def test_optuna_is_accepted(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(n_trials_optuna=2)
        result = run(
            config=cfg,
            task="optuna",
            dataframe=df,
            cache_home=tmp_path,
            project_name="optuna_ok",
        )
        assert isinstance(result, pd.DataFrame)
        assert "forecast" in result.columns
        assert len(result) == 6

    def test_spotoptim_is_accepted(self, tmp_path):
        df = _synth_df()
        cfg = _minimal_cfg(n_trials_spotoptim=2, n_initial_spotoptim=1)
        result = run(
            config=cfg,
            task="spotoptim",
            dataframe=df,
            cache_home=tmp_path,
            project_name="spotoptim_ok",
        )
        assert isinstance(result, pd.DataFrame)
        assert "forecast" in result.columns
        assert len(result) == 6

    def test_unknown_task_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown task"):
            run(task="banana", cache_home=tmp_path)

    def test_error_mentions_supported_tasks(self, tmp_path):
        with pytest.raises(ValueError, match="lazy"):
            run(task="unknown_xyz", cache_home=tmp_path)


# ---------------------------------------------------------------------------
# clean task
# ---------------------------------------------------------------------------


class TestRunCleanTask:
    def test_clean_returns_empty_dataframe(self, tmp_path):
        result = run(task="clean", cache_home=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_clean_with_existing_cache(self, tmp_path):
        cache = tmp_path / "sf2_cache"
        cache.mkdir()
        (cache / "models").mkdir()
        result = run(task="clean", cache_home=cache, project_name="test")
        assert result.empty


# ---------------------------------------------------------------------------
# End-to-end with synthetic data
# ---------------------------------------------------------------------------


class TestRunEndToEnd:
    def test_defaults_returns_forecast_dataframe(self, tmp_path):
        df = _synth_df()
        result = run(
            config=_minimal_cfg(),
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="test_e2e",
        )
        assert isinstance(result, pd.DataFrame)
        assert "forecast" in result.columns
        assert len(result) == 6
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result["forecast"].notna().all()

    def test_lazy_task_runs_without_cache(self, tmp_path):
        df = _synth_df()
        result = run(
            config=_minimal_cfg(),
            task="lazy",
            dataframe=df,
            cache_home=tmp_path,
            project_name="lazy_test",
        )
        assert "forecast" in result.columns
        assert len(result) == 6

    def test_project_name_forwarded_to_config(self, tmp_path):
        """``project_name`` sets ``config.data_frame_name``."""
        df = _synth_df()
        cfg = _minimal_cfg()
        run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="my_project",
        )
        assert cfg.data_frame_name == "my_project"

    def test_overrides_forwarded_to_config(self, tmp_path):
        """``**overrides`` are forwarded to ``config.set_params``."""
        df = _synth_df()
        cfg = _minimal_cfg()
        result = run(
            config=cfg,
            task="defaults",
            dataframe=df,
            cache_home=tmp_path,
            project_name="override_test",
            predict_size=4,
        )
        assert cfg.predict_size == 4
        assert len(result) == 4

    def test_unknown_override_key_raises(self, tmp_path):
        """An unknown override key is rejected by ``config.set_params``."""
        df = _synth_df()
        with pytest.raises((ValueError, TypeError, AttributeError)):
            run(
                config=_minimal_cfg(),
                task="defaults",
                dataframe=df,
                cache_home=tmp_path,
                not_a_real_param=123,
            )
