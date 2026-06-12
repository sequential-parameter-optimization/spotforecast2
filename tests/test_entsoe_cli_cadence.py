# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the 7-day retraining cadence gate in entsoe_cli.train."""

import sys
from unittest.mock import patch

import pandas as pd

import spotforecast2.entsoe_cli as entsoe_cli


def _make_model_file(tmp_path, project: str, age_days: float):
    """Create a placeholder .joblib whose mtime is ``age_days`` in the past."""
    project_dir = tmp_path / "models" / project
    project_dir.mkdir(parents=True, exist_ok=True)
    f = project_dir / "forecaster_Actual Load.joblib"
    f.write_bytes(b"\x00")
    age_seconds = age_days * 24 * 3600
    now = pd.Timestamp.now(tz="UTC").timestamp()
    import os

    os.utime(f, (now - age_seconds, now - age_seconds))
    return f


def test_train_subcommand_skips_when_recent_model_exists(tmp_path, monkeypatch):
    """A model trained 1 day ago should NOT trigger a retrain."""
    monkeypatch.setattr(
        entsoe_cli, "get_cache_home", lambda *_args, **_kwargs: tmp_path
    )
    _make_model_file(tmp_path, "entsoe-lgbm", age_days=1)

    with patch.object(entsoe_cli, "_run_entsoe_pipeline") as mock_pipeline:
        monkeypatch.setattr(sys, "argv", ["spotforecast2-entsoe", "train", "lgbm"])
        entsoe_cli.main()

    mock_pipeline.assert_not_called()


def test_train_subcommand_retrains_with_force(tmp_path, monkeypatch):
    """``--force`` bypasses the cadence gate."""
    monkeypatch.setattr(
        entsoe_cli, "get_cache_home", lambda *_args, **_kwargs: tmp_path
    )
    _make_model_file(tmp_path, "entsoe-lgbm", age_days=1)

    with patch.object(entsoe_cli, "_run_entsoe_pipeline") as mock_pipeline:
        monkeypatch.setattr(
            sys, "argv", ["spotforecast2-entsoe", "train", "lgbm", "--force"]
        )
        entsoe_cli.main()

    mock_pipeline.assert_called_once()


def test_train_subcommand_retrains_when_no_previous_model(tmp_path, monkeypatch):
    """No saved model → retrain immediately."""
    monkeypatch.setattr(
        entsoe_cli, "get_cache_home", lambda *_args, **_kwargs: tmp_path
    )

    with patch.object(entsoe_cli, "_run_entsoe_pipeline") as mock_pipeline:
        monkeypatch.setattr(sys, "argv", ["spotforecast2-entsoe", "train", "lgbm"])
        entsoe_cli.main()

    mock_pipeline.assert_called_once()


def test_train_subcommand_retrains_when_model_is_old(tmp_path, monkeypatch):
    """A model older than retrain_max_age triggers a retrain."""
    monkeypatch.setattr(
        entsoe_cli, "get_cache_home", lambda *_args, **_kwargs: tmp_path
    )
    _make_model_file(tmp_path, "entsoe-lgbm", age_days=10)

    with patch.object(entsoe_cli, "_run_entsoe_pipeline") as mock_pipeline:
        monkeypatch.setattr(sys, "argv", ["spotforecast2-entsoe", "train", "lgbm"])
        entsoe_cli.main()

    mock_pipeline.assert_called_once()


def test_latest_saved_model_timestamp_returns_none_when_dir_missing(tmp_path):
    from spotforecast2_safe.configurator import ConfigEntsoe

    config = ConfigEntsoe()
    config.cache_home = tmp_path
    assert entsoe_cli._latest_saved_model_timestamp(config, "no-such") is None
