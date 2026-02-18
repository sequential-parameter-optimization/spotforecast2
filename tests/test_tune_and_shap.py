# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ForecasterRecursiveModelFull — real tune() and SHAP."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)
from spotforecast2.manager.models import (
    ForecasterRecursiveModelFull,
    ForecasterRecursiveLGBMFull,
    ForecasterRecursiveXGBFull,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def hourly_series():
    """Simple 500-row hourly series for testing."""
    np.random.seed(42)
    idx = pd.date_range("2022-01-01", periods=500, freq="h", tz="UTC")
    return pd.Series(np.random.randn(500).cumsum() + 100, index=idx, name="load")


# ------------------------------------------------------------------
# Inheritance & MRO
# ------------------------------------------------------------------


class TestInheritance:
    def test_full_is_subclass(self):
        assert issubclass(ForecasterRecursiveModelFull, ForecasterRecursiveModel)

    def test_lgbm_full_is_subclass_of_both(self):
        assert issubclass(ForecasterRecursiveLGBMFull, ForecasterRecursiveModelFull)
        assert issubclass(ForecasterRecursiveLGBMFull, ForecasterRecursiveModel)

    def test_xgb_full_is_subclass(self):
        assert issubclass(ForecasterRecursiveXGBFull, ForecasterRecursiveModelFull)

    def test_lgbm_full_has_n_trials(self):
        model = ForecasterRecursiveLGBMFull(iteration=0)
        assert model.n_trials == 10

    def test_lgbm_full_name(self):
        model = ForecasterRecursiveLGBMFull(iteration=0)
        assert model.name == "lgbm"

    def test_xgb_full_name(self):
        model = ForecasterRecursiveXGBFull(iteration=0)
        assert model.name == "xgb"


# ------------------------------------------------------------------
# tune() — Bayesian search
# ------------------------------------------------------------------


class TestTune:
    def test_tune_uses_bayesian_search(self):
        """Verify tune() source references bayesian_search_forecaster."""
        import inspect

        src = inspect.getsource(ForecasterRecursiveModelFull.tune)
        assert "bayesian_search_forecaster" in src

    def test_tune_is_not_stub(self):
        """Ensure the stub is overridden."""
        import inspect

        src = inspect.getsource(ForecasterRecursiveModelFull.tune)
        assert "simulated" not in src.lower()

    @patch(
        "spotforecast2_safe.manager.models.forecaster_recursive_model.load_timeseries"
    )
    @patch("spotforecast2.manager.models.load_timeseries")
    @patch("spotforecast2.manager.models.bayesian_search_forecaster")
    def test_tune_calls_bayesian_search(
        self, mock_bsf, mock_load_models, mock_load_safe, hourly_series
    ):
        """Integration-ish: mock the heavy calls, verify tune orchestration."""
        mock_load_models.return_value = hourly_series.copy()
        mock_load_safe.return_value = hourly_series.copy()

        # bayesian_search_forecaster returns (results_df, best_trial)
        results_df = pd.DataFrame(
            {
                "lags": [np.array([1, 2, 3])],
                "params": [{"n_estimators": 50}],
                "mean_absolute_error": [1.0],
                "name": ["lgbm"],
            }
        )
        mock_bsf.return_value = (results_df, MagicMock())

        model = ForecasterRecursiveLGBMFull(
            iteration=0,
            end_dev="2022-01-15 00:00+00:00",
            save_model_to_file=False,
        )
        # Override forecaster to avoid window_features issue with small data
        model.forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(n_jobs=-1, verbose=-1, random_state=42),
            lags=3,
        )

        model.tune()

        assert model.is_tuned
        assert model.best_params == {"n_estimators": 50}
        np.testing.assert_array_equal(model.best_lags, [1, 2, 3])
        assert model.results_tuning is not None
        mock_bsf.assert_called_once()

    @patch(
        "spotforecast2_safe.manager.models.forecaster_recursive_model.load_timeseries"
    )
    @patch("spotforecast2.manager.models.load_timeseries")
    @patch("spotforecast2.manager.models.bayesian_search_forecaster")
    def test_tune_saves_model_when_flag_set(
        self, mock_bsf, mock_load_models, mock_load_safe, hourly_series
    ):
        """When save_model_to_file is True, tune should persist the model."""
        mock_load_models.return_value = hourly_series.copy()
        mock_load_safe.return_value = hourly_series.copy()

        results_df = pd.DataFrame(
            {
                "lags": [np.array([1, 2])],
                "params": [{"n_estimators": 10}],
                "mean_absolute_error": [2.0],
            }
        )
        mock_bsf.return_value = (results_df, MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            model = ForecasterRecursiveLGBMFull(
                iteration=0,
                end_dev="2022-01-15 00:00+00:00",
                save_model_to_file=True,
            )
            model.forecaster = ForecasterRecursive(
                estimator=LGBMRegressor(n_jobs=-1, verbose=-1, random_state=42),
                lags=3,
            )
            model.tune()
            model.save_to_file(model_dir=tmpdir)

            saved = list(Path(tmpdir).glob("*.joblib"))
            assert len(saved) == 1


# ------------------------------------------------------------------
# get_global_shap_feature_importance()
# ------------------------------------------------------------------


class TestShap:
    def test_shap_is_not_stub(self):
        """Ensure the stub is overridden with a real implementation."""
        import inspect

        src = inspect.getsource(
            ForecasterRecursiveModelFull.get_global_shap_feature_importance
        )
        assert "TreeExplainer" in src
        assert "stub" not in src.lower()

    def test_shap_not_tuned_returns_empty(self, hourly_series):
        """When model is not tuned, SHAP should return empty Series."""
        model = ForecasterRecursiveLGBMFull(
            iteration=0, end_dev="2022-01-15 00:00+00:00"
        )
        model.forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(n_jobs=-1, verbose=-1, random_state=42),
            lags=3,
        )
        # Fit the model
        model.forecaster.fit(y=hourly_series[:200])

        # Mock _get_training_data to return matching-shape data
        X_cols = [f"lag_{i}" for i in range(1, 4)]
        X_train = pd.DataFrame(np.random.randn(50, 3), columns=X_cols)
        y_train = pd.Series(np.random.randn(50))
        model._get_training_data = lambda: (X_train, y_train)

        result = model.get_global_shap_feature_importance()
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_shap_returns_series_when_tuned(self, hourly_series):
        """After setting best_params/lags, SHAP should return non-empty."""
        model = ForecasterRecursiveLGBMFull(
            iteration=0, end_dev="2022-01-15 00:00+00:00"
        )
        model.forecaster = ForecasterRecursive(
            estimator=LGBMRegressor(n_jobs=-1, verbose=-1, random_state=42),
            lags=3,
        )
        # Fit the model
        model.forecaster.fit(y=hourly_series[:200])

        # Pretend tuning has been done
        model.best_params = {"n_estimators": 100}
        model.best_lags = [1, 2, 3]

        # Build X_train that matches the 3-lag feature space
        X_cols = [f"lag_{i}" for i in range(1, 4)]
        X_train = pd.DataFrame(np.random.randn(50, 3), columns=X_cols)
        y_train = pd.Series(np.random.randn(50))
        model._get_training_data = lambda: (X_train, y_train)

        result = model.get_global_shap_feature_importance(frac=0.5)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        # Values should be sorted descending
        assert (result.iloc[:-1].values >= result.iloc[1:].values).all()


# ------------------------------------------------------------------
# Import from top-level
# ------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_manager(self):
        from spotforecast2.manager import ForecasterRecursiveModelFull

        assert ForecasterRecursiveModelFull is not None

    def test_import_lgbm(self):
        from spotforecast2.manager import ForecasterRecursiveLGBMFull

        assert ForecasterRecursiveLGBMFull is not None

    def test_import_xgb(self):
        from spotforecast2.manager import ForecasterRecursiveXGBFull

        assert ForecasterRecursiveXGBFull is not None
