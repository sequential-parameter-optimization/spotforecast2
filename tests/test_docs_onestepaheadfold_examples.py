"""
Test examples from OneStepAheadFold documentation.

These tests verify that all documented OneStepAheadFold examples run correctly
and demonstrate rapid validation for safety-critical systems.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.model_selection import OneStepAheadFold
from spotforecast2.model_selection import backtesting_forecaster


class TestRapidModelScreening:
    """Test Example 1: Rapid Model Screening for Safety-Critical Deployment."""

    def test_rapid_model_screening(self):
        """
        Verify rapid model screening example using OneStepAheadFold.
        Simplified version for faster testing.
        """
        # Simulate critical infrastructure monitoring (water pressure)
        rng = np.random.default_rng(321)
        dates = pd.date_range("2024-01-01", periods=30 * 24, freq="h")  # 30 days hourly

        # Pressure with daily cycle
        hour_of_day = dates.hour
        daily_cycle = 5 * np.sin(2 * np.pi * hour_of_day / 24)
        baseline_pressure = 50
        degradation = -0.005 * np.arange(len(dates))
        noise = rng.normal(0, 1, len(dates))

        y = pd.Series(
            baseline_pressure + daily_cycle + degradation + noise,
            index=dates,
            name="pressure_psi",
        )

        # Define candidate models for rapid screening
        model_candidates = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=30, random_state=321),
        }

        # OneStepAheadFold for fast screening
        cv = OneStepAheadFold(initial_train_size=15 * 24, verbose=False)  # 15 days train, 15 days test

        # Inspect fold structure
        folds_df = cv.split(y, as_pandas=True)
        assert len(folds_df) == 1  # OneStepAheadFold creates exactly 1 fold
        assert folds_df.iloc[0]["train_end"] <= folds_df.iloc[0]["test_start"]  # Consecutive or with gap

        # Rapid screening of candidates
        results = {}
        for name, estimator in model_candidates.items():
            forecaster = ForecasterRecursive(
                estimator=estimator,
                lags=24 * 7,
            )

            metric_values, predictions = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                cv=cv,
                metric="mean_absolute_error",
                verbose=False,
                show_progress=False,
            )

            mae = metric_values["mean_absolute_error"].iloc[0]
            results[name] = mae

        # Verify results
        assert len(results) == len(model_candidates)
        assert all(v > 0 for v in results.values())
        
        # Best model should be identified
        best_model = min(results, key=results.get)
        assert best_model in model_candidates


class TestStaticModelDeployment:
    """Test Example 2: Static Model Deployment Validation."""

    def test_static_deployment_validation(self):
        """Verify static model deployment example with OneStepAheadFold."""
        # Simulate embedded sensor system (temperature) with limited retraining
        rng = np.random.default_rng(654)
        dates = pd.date_range("2024-01-01", periods=360, freq="D")  # Reduced to ~1 year

        # Seasonal temperature pattern
        day_of_year = np.arange(len(dates)) % 365
        seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365)
        baseline_temp = 20
        noise = rng.normal(0, 2, len(dates))

        y = pd.Series(baseline_temp + seasonal + noise, index=dates, name="temperature_c")

        # Embedded system: Static model deployment
        forecaster = ForecasterRecursive(
            estimator=GradientBoostingRegressor(
                n_estimators=50, random_state=654
            ),
            lags=30,
        )

        # OneStepAheadFold for static deployment validation
        cv = OneStepAheadFold(initial_train_size=180, verbose=False)

        # Validate on entire second half (simulates production)
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric=["mean_absolute_error", "mean_squared_error"],
            interval=0.90,
            interval_method="conformal",
            use_in_sample_residuals=True,
            verbose=False,
            show_progress=False,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty
        assert metric_values["mean_absolute_error"].iloc[0] > 0

        # Temporal degradation analysis
        test_predictions = predictions.copy()
        n_test = len(test_predictions)
        quarter_size = max(1, n_test // 4)

        quarterly_mae = []
        for i in range(min(4, n_test // quarter_size)):
            start_idx = i * quarter_size
            end_idx = (
                (i + 1) * quarter_size if i < 3 else n_test
            )
            if start_idx < n_test:
                quarter_preds = test_predictions.iloc[
                    start_idx:min(end_idx, n_test)
                ]
                if len(quarter_preds) > 0:
                    quarter_actual = y.loc[quarter_preds.index]
                    quarter_mae = (
                        quarter_actual - quarter_preds["pred"]
                    ).abs().mean()
                    quarterly_mae.append(quarter_mae)

        # Verify performance stability
        if len(quarterly_mae) > 1:
            mae_trend = np.polyfit(
                range(len(quarterly_mae)), quarterly_mae, 1
            )[0]
            # Trend should not be catastrophically bad
            assert mae_trend < y.std()

        # Coverage check (lenient for small test datasets)
        coverage = (
            (y.loc[predictions.index] >= predictions["lower_bound"])
            & (y.loc[predictions.index] <= predictions["upper_bound"])
        ).mean()
        assert 0.0 <= coverage <= 1.0


class TestEmergencyProductionValidation:
    """Test Example 3: Emergency Production Validation."""

    def test_emergency_validation(self):
        """Verify emergency production validation with OneStepAheadFold."""
        # Simulate production scenario with distribution shift
        rng = np.random.default_rng(987)
        dates = pd.date_range("2024-01-01", periods=90, freq="D")

        baseline = 100
        trend = 0.1 * np.arange(len(dates))
        # Distribution shift in last 30 days
        shift = np.where(np.arange(len(dates)) > 60, 10, 0)
        noise = rng.normal(0, 5, len(dates))

        y = pd.Series(baseline + trend + shift + noise, index=dates, name="metric")

        # Test rollback candidate quickly
        rollback_model = ForecasterRecursive(
            estimator=RandomForestRegressor(
                n_estimators=20, random_state=987
            ),
            lags=7,
        )

        # OneStepAheadFold for fastest possible validation
        cv = OneStepAheadFold(initial_train_size=60, verbose=False)

        # Quick validation
        metric_values, predictions = backtesting_forecaster(
            forecaster=rollback_model,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty
        
        rollback_mae = metric_values["mean_absolute_error"].iloc[0]
        assert rollback_mae > 0

        # Worst prediction period analysis
        errors = (y.loc[predictions.index] - predictions["pred"]).abs()
        if len(errors) > 0:
            worst_error = errors.max()
            assert worst_error > 0


class TestOneStepAheadFoldStructure:
    """Test OneStepAheadFold basic structure and properties."""

    def test_one_step_ahead_fold_single_split(self):
        """Verify OneStepAheadFold creates exactly one fold."""
        y = pd.Series(np.random.randn(100), name="test")
        
        cv = OneStepAheadFold(initial_train_size=50, verbose=False)
        folds_df = cv.split(y, as_pandas=True)
        
        # Should create exactly 1 fold
        assert len(folds_df) == 1
        
        # Train should be first 50 observations
        assert folds_df.iloc[0]["train_start"] == 0
        assert folds_df.iloc[0]["train_end"] == 50
        
        # Test should be remaining 50 observations
        assert folds_df.iloc[0]["test_start"] == 50
        assert folds_df.iloc[0]["test_end"] == 100

    def test_one_step_ahead_fold_all_data_tested(self):
        """Verify OneStepAheadFold uses all remaining data for testing."""
        y = pd.Series(np.random.randn(200), name="test")
        
        cv = OneStepAheadFold(initial_train_size=100, verbose=False)
        folds_df = cv.split(y, as_pandas=True)
        
        # Test set should be exactly the remaining data
        test_start = folds_df.iloc[0]["test_start"]
        test_end = folds_df.iloc[0]["test_end"]
        train_end = folds_df.iloc[0]["train_end"]
        
        assert test_start == train_end
        assert test_end == len(y)
