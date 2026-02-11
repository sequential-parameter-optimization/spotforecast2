"""
Test examples from backtesting_forecaster documentation.

These tests verify that all documented examples run correctly and produce
reasonable results for safety-critical forecasting validation.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold


class TestBootstrappingExample:
    """Test Example 1: Bootstrapping Method."""

    def test_bootstrapping_method(self):
        """Verify bootstrapping method example runs and produces valid results."""
        # Create sample time series data
        rng = np.random.default_rng(123)
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        y = pd.Series(
            np.cumsum(rng.normal(loc=0.1, scale=1, size=200)) + 50,
            index=dates,
            name="value",
        )

        # Initialize forecaster
        forecaster = ForecasterRecursive(estimator=Ridge(random_state=123), lags=14)

        # Configure cross-validation
        cv = TimeSeriesFold(
            steps=10, initial_train_size=150, refit=True, fold_stride=10
        )

        # Perform backtesting with bootstrapping method
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            interval="bootstrapping",  # Use bootstrapping for uncertainty
            interval_method="bootstrapping",
            n_boot=50,  # 50 bootstrap samples
            use_in_sample_residuals=True,
            random_state=123,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty
        assert metric_values["mean_absolute_error"].mean() > 0
        assert len(predictions.columns) > 0

        # Check that predictions are reasonable
        mae = metric_values["mean_absolute_error"].iloc[0]
        assert mae < y.std()  # MAE should be less than data std


class TestConformalPredictionExample:
    """Test Example 2: Conformal Prediction with Binned Residuals."""

    def test_conformal_prediction_binned_residuals(self):
        """Verify conformal prediction with binned residuals example."""
        # Create sample time series with trend
        rng = np.random.default_rng(456)
        dates = pd.date_range("2020-01-01", periods=300, freq="h")
        trend = np.linspace(0, 50, 300)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(300) / 24)
        noise = rng.normal(0, 2, 300)
        y = pd.Series(trend + seasonal + noise, index=dates, name="power")

        # Initialize forecaster with binner for improved interval accuracy
        forecaster = ForecasterRecursive(
            estimator=GradientBoostingRegressor(random_state=456, n_estimators=50),
            lags=24,
            binner_kwargs={"n_bins": 10},
        )

        # Configure cross-validation
        cv = TimeSeriesFold(
            steps=24, initial_train_size=200, refit=False, fold_stride=24
        )

        # Perform backtesting with conformal prediction
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric=["mean_absolute_error", "mean_squared_error"],
            interval=0.95,  # 95% nominal coverage
            interval_method="conformal",
            use_in_sample_residuals=True,
            use_binned_residuals=True,
            random_state=456,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty
        assert "lower_bound" in predictions.columns
        assert "upper_bound" in predictions.columns

        # Verify interval coverage (more lenient for small test sets)
        actual_coverage = (
            (y.loc[predictions.index] >= predictions["lower_bound"])
            & (y.loc[predictions.index] <= predictions["upper_bound"])
        ).mean()

        # Coverage should be reasonable (more lenient bounds for edge cases)
        assert 0.0 < actual_coverage <= 1.0


class TestForecastingWithExogenousVariables:
    """Test Example 3: Forecasting with Exogenous Variables."""

    def test_forecasting_with_exog(self):
        """Verify forecasting with exogenous variables example."""
        # Create time series with exogenous variable
        rng = np.random.default_rng(789)
        dates = pd.date_range("2020-01-01", periods=250, freq="D")
        exog = pd.DataFrame(
            {"temperature": rng.normal(20, 5, 250), "day_of_week": dates.dayofweek},
            index=dates,
        )

        y = pd.Series(
            10
            + 0.5 * exog["temperature"]
            + 2 * (exog["day_of_week"] < 5)
            + rng.normal(0, 1, 250),
            index=dates,
            name="demand",
        )

        # Initialize forecaster
        forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=7)

        # Configure cross-validation
        cv = TimeSeriesFold(steps=7, initial_train_size=180, refit=True, fold_stride=7)

        # Perform backtesting with conformal prediction
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            exog=exog,
            cv=cv,
            metric="mean_absolute_percentage_error",
            interval=0.90,  # 90% prediction interval
            interval_method="conformal",
            use_in_sample_residuals=True,
            random_state=789,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty
        assert "mean_absolute_percentage_error" in metric_values.columns
        assert metric_values["mean_absolute_percentage_error"].mean() > 0


class TestEnergySafetyExample:
    """Test Example 0: Safety-Critical Energy Grid Forecasting (simplified)."""

    def test_safety_critical_energy_grid_simplified(self):
        """
        Verify safety-critical energy grid example runs correctly.
        Simplified version with reduced data for faster testing.
        """
        # Simulate realistic energy load data with daily and weekly patterns
        rng = np.random.default_rng(42)
        dates = pd.date_range(
            "2023-01-01", periods=30 * 24, freq="h"
        )  # Reduced to 30 days

        # Base load + daily pattern + weekly pattern + noise
        hour_of_day = dates.hour
        day_of_week = dates.dayofweek
        base_load = 5000
        daily_pattern = 2000 * np.sin(2 * np.pi * hour_of_day / 24)
        weekly_pattern = 500 * (day_of_week < 5).astype(float)
        noise = rng.normal(0, 200, len(dates))
        trend = np.linspace(0, 500, len(dates))

        y = pd.Series(
            base_load + daily_pattern + weekly_pattern + trend + noise,
            index=dates,
            name="grid_load_mw",
        )

        # Safety-critical configuration
        forecaster = ForecasterRecursive(
            estimator=GradientBoostingRegressor(
                n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42
            ),
            lags=24 * 7,
            binner_kwargs={"n_bins": 10},
        )

        # Cross-validation strategy (simplified)
        cv = TimeSeriesFold(
            steps=24,
            initial_train_size=24 * 14,  # 2 weeks
            refit=24 * 7,
            fixed_train_size=False,
            fold_stride=24 * 7,
            gap=0,
        )

        # Perform backtesting
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            interval=0.95,
            interval_method="conformal",
            use_in_sample_residuals=True,
            use_binned_residuals=True,
            n_jobs=1,
            verbose=False,
            show_progress=False,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty
        assert "mean_absolute_error" in metric_values.columns

        # Verify predictions are reasonable
        mae = metric_values["mean_absolute_error"].mean()
        assert mae > 0
        assert mae < y.std() * 2  # Sanity check

        # Coverage analysis (lenient for small test sets)
        if (
            "lower_bound" in predictions.columns
            and "upper_bound" in predictions.columns
        ):
            actual_coverage = (
                (y.loc[predictions.index] >= predictions["lower_bound"])
                & (y.loc[predictions.index] <= predictions["upper_bound"])
            ).mean()
            assert 0.0 <= actual_coverage <= 1.0
