"""
Test examples from TimeSeriesFold documentation.

These tests verify that all documented TimeSeriesFold examples run correctly
and demonstrate proper temporal validation for safety-critical systems.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold


class TestMedicalDeviceExample:
    """Test Example 1: Medical Device Validation - Expanding Window Strategy."""

    def test_medical_device_expanding_window(self):
        """
        Verify medical device example with expanding window strategy.
        Simplified version for testing.
        """
        # Simulate patient vital signs monitoring (heart rate) - simplified
        rng = np.random.default_rng(123)
        dates = pd.date_range(
            "2024-01-01", periods=21 * 24 * 60, freq="min"
        )  # 21 days, 1-minute intervals

        # Create heart rate data with circadian pattern
        hour_of_day = dates.hour + dates.minute / 60
        circadian_pattern = 10 * np.sin(
            2 * np.pi * (hour_of_day - 6) / 24
        )  # Peak afternoon
        baseline_hr = 70
        noise = rng.normal(0, 3, len(dates))
        deterioration = np.linspace(0, 15, len(dates))

        y = pd.Series(
            baseline_hr + circadian_pattern + deterioration + noise,
            index=dates,
            name="heart_rate_bpm",
        )

        # Medical device configuration
        forecaster = ForecasterRecursive(
            estimator=RandomForestRegressor(n_estimators=30, random_state=123),
            lags=60,
        )

        # Expanding window validation
        cv = TimeSeriesFold(
            steps=30,
            initial_train_size=24 * 60 * 3,  # 3 days
            refit=24 * 60 * 2,  # Refit every 2 days
            fixed_train_size=False,  # Expanding window
            fold_stride=24 * 60 * 2,  # Stride 2 days
            gap=5,
            allow_incomplete_fold=True,
            verbose=False,
        )

        # Inspect fold structure
        folds_df = cv.split(y, as_pandas=True)
        assert len(folds_df) > 0

        # In expanding window mode, training data should not shrink
        # (train_end should be non-decreasing)
        if len(folds_df) >= 2:
            train_ends = folds_df["train_end"].tolist()
            for i in range(1, len(train_ends)):
                assert (
                    train_ends[i] >= train_ends[i - 1]
                ), "Expanding window should have non-decreasing train_end"

        # Run backtesting
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty
        assert metric_values["mean_absolute_error"].mean() > 0


class TestHighFrequencyTradingExample:
    """Test Example 2: High-Frequency Trading - Fixed Window Strategy."""

    def test_hft_fixed_window(self):
        """Verify HFT example with fixed window strategy."""
        # Simulate high-frequency price data - simplified
        rng = np.random.default_rng(456)
        dates = pd.date_range(
            "2024-01-01 09:30", periods=3600, freq="s"
        )  # 1 hour of data

        # Price with mean reversion
        price = 100.0
        prices = [price]
        for i in range(len(dates) - 1):
            drift = -0.0001 * (price - 100.0)
            volatility = 0.01 * (1 + 0.5 * abs(rng.normal()))
            price += drift + rng.normal(0, volatility)
            prices.append(price)

        y = pd.Series(prices, index=dates, name="price_usd")

        # HFT configuration
        forecaster = ForecasterRecursive(estimator=Ridge(alpha=0.1), lags=60)

        # Fixed window validation
        cv = TimeSeriesFold(
            steps=10,
            initial_train_size=600,  # 10 minutes
            refit=300,  # 5 minutes
            fixed_train_size=True,  # Fixed window
            fold_stride=300,
            gap=1,
            allow_incomplete_fold=False,
            verbose=False,
        )

        # Inspect fold structure
        folds_df = cv.split(y, as_pandas=True)
        assert len(folds_df) > 0

        # Check fixed window property (training size should be constant)
        train_sizes = (folds_df["train_end"] - folds_df["train_start"]).unique()
        assert len(train_sizes) == 1  # All training windows should be same size

        # Run backtesting
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

        # Verify results
        assert not metric_values.empty
        assert not predictions.empty


class TestOverlappingFoldsExample:
    """Test Example 3: Overlapping Folds for Robust Evaluation."""

    def test_overlapping_folds(self):
        """Verify overlapping folds example."""
        # Simulate industrial sensor data (temperature)
        rng = np.random.default_rng(789)
        dates = pd.date_range("2024-01-01", periods=365, freq="D")

        # Seasonal pattern + trend + noise
        day_of_year = np.arange(len(dates))
        seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365)
        trend = 0.01 * day_of_year
        noise = rng.normal(0, 2, len(dates))

        y = pd.Series(20 + seasonal + trend + noise, index=dates, name="temperature_c")

        # Industrial monitoring configuration
        forecaster = ForecasterRecursive(
            estimator=GradientBoostingRegressor(n_estimators=30, random_state=789),
            lags=30,
        )

        # Overlapping folds: fold_stride < steps creates overlap
        cv = TimeSeriesFold(
            steps=7,
            initial_train_size=180,
            refit=7,
            fixed_train_size=False,
            fold_stride=1,  # Advance 1 day = creates overlap
            gap=0,
            allow_incomplete_fold=True,
            verbose=False,
        )

        # Inspect overlapping structure
        folds_df = cv.split(y, as_pandas=True)
        assert len(folds_df) > cv.steps  # Overlapping should create many folds

        # Run backtesting
        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

        # Verify results with overlapping predictions
        assert not metric_values.empty
        assert not predictions.empty

        # With overlapping folds, we might have multiple predictions per timestamp
        unique_timestamps = len(predictions.index.unique())
        total_predictions = len(predictions)
        assert total_predictions >= unique_timestamps
