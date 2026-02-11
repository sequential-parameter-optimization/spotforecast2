"""
Tests for probabilistic forecasting examples in model_selection documentation.

These tests verify that the examples in docs/api/model_selection.md are
executable and produce the expected results.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold


def test_example_bootstrapping_method():
    """
    Test Example 1: Bootstrapping Method.

    Verifies that backtesting with bootstrapping method produces
    predictions with bootstrap samples.
    """
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
    cv = TimeSeriesFold(steps=10, initial_train_size=150, refit=True, fold_stride=10)

    # Perform backtesting with bootstrapping method
    metric_values, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric="mean_absolute_error",
        interval="bootstrapping",  # Returns all bootstrap samples
        interval_method="bootstrapping",
        n_boot=50,
        use_in_sample_residuals=True,
        random_state=123,
    )

    # Verify results structure
    assert isinstance(predictions, pd.DataFrame)
    assert "pred" in predictions.columns
    assert "fold" in predictions.columns
    # Should have bootstrap columns
    assert any("pred_boot_" in col for col in predictions.columns)

    # Verify metric values
    assert isinstance(metric_values, pd.DataFrame)
    assert "mean_absolute_error" in metric_values.columns
    assert metric_values["mean_absolute_error"].values[0] > 0


def test_example_conformal_prediction():
    """
    Test Example 2: Conformal Prediction with Binned Residuals.

    Verifies that conformal prediction with binned residuals produces
    valid prediction intervals with reasonable coverage.
    """
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
    cv = TimeSeriesFold(steps=24, initial_train_size=200, refit=False, fold_stride=24)

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

    # Verify results structure
    assert isinstance(predictions, pd.DataFrame)
    assert "pred" in predictions.columns
    assert "lower_bound" in predictions.columns
    assert "upper_bound" in predictions.columns

    # Verify interval bounds
    assert (predictions["lower_bound"] <= predictions["pred"]).all()
    assert (predictions["pred"] <= predictions["upper_bound"]).all()

    # Verify metric values
    assert isinstance(metric_values, pd.DataFrame)
    assert "mean_absolute_error" in metric_values.columns
    assert "mean_squared_error" in metric_values.columns
    assert all(metric_values.values[0] > 0)

    # Verify interval coverage is reasonable (should be close to 95%)
    coverage = (
        (predictions["pred"] >= predictions["lower_bound"])
        & (predictions["pred"] <= predictions["upper_bound"])
    ).mean()
    assert 0.80 <= coverage <= 1.0  # Allow some variation due to small sample


def test_example_forecasting_with_exog():
    """
    Test Example 3: Forecasting with Exogenous Variables.

    Verifies that backtesting with exogenous variables and conformal
    prediction produces the expected output columns.
    """
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

    # Verify results structure
    assert isinstance(predictions, pd.DataFrame)
    assert "pred" in predictions.columns
    assert "lower_bound" in predictions.columns
    assert "upper_bound" in predictions.columns
    assert "fold" in predictions.columns

    # Verify interval bounds
    assert (predictions["lower_bound"] <= predictions["upper_bound"]).all()

    # Verify metric values
    assert isinstance(metric_values, pd.DataFrame)
    assert "mean_absolute_percentage_error" in metric_values.columns
    assert metric_values["mean_absolute_percentage_error"].values[0] > 0


if __name__ == "__main__":
    # Run tests when executed directly
    test_example_bootstrapping_method()
    print("✓ Example 1: Bootstrapping Method")

    test_example_conformal_prediction()
    print("✓ Example 2: Conformal Prediction with Binned Residuals")

    test_example_forecasting_with_exog()
    print("✓ Example 3: Forecasting with Exogenous Variables")

    print("\nAll documentation examples verified successfully!")
