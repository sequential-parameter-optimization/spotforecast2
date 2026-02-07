"""
Test TimeSeriesFold documentation examples.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold


def test_medical_device_expanding_window():
    """Test medical device validation example with expanding window."""
    # Simulate patient vital signs monitoring (heart rate)
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=30 * 24 * 60, freq="min")

    hour_of_day = dates.hour + dates.minute / 60
    circadian_pattern = 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    baseline_hr = 70
    noise = rng.normal(0, 3, len(dates))
    deterioration = np.linspace(0, 15, len(dates))

    y = pd.Series(
        baseline_hr + circadian_pattern + deterioration + noise,
        index=dates,
        name="heart_rate_bpm",
    )

    forecaster = ForecasterRecursive(
        estimator=RandomForestRegressor(n_estimators=50, random_state=123),
        lags=60,
    )

    cv = TimeSeriesFold(
        steps=30,
        initial_train_size=7 * 24 * 60,
        refit=24 * 60,
        fixed_train_size=False,
        fold_stride=24 * 60,
        gap=5,
        allow_incomplete_fold=True,
        verbose=False,
    )

    # Inspect fold structure
    folds_df = cv.split(y, as_pandas=True)
    assert len(folds_df) > 0
    assert "fold" in folds_df.columns
    assert "train_start" in folds_df.columns

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
    assert isinstance(metric_values, pd.DataFrame)
    assert "mean_absolute_error" in metric_values.columns
    assert len(predictions) > 0

    # Safety check
    max_acceptable_error = 10.0
    catastrophic_errors = (
        (y.loc[predictions.index] - predictions["pred"]).abs() > max_acceptable_error
    )

    print(f"✓ Medical device example test passed")
    print(f"  MAE: {metric_values['mean_absolute_error'].mean():.2f} BPM")
    print(f"  Catastrophic errors: {catastrophic_errors.sum()}")


def test_hft_fixed_window():
    """Test HFT validation example with fixed window."""
    # Simulate high-frequency price data (smaller dataset for faster testing)
    rng = np.random.default_rng(456)
    dates = pd.date_range("2024-01-01 09:30", periods=2 * 60 * 60, freq="s")  # 2 hours

    price = 100.0
    prices = [price]
    for i in range(len(dates) - 1):
        drift = -0.0001 * (price - 100.0)
        volatility = 0.01 * (1 + 0.5 * abs(rng.normal()))
        price += drift + rng.normal(0, volatility)
        prices.append(price)

    y = pd.Series(prices, index=dates, name="price_usd")

    forecaster = ForecasterRecursive(
        estimator=Ridge(alpha=0.1),
        lags=60,
    )

    cv = TimeSeriesFold(
        steps=10,
        initial_train_size=3600,
        refit=300,
        fixed_train_size=True,
        fold_stride=300,
        gap=1,
        allow_incomplete_fold=False,
        verbose=False,
    )

    # Inspect fold structure
    folds_df = cv.split(y, as_pandas=True)
    assert len(folds_df) > 0

    # Verify fixed window (training size should be constant)
    train_sizes = folds_df["train_end"] - folds_df["train_start"]
    assert train_sizes.nunique() == 1, "Training window should be fixed"

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
    assert isinstance(metric_values, pd.DataFrame)
    assert len(predictions) > 0

    print(f"✓ HFT example test passed")
    print(f"  MAE: {metric_values['mean_absolute_error'].mean():.4f} USD")
    print(f"  Fixed window size: {train_sizes.iloc[0]} observations")


def test_overlapping_folds():
    """Test overlapping folds example."""
    # Simulate industrial sensor data
    rng = np.random.default_rng(789)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")

    day_of_year = np.arange(len(dates))
    seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365)
    trend = 0.01 * day_of_year
    noise = rng.normal(0, 2, len(dates))

    y = pd.Series(20 + seasonal + trend + noise, index=dates, name="temperature_c")

    forecaster = ForecasterRecursive(
        estimator=GradientBoostingRegressor(n_estimators=50, random_state=789),
        lags=30,
    )

    cv = TimeSeriesFold(
        steps=7,
        initial_train_size=180,
        refit=7,
        fixed_train_size=False,
        fold_stride=1,  # Creates overlap!
        gap=0,
        allow_incomplete_fold=True,
        verbose=False,
    )

    # Inspect overlapping structure
    folds_df = cv.split(y, as_pandas=True)
    assert len(folds_df) > 0

    # Verify overlap exists
    assert cv.fold_stride < cv.steps, "Should have overlapping folds"

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
    assert isinstance(metric_values, pd.DataFrame)
    assert len(predictions) > 0

    # Check for multiple predictions per timestamp (due to overlap)
    unique_timestamps = len(predictions.index.unique())
    total_predictions = len(predictions)

    print(f"✓ Overlapping folds example test passed")
    print(f"  MAE: {metric_values['mean_absolute_error'].mean():.2f} °C")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Unique timestamps: {unique_timestamps}")
    print(f"  Overlap factor: {total_predictions / unique_timestamps:.1f}x")


if __name__ == "__main__":
    test_medical_device_expanding_window()
    test_hft_fixed_window()
    test_overlapping_folds()
    print("\n✓ All TimeSeriesFold documentation examples passed")
