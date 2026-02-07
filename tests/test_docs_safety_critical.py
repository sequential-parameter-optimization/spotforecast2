"""
Test the safety-critical forecasting documentation example.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold


def test_safety_critical_example():
    """Test that the safety-critical example from documentation executes correctly."""
    # Simulate realistic energy load data with daily and weekly patterns
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=365 * 24, freq="h")

    # Base load + daily pattern + weekly pattern + noise
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    base_load = 5000
    daily_pattern = 2000 * np.sin(2 * np.pi * hour_of_day / 24)
    weekly_pattern = 500 * (day_of_week < 5).astype(float)  # Weekday boost
    noise = rng.normal(0, 200, len(dates))
    trend = np.linspace(0, 500, len(dates))  # Gradual load increase

    y = pd.Series(
        base_load + daily_pattern + weekly_pattern + trend + noise,
        index=dates,
        name="grid_load_mw",
    )

    # Safety-critical configuration
    forecaster = ForecasterRecursive(
        estimator=GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        lags=24 * 7,  # One week of hourly data
        binner_kwargs={"n_bins": 10},
    )

    # Cross-validation strategy
    cv = TimeSeriesFold(
        steps=24,
        initial_train_size=24 * 30 * 6,  # 6 months initial training
        refit=24 * 7,  # Retrain weekly
        fixed_train_size=False,
        fold_stride=24 * 7,
        gap=0,
    )

    # Perform backtesting
    metric_values, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric=["mean_absolute_error", "mean_squared_error"],
        interval=0.95,
        interval_method="conformal",
        use_in_sample_residuals=True,
        use_binned_residuals=True,
        n_jobs=1,
        verbose=False,
        show_progress=False,
    )

    # Assertions: Verify output structure
    assert isinstance(metric_values, pd.DataFrame)
    assert "mean_absolute_error" in metric_values.columns
    assert "mean_squared_error" in metric_values.columns
    assert len(metric_values) > 0

    assert isinstance(predictions, pd.DataFrame)
    assert "pred" in predictions.columns
    assert "lower_bound" in predictions.columns
    assert "upper_bound" in predictions.columns

    # Verify predictions are reasonable
    assert predictions["pred"].notna().all()
    assert (predictions["upper_bound"] > predictions["lower_bound"]).all()

    # Verify coverage calculation works
    actual_coverage = (
        (y.loc[predictions.index] >= predictions["lower_bound"])
        & (y.loc[predictions.index] <= predictions["upper_bound"])
    ).mean()
    assert 0.0 <= actual_coverage <= 1.0

    # Verify metrics are reasonable (not NaN or infinite)
    assert metric_values["mean_absolute_error"].notna().all()
    assert metric_values["mean_squared_error"].notna().all()
    assert np.isfinite(metric_values["mean_absolute_error"]).all()
    assert np.isfinite(metric_values["mean_squared_error"]).all()

    print("✓ Safety-critical example test passed")
    print(f"  MAE: {metric_values['mean_absolute_error'].mean():.2f} MW")
    print(f"  Coverage: {actual_coverage:.1%}")


if __name__ == "__main__":
    test_safety_critical_example()
