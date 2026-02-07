"""
Safety-Critical Forecasting Example: Energy Grid Load Prediction

This example demonstrates model validation for a safety-critical application
where forecast errors could lead to grid instability or blackouts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold

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

# Safety-critical configuration: Conservative forecaster with uncertainty quantification
forecaster = ForecasterRecursive(
    estimator=GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
    lags=24 * 7,  # One week of hourly data
    binner_kwargs={"n_bins": 10},  # Enable binned residuals for better intervals
)

# Cross-validation strategy: Realistic evaluation with weekly retraining
cv = TimeSeriesFold(
    steps=24,  # Forecast 24 hours ahead
    initial_train_size=24 * 30 * 6,  # 6 months initial training
    refit=24 * 7,  # Retrain weekly (safety-critical: fresh models)
    fixed_train_size=False,  # Expanding window (use all available data)
    fold_stride=24 * 7,  # Evaluate weekly
    gap=0,  # No gap (immediate forecasting)
)

# Perform backtesting with probabilistic forecasts
print("Running safety-critical backtesting...")
print("=" * 60)

metric_values, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    metric=["mean_absolute_error", "mean_squared_error"],
    interval=0.95,  # 95% prediction interval for risk management
    interval_method="conformal",  # Distribution-free guarantees
    use_in_sample_residuals=True,
    use_binned_residuals=True,  # Account for heteroscedasticity
    n_jobs=1,  # Sequential for safety (avoid race conditions)
    verbose=True,
    show_progress=True,
)

# Safety metrics: Evaluate forecast reliability
print("\nSafety-Critical Performance Metrics:")
print("=" * 60)
print(f"Mean Absolute Error: {metric_values['mean_absolute_error'].mean():.2f} MW")
print(f"Root Mean Squared Error: {np.sqrt(metric_values['mean_squared_error'].mean()):.2f} MW")

# Coverage analysis: Critical for safety applications
actual_coverage = (
    (y.loc[predictions.index] >= predictions["lower_bound"])
    & (y.loc[predictions.index] <= predictions["upper_bound"])
).mean()
print(f"\nPrediction Interval Coverage: {actual_coverage:.1%}")
print(f"Target Coverage: 95.0%")
print(f"Coverage Gap: {(actual_coverage - 0.95) * 100:+.1f} percentage points")

# Identify critical failures: Predictions outside safety bounds
safety_margin = 0.10  # 10% safety margin
upper_safety_limit = y.max() * (1 + safety_margin)
lower_safety_limit = y.min() * (1 - safety_margin)

critical_failures = (
    (predictions["upper_bound"] > upper_safety_limit)
    | (predictions["lower_bound"] < lower_safety_limit)
)

print(f"\nCritical Failures (exceeding safety bounds): {critical_failures.sum()}")
print(f"Failure Rate: {critical_failures.mean():.2%}")

# Visualization for safety review
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Top panel: Predictions with uncertainty
ax1 = axes[0]
sample_period = slice("2024-06-01", "2024-06-14")  # Two weeks for clarity
ax1.plot(
    y.loc[sample_period].index,
    y.loc[sample_period].values,
    label="Actual Load",
    color="black",
    linewidth=2,
    alpha=0.7,
)
pred_sample = predictions.loc[sample_period]
ax1.plot(
    pred_sample.index,
    pred_sample["pred"],
    label="Forecast",
    color="blue",
    linewidth=2,
)
ax1.fill_between(
    pred_sample.index,
    pred_sample["lower_bound"],
    pred_sample["upper_bound"],
    alpha=0.3,
    color="blue",
    label="95% Prediction Interval",
)
ax1.axhline(
    upper_safety_limit, color="red", linestyle="--", alpha=0.5, label="Safety Limits"
)
ax1.axhline(lower_safety_limit, color="red", linestyle="--", alpha=0.5)
ax1.set_ylabel("Grid Load (MW)")
ax1.set_title("Safety-Critical Energy Load Forecasting with Uncertainty Quantification")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Bottom panel: Forecast errors over time
ax2 = axes[1]
errors = y.loc[predictions.index] - predictions["pred"]
ax2.plot(errors.index, errors.values, color="red", alpha=0.6, linewidth=1)
ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax2.axhline(
    errors.std() * 2, color="orange", linestyle="--", alpha=0.5, label="±2σ"
)
ax2.axhline(-errors.std() * 2, color="orange", linestyle="--", alpha=0.5)
ax2.set_xlabel("Date")
ax2.set_ylabel("Forecast Error (MW)")
ax2.set_title("Forecast Error Distribution (Monitoring for Drift)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/safety_critical_forecasting.png", dpi=150, bbox_inches="tight")
print(f"\nVisualization saved to: /tmp/safety_critical_forecasting.png")

# Decision support: Model deployment recommendation
print("\n" + "=" * 60)
print("DEPLOYMENT DECISION SUPPORT")
print("=" * 60)

if actual_coverage >= 0.93 and critical_failures.mean() < 0.01:
    print("✓ RECOMMENDATION: APPROVE for production deployment")
    print("  - Coverage within acceptable range")
    print("  - Critical failure rate below threshold")
elif actual_coverage >= 0.90:
    print("⚠ RECOMMENDATION: CONDITIONAL APPROVAL with monitoring")
    print("  - Coverage acceptable but monitor closely")
    print("  - Implement automated alerts for coverage drift")
else:
    print("✗ RECOMMENDATION: REJECT deployment")
    print("  - Coverage below safety threshold")
    print("  - Additional model development required")

print("\nNext Steps:")
print("1. Review forecast errors for systematic patterns")
print("2. Validate on most recent data (temporal stability check)")
print("3. Conduct stress testing with extreme weather scenarios")
print("4. Establish monitoring dashboards for production")
