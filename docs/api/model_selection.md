# Model Selection Module

## Model Selection for Safety-Critical Forecasting

The model selection module provides robust validation and cross-validation tools designed for production environments where forecast reliability is paramount. These tools enable rigorous model evaluation through time series backtesting, ensuring models perform consistently across different temporal conditions.

## Validation: Backtesting Forecaster

::: spotforecast2_safe.model_selection.backtesting_forecaster

### Design Philosophy for Safety-Critical Systems

In safety-critical environments (energy grid management, medical monitoring, industrial control systems), forecast failures can have severe consequences. The backtesting_forecaster function implements several defensive design patterns:

1. Temporal Integrity: Strict enforcement of temporal ordering prevents data leakage that could mask model weaknesses
2. Refit Strategy Control: Configurable refit intervals allow balancing between model freshness and computational cost
3. Probabilistic Quantification: Prediction intervals provide uncertainty estimates essential for risk management
4. Parallel Execution Safety: Careful handling of stateful operations during parallelization prevents race conditions

### Fallback Mechanisms in Production

The backtesting framework serves as a critical fallback validation layer:

- Pre-deployment Validation: Comprehensive backtesting before model deployment catches issues that unit tests miss
- Continuous Monitoring: Regular backtesting on recent data detects model degradation
- A/B Testing Foundation: Provides fair comparison framework for evaluating model updates
- Rollback Decision Support: Quantitative metrics guide decisions to revert problematic model changes

### Understanding Probabilistic Forecasting

When forecasting in safety-critical contexts, point predictions alone are insufficient. Prediction intervals quantify uncertainty, enabling downstream systems to make risk-aware decisions. The backtesting_forecaster supports two interval estimation methods:

1. Bootstrapping: Resamples residuals to generate empirical prediction distributions
2. Conformal Prediction: Provides distribution-free coverage guarantees under mild assumptions

Both methods require residuals (forecast errors) for interval construction:

- In-sample residuals: Computed from training data (default, always available)
- Out-of-sample residuals: Computed from held-out calibration data (more reliable, requires setup)
- Binned residuals: Stratified by prediction magnitude for heteroscedastic errors (recommended)

### Probabilistic Forecasting with Residuals

When using probabilistic forecasting methods (`interval` parameter), the forecaster requires residuals for generating prediction intervals:

- In-sample residuals: Automatically stored when `use_in_sample_residuals=True` (default). The `backtesting_forecaster` function handles this by setting `store_in_sample_residuals=True` during training.

- Out-of-sample residuals: Used when `use_in_sample_residuals=False`. These must be precomputed using the forecaster's `set_out_sample_residuals(y_true, y_pred)` method before calling `backtesting_forecaster`.

- Binned residuals: When `use_binned_residuals=True` (default), residuals are selected based on predicted values for improved interval accuracy. This requires the forecaster to have a binner configured during initialization.

For conformal prediction (`interval_method='conformal'`), the method automatically uses the appropriate residuals based on the `use_in_sample_residuals` setting.

### Complete Examples

#### Example 0: Safety-Critical Energy Grid Forecasting

This comprehensive example demonstrates model validation for a production energy grid management system where forecast reliability is mission-critical. The example shows defensive programming practices, uncertainty quantification, and deployment decision support.

```python
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
print("Safety-Critical Performance Metrics:")
print(f"Mean Absolute Error: {metric_values['mean_absolute_error'].mean():.2f} MW")
print(f"RMSE: {np.sqrt(metric_values['mean_squared_error'].mean()):.2f} MW")

# Coverage analysis: Critical for safety applications
actual_coverage = (
    (y.loc[predictions.index] >= predictions["lower_bound"])
    & (y.loc[predictions.index] <= predictions["upper_bound"])
).mean()
print(f"Prediction Interval Coverage: {actual_coverage:.1%} (target: 95.0%)")

# Decision support: Model deployment recommendation
if actual_coverage >= 0.93:
    print("✓ RECOMMENDATION: APPROVE for production deployment")
else:
    print("✗ RECOMMENDATION: REJECT - Coverage below safety threshold")
```

Key safety-critical design elements:

1. Expanding training window: Uses all historical data for maximum information
2. Regular retraining: Weekly updates prevent model staleness
3. Conformal intervals: Provides distribution-free coverage guarantees
4. Binned residuals: Accounts for heteroscedastic errors (variance changes with load level)
5. Sequential execution: Avoids parallelization race conditions in critical systems
6. Coverage monitoring: Validates that uncertainty estimates are calibrated
7. Deployment gates: Quantitative criteria for production approval

#### Example 1: Bootstrapping Method

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold

# Create sample time series data
rng = np.random.default_rng(123)
dates = pd.date_range("2020-01-01", periods=200, freq="D")
y = pd.Series(
    np.cumsum(rng.normal(loc=0.1, scale=1, size=200)) + 50,
    index=dates,
    name="value"
)

# Initialize forecaster
forecaster = ForecasterRecursive(
    estimator=Ridge(random_state=123),
    lags=14
)

# Configure cross-validation
cv = TimeSeriesFold(
    steps=10,
    initial_train_size=150,
    refit=True,
    fold_stride=10
)

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
    random_state=123
)

# Visualize results with bootstrap samples
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y.index, y.values, label="Actual", color="black", linewidth=2)
ax.plot(predictions.index, predictions["pred"], label="Prediction", color="blue", linewidth=2)

# Plot bootstrap samples for uncertainty visualization
boot_cols = [col for col in predictions.columns if col.startswith("pred_boot_")]
for col in boot_cols[:10]:  # Plot first 10 bootstrap samples
    ax.plot(predictions.index, predictions[col], alpha=0.1, color="gray")

ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.set_title("Bootstrapping Method: Predictions with Bootstrap Samples")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"MAE: {metric_values['mean_absolute_error'].values[0]:.3f}")
```

#### Example 2: Conformal Prediction with Binned Residuals

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold

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
    binner_kwargs={"n_bins": 10}
)

# Configure cross-validation
cv = TimeSeriesFold(
    steps=24,
    initial_train_size=200,
    refit=False,
    fold_stride=24
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
    random_state=456
)

# Visualize results with prediction intervals
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(y.index, y.values, label="Actual", color="black", linewidth=2, alpha=0.7)
ax.plot(predictions.index, predictions["pred"], label="Prediction", color="blue", linewidth=2)
ax.fill_between(
    predictions.index,
    predictions["lower_bound"],
    predictions["upper_bound"],
    alpha=0.3,
    color="blue",
    label="95% Prediction Interval"
)

ax.set_xlabel("Date")
ax.set_ylabel("Power")
ax.set_title("Conformal Prediction with Binned Residuals (95% Interval)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Verify prediction interval coverage
coverage = (
    (predictions["pred"] >= predictions["lower_bound"]) &
    (predictions["pred"] <= predictions["upper_bound"])
).mean()

print(f"Interval coverage: {coverage:.2%}")
print(metric_values)
```

#### Example 3: Forecasting with Exogenous Variables

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold

# Create time series with exogenous variable
rng = np.random.default_rng(789)
dates = pd.date_range("2020-01-01", periods=250, freq="D")
exog = pd.DataFrame({
    "temperature": rng.normal(20, 5, 250),
    "day_of_week": dates.dayofweek
}, index=dates)

y = pd.Series(
    10 + 0.5 * exog["temperature"] + 2 * (exog["day_of_week"] < 5) + rng.normal(0, 1, 250),
    index=dates,
    name="demand"
)

# Initialize forecaster
forecaster = ForecasterRecursive(
    estimator=LinearRegression(),
    lags=7
)

# Configure cross-validation
cv = TimeSeriesFold(
    steps=7,
    initial_train_size=180,
    refit=True,
    fold_stride=7
)

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
    random_state=789
)

# Visualize results with prediction intervals and exogenous variable
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top plot: Predictions with intervals
ax1.plot(y.index, y.values, label="Actual Demand", color="black", linewidth=2, alpha=0.7)
ax1.plot(predictions.index, predictions["pred"], label="Prediction", color="blue", linewidth=2)
ax1.fill_between(
    predictions.index,
    predictions["lower_bound"],
    predictions["upper_bound"],
    alpha=0.3,
    color="blue",
    label="90% Prediction Interval"
)
ax1.set_ylabel("Demand")
ax1.set_title("Demand Forecasting with Exogenous Variables (90% Interval)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom plot: Exogenous variable (temperature)
ax2.plot(exog.index, exog["temperature"], label="Temperature", color="red", linewidth=1.5)
ax2.set_xlabel("Date")
ax2.set_ylabel("Temperature (°C)")
ax2.set_title("Exogenous Variable: Temperature")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"MAPE: {metric_values['mean_absolute_percentage_error'].values[0]:.3f}")
```


## Time Series Cross-Validation: TimeSeriesFold

The TimeSeriesFold class provides a robust framework for splitting time series data into training and validation folds while respecting temporal ordering. This is critical for safety-critical systems where data leakage from future observations could create dangerously optimistic performance estimates.

::: spotforecast2.model_selection.split_ts_cv.TimeSeriesFold

### Design Philosophy for Safety-Critical Validation

In safety-critical forecasting (medical devices, autonomous systems, financial trading), improper validation can lead to catastrophic failures in production. TimeSeriesFold implements several defensive patterns:

1. Temporal Integrity Enforcement: Strict chronological ordering prevents look-ahead bias that would invalidate safety assessments
2. Realistic Retraining Simulation: Configurable refit strategies mirror actual production deployment patterns
3. Gap Handling: Models the delay between data availability and prediction requirements
4. Incomplete Fold Management: Handles edge cases at data boundaries that could cause production failures

### TimeSeriesFold as a Fallback Mechanism

The TimeSeriesFold class serves multiple roles in a defense-in-depth validation strategy:

- Primary Validation Layer: Provides the fundamental train/test split infrastructure for all model evaluation
- Degradation Detection: Regular backtesting with fixed fold configurations detects model performance decay over time
- A/B Test Infrastructure: Consistent fold generation ensures fair comparison between model versions
- Rollback Validation: Enables testing whether reverting to an older model would improve performance
- Stress Testing Framework: Configurable parameters allow testing models under various temporal conditions

### Understanding Fold Configuration Parameters

The fold configuration directly impacts validation realism and computational cost. Key tradeoffs:

1. initial_train_size: Larger values provide more stable models but reduce validation data
2. refit frequency: More frequent refitting increases realism but multiplies computational cost
3. fixed_train_size vs expanding window: Fixed mimics resource-constrained systems, expanding uses all available information
4. fold_stride: Smaller strides provide more evaluation points but increase overlap and computation
5. gap: Models real-world delays between data collection and prediction deployment

### Safety-Critical Configuration Patterns

Different safety-critical applications require different validation strategies:

1. High-Frequency Trading: Small gaps (seconds), frequent refit, fixed window to match production constraints
2. Medical Monitoring: Expanding window (use all patient history), moderate refit, strict temporal gaps
3. Energy Grid Management: Daily/weekly refit cycles, expanding window, minimal gaps for immediate response
4. Autonomous Vehicles: Very frequent refit (continuous learning), small fixed windows for real-time constraints

### Complete Examples

#### Example 1: Medical Device Validation - Expanding Window Strategy

This example demonstrates validation for a medical monitoring system where patient safety depends on reliable predictions. The expanding window strategy uses all available patient history.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold

# Simulate patient vital signs monitoring (heart rate)
rng = np.random.default_rng(123)
dates = pd.date_range("2024-01-01", periods=30 * 24 * 60, freq="min")  # 30 days, 1-minute intervals

# Baseline heart rate with circadian rhythm and random variations
hour_of_day = dates.hour + dates.minute / 60
circadian_pattern = 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak afternoon
baseline_hr = 70
noise = rng.normal(0, 3, len(dates))
# Simulate gradual patient deterioration
deterioration = np.linspace(0, 15, len(dates))

y = pd.Series(
    baseline_hr + circadian_pattern + deterioration + noise,
    index=dates,
    name="heart_rate_bpm",
)

# Medical device configuration: Conservative, expanding window
forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=50, random_state=123),
    lags=60,  # Last 60 minutes of data
)

# Validation strategy: Expanding window to use all patient history
cv = TimeSeriesFold(
    steps=30,  # Predict 30 minutes ahead
    initial_train_size=7 * 24 * 60,  # 7 days initial training
    refit=24 * 60,  # Retrain daily (balance freshness vs computation)
    fixed_train_size=False,  # Expanding: use all patient history
    fold_stride=24 * 60,  # Evaluate daily
    gap=5,  # 5-minute processing delay (realistic constraint)
    allow_incomplete_fold=True,  # Don't discard recent data
    verbose=True,
)

# Inspect fold structure before running expensive backtesting
folds_df = cv.split(y, as_pandas=True)
print("Fold Structure for Medical Device Validation:")
print(folds_df[["fold", "train_start", "train_end", "test_start", "test_end"]])
print(f"\nTotal folds: {len(folds_df)}")
print(f"Training data grows from {folds_df.iloc[0]['train_end'] - folds_df.iloc[0]['train_start']} "
      f"to {folds_df.iloc[-1]['train_end'] - folds_df.iloc[-1]['train_start']} observations")

# Run backtesting with the configured folds
metric_values, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    metric="mean_absolute_error",
    verbose=False,
    show_progress=True,
)

print(f"\nMedical Device Validation Results:")
print(f"Mean Absolute Error: {metric_values['mean_absolute_error'].mean():.2f} BPM")
print(f"Max Error: {(y.loc[predictions.index] - predictions['pred']).abs().max():.2f} BPM")

# Safety check: Verify no catastrophic errors
max_acceptable_error = 10.0  # BPM
catastrophic_errors = (y.loc[predictions.index] - predictions['pred']).abs() > max_acceptable_error
if catastrophic_errors.any():
    print(f"⚠ WARNING: {catastrophic_errors.sum()} predictions exceed {max_acceptable_error} BPM threshold")
else:
    print(f"✓ All predictions within {max_acceptable_error} BPM safety threshold")
```

Key medical device validation elements:

1. Expanding window: Uses complete patient history for maximum information
2. Daily retraining: Balances model freshness with computational constraints
3. Gap parameter: Models realistic processing delays in medical devices
4. Incomplete folds allowed: Ensures most recent data is evaluated
5. Safety thresholds: Explicit error bounds for clinical acceptability

#### Example 2: High-Frequency Trading - Fixed Window Strategy

This example demonstrates validation for a high-frequency trading system where computational resources are constrained and only recent data is relevant.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold

# Simulate high-frequency price data
rng = np.random.default_rng(456)
dates = pd.date_range("2024-01-01 09:30", periods=6.5 * 60 * 60, freq="s")  # Trading day, 1-second bars

# Price with mean reversion and volatility clustering
price = 100.0
prices = [price]
for i in range(len(dates) - 1):
    # Mean reversion + random walk
    drift = -0.0001 * (price - 100.0)
    volatility = 0.01 * (1 + 0.5 * abs(rng.normal()))
    price += drift + rng.normal(0, volatility)
    prices.append(price)

y = pd.Series(prices, index=dates, name="price_usd")

# HFT configuration: Fast, fixed window
forecaster = ForecasterRecursive(
    estimator=Ridge(alpha=0.1),  # Fast linear model
    lags=60,  # Last 60 seconds
)

# Validation strategy: Fixed window mimicking production constraints
cv = TimeSeriesFold(
    steps=10,  # Predict 10 seconds ahead
    initial_train_size=3600,  # 1 hour initial training
    refit=300,  # Retrain every 5 minutes (production-realistic)
    fixed_train_size=True,  # Fixed: only use recent data (memory constraint)
    fold_stride=300,  # Evaluate every 5 minutes
    gap=1,  # 1-second execution delay
    allow_incomplete_fold=False,  # Strict: only complete folds
    verbose=True,
)

# Inspect fold structure
folds_df = cv.split(y, as_pandas=True)
print("Fold Structure for HFT Validation:")
print(folds_df[["fold", "train_start", "train_end", "test_start", "test_end"]].head(10))
print(f"\nTotal folds: {len(folds_df)}")
print(f"Training window size: {folds_df.iloc[0]['train_end'] - folds_df.iloc[0]['train_start']} observations (constant)")

# Run backtesting
metric_values, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    metric="mean_absolute_error",
    verbose=False,
    show_progress=True,
)

print(f"\nHFT Validation Results:")
print(f"Mean Absolute Error: {metric_values['mean_absolute_error'].mean():.4f} USD")

# Profitability check: Can we beat transaction costs?
transaction_cost = 0.001  # $0.001 per trade
prediction_accuracy = metric_values['mean_absolute_error'].mean()
if prediction_accuracy < transaction_cost:
    print(f"✓ Prediction accuracy ({prediction_accuracy:.4f}) beats transaction costs ({transaction_cost:.4f})")
else:
    print(f"✗ Prediction accuracy ({prediction_accuracy:.4f}) exceeds transaction costs ({transaction_cost:.4f})")
    print("  Strategy not viable for production deployment")
```

Key HFT validation elements:

1. Fixed window: Mimics production memory constraints and recency bias
2. Frequent retraining: Matches realistic production update cycles
3. Small gap: Models minimal execution delay
4. No incomplete folds: Ensures all evaluations use complete test sets
5. Transaction cost analysis: Economic viability check before deployment

#### Example 3: Overlapping Folds for Robust Evaluation

This example demonstrates using overlapping folds to increase evaluation density without changing the forecast horizon.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, TimeSeriesFold

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
    estimator=GradientBoostingRegressor(n_estimators=50, random_state=789),
    lags=30,  # Last 30 days
)

# Overlapping folds: More evaluation points for robust assessment
cv = TimeSeriesFold(
    steps=7,  # Predict 7 days ahead
    initial_train_size=180,  # 6 months initial training
    refit=7,  # Retrain weekly
    fixed_train_size=False,  # Expanding window
    fold_stride=1,  # Advance 1 day at a time (creates overlap!)
    gap=0,
    allow_incomplete_fold=True,
    verbose=True,
)

# Inspect overlapping structure
folds_df = cv.split(y, as_pandas=True)
print("Overlapping Fold Structure:")
print(folds_df[["fold", "test_start", "test_end"]].head(15))
print(f"\nTotal folds: {len(folds_df)}")
print(f"Overlap: Each observation appears in up to {cv.steps} different test sets")

# Run backtesting
metric_values, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    metric="mean_absolute_error",
    verbose=False,
    show_progress=True,
)

print(f"\nIndustrial Monitoring Validation Results:")
print(f"Mean Absolute Error: {metric_values['mean_absolute_error'].mean():.2f} °C")
print(f"Number of predictions: {len(predictions)}")
print(f"Number of unique timestamps: {len(predictions.index.unique())}")

# Note: With overlapping folds, some timestamps have multiple predictions
# This provides multiple independent forecasts for the same period
```

Key overlapping fold elements:

1. fold_stride < steps: Creates overlapping test sets
2. Multiple forecasts per timestamp: Provides ensemble-like robustness assessment
3. Increased evaluation density: More data points for statistical significance
4. Computational cost: Proportional to number of folds (trade-off consideration)

## One Step Ahead Fold: Simplified Validation for Specific Use Cases

The OneStepAheadFold class provides a streamlined validation approach for one-step-ahead forecasting scenarios. Unlike TimeSeriesFold which creates multiple temporal folds, OneStepAheadFold creates a single train/test split optimized for evaluating models on all remaining data after initial training.

::: spotforecast2_safe.model_selection.OneStepAheadFold

### Design Philosophy for Safety-Critical Validation

OneStepAheadFold serves as a complementary validation strategy to TimeSeriesFold, particularly valuable in safety-critical contexts where:

1. Rapid Model Assessment: Quick validation without multiple retraining cycles reduces time-to-deployment
2. Maximum Test Coverage: Uses all post-training data for evaluation, maximizing statistical power
3. Computational Efficiency: Single training run minimizes resource consumption in constrained environments
4. Baseline Establishment: Provides fast baseline performance metrics before more expensive cross-validation

### OneStepAheadFold as a Fallback Mechanism

The OneStepAheadFold class serves critical roles in a layered validation strategy:

- Fast Sanity Check: Quick validation before committing to expensive multi-fold cross-validation
- Computational Fallback: When TimeSeriesFold is too expensive, OneStepAheadFold provides rapid assessment
- Maximum Data Utilization: Evaluates on all available post-training data without gaps
- Model Comparison Baseline: Establishes performance floor before testing retraining strategies
- Emergency Validation: When production issues require immediate model assessment with minimal computation

### When to Use OneStepAheadFold vs TimeSeriesFold

Choose OneStepAheadFold when:

1. Computational resources are severely constrained
2. You need rapid initial model assessment
3. The model will not be retrained in production (static deployment)
4. You want to maximize test set size for statistical significance
5. Establishing a performance baseline before more complex validation

Choose TimeSeriesFold when:

1. The model will be retrained in production (requires realistic refit simulation)
2. You need to assess model degradation over time
3. Computational resources allow multiple training runs
4. You want to test different retraining strategies
5. Production deployment involves rolling forecasts with periodic updates

### Understanding the Single Split Strategy

OneStepAheadFold creates exactly one fold:

- Training Set: First `initial_train_size` observations
- Test Set: All remaining observations after training set
- No Retraining: Model trained once, evaluated on all subsequent data
- No Gaps: Immediate evaluation after training period ends

This differs fundamentally from TimeSeriesFold's multiple overlapping or sequential folds with configurable retraining.

### Complete Examples

#### Example 1: Rapid Model Screening for Safety-Critical Deployment

This example demonstrates using OneStepAheadFold for fast initial screening of multiple model candidates before committing to expensive cross-validation.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, OneStepAheadFold

# Simulate critical infrastructure monitoring (water pressure)
rng = np.random.default_rng(321)
dates = pd.date_range("2024-01-01", periods=365 * 24, freq="h")

# Pressure with daily cycle and gradual degradation
hour_of_day = dates.hour
daily_cycle = 5 * np.sin(2 * np.pi * hour_of_day / 24)
baseline_pressure = 50  # PSI
degradation = -0.01 * np.arange(len(dates))  # Gradual pressure loss
noise = rng.normal(0, 1, len(dates))

y = pd.Series(
    baseline_pressure + daily_cycle + degradation + noise,
    index=dates,
    name="pressure_psi",
)

# Define candidate models for rapid screening
model_candidates = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=321),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=321),
}

# OneStepAheadFold: Fast screening with single train/test split
cv = OneStepAheadFold(
    initial_train_size=180 * 24,  # 6 months training
    verbose=True,
)

# Inspect the single fold structure
folds_df = cv.split(y, as_pandas=True)
print("OneStepAheadFold Structure:")
print(folds_df)
print(f"\nTraining observations: {folds_df['train_end'].iloc[0] - folds_df['train_start'].iloc[0]}")
print(f"Test observations: {folds_df['test_end'].iloc[0] - folds_df['test_start'].iloc[0]}")

# Rapid screening of all candidates
results = {}
print("\nRapid Model Screening Results:")
print("=" * 60)

for name, estimator in model_candidates.items():
    forecaster = ForecasterRecursive(
        estimator=estimator,
        lags=24 * 7,  # One week of hourly data
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
    print(f"{name:20s}: MAE = {mae:.3f} PSI")

# Select best model for further validation
best_model = min(results, key=results.get)
print(f"\n✓ Best model for detailed validation: {best_model}")
print(f"  MAE: {results[best_model]:.3f} PSI")
print(f"  Next step: Run TimeSeriesFold cross-validation on {best_model}")

# Safety check: Verify best model meets minimum requirements
max_acceptable_mae = 2.0  # PSI
if results[best_model] <= max_acceptable_mae:
    print(f"✓ Best model meets safety threshold ({max_acceptable_mae} PSI)")
else:
    print(f"✗ WARNING: Best model exceeds safety threshold")
    print(f"  Consider additional feature engineering or model development")
```

Key rapid screening elements:

1. Single split: Minimal computation for fast iteration
2. Multiple candidates: Screen many models quickly
3. Maximum test data: Uses all post-training data for robust assessment
4. Safety gates: Immediate feedback on viability
5. Workflow integration: Identifies candidates for deeper validation

#### Example 2: Static Model Deployment Validation

This example demonstrates validating a model that will be deployed without retraining, making OneStepAheadFold the appropriate validation strategy.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, OneStepAheadFold

# Simulate embedded sensor system (temperature) with limited compute for retraining
rng = np.random.default_rng(654)
dates = pd.date_range("2024-01-01", periods=730, freq="D")  # 2 years

# Seasonal temperature pattern
day_of_year = np.arange(len(dates)) % 365
seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365)
baseline_temp = 20
noise = rng.normal(0, 2, len(dates))

y = pd.Series(baseline_temp + seasonal + noise, index=dates, name="temperature_c")

# Embedded system: Model will be deployed statically (no retraining capability)
forecaster = ForecasterRecursive(
    estimator=GradientBoostingRegressor(n_estimators=100, random_state=654),
    lags=30,  # Last 30 days
)

# OneStepAheadFold: Matches static deployment (train once, predict forever)
cv = OneStepAheadFold(
    initial_train_size=365,  # Train on first year
    verbose=True,
)

print("Static Deployment Validation:")
print("=" * 60)

# Validate on entire second year (simulates production behavior)
metric_values, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=y,
    cv=cv,
    metric=["mean_absolute_error", "mean_squared_error"],
    interval=0.90,  # 90% prediction interval
    interval_method="conformal",
    use_in_sample_residuals=True,
    verbose=False,
    show_progress=True,
)

print(f"\nStatic Model Performance (Year 2):")
print(f"Mean Absolute Error: {metric_values['mean_absolute_error'].iloc[0]:.2f} °C")
print(f"RMSE: {np.sqrt(metric_values['mean_squared_error'].iloc[0]):.2f} °C")

# Temporal degradation analysis: Check if performance degrades over time
# Split test period into quarters
test_predictions = predictions.copy()
n_test = len(test_predictions)
quarter_size = n_test // 4

quarterly_mae = []
for i in range(4):
    start_idx = i * quarter_size
    end_idx = (i + 1) * quarter_size if i < 3 else n_test
    quarter_preds = test_predictions.iloc[start_idx:end_idx]
    quarter_actual = y.loc[quarter_preds.index]
    quarter_mae = (quarter_actual - quarter_preds["pred"]).abs().mean()
    quarterly_mae.append(quarter_mae)
    print(f"Quarter {i+1} MAE: {quarter_mae:.2f} °C")

# Degradation check: Is performance stable over time?
mae_trend = np.polyfit(range(4), quarterly_mae, 1)[0]
if mae_trend > 0.1:
    print(f"\n⚠ WARNING: Performance degrading over time (trend: +{mae_trend:.3f} °C/quarter)")
    print("  Consider implementing periodic retraining capability")
else:
    print(f"\n✓ Performance stable over time (trend: {mae_trend:+.3f} °C/quarter)")
    print("  Static deployment validated for production")

# Coverage stability check
coverage = (
    (y.loc[predictions.index] >= predictions["lower_bound"])
    & (y.loc[predictions.index] <= predictions["upper_bound"])
).mean()
print(f"\nPrediction Interval Coverage: {coverage:.1%} (target: 90%)")
```

Key static deployment elements:

1. Single training: Matches production constraint (no retraining)
2. Long test period: Validates sustained performance
3. Temporal degradation analysis: Detects performance decay
4. Coverage stability: Ensures uncertainty estimates remain calibrated
5. Deployment decision: Clear criteria for static vs dynamic deployment

#### Example 3: Emergency Production Validation

This example demonstrates using OneStepAheadFold for rapid validation when production issues require immediate model assessment.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, OneStepAheadFold
import time

# Simulate production scenario: Model performance suddenly degraded
# Need to quickly validate if rolling back to previous model version helps
rng = np.random.default_rng(987)
dates = pd.date_range("2024-01-01", periods=90, freq="D")

# Simulated production data with recent distribution shift
baseline = 100
trend = 0.1 * np.arange(len(dates))
# Distribution shift in last 30 days
shift = np.where(np.arange(len(dates)) > 60, 10, 0)
noise = rng.normal(0, 5, len(dates))

y = pd.Series(baseline + trend + shift + noise, index=dates, name="metric")

# Emergency scenario: Current model failing, test rollback candidate quickly
rollback_model = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=30, random_state=987),  # Faster
    lags=7,
)

# OneStepAheadFold: Fastest possible validation
cv = OneStepAheadFold(
    initial_train_size=60,  # Train on pre-shift data
    verbose=False,  # Suppress output for speed
)

print("Emergency Production Validation:")
print("=" * 60)
print("Scenario: Production model failing, testing rollback candidate...")

# Time the validation
start_time = time.time()

metric_values, predictions = backtesting_forecaster(
    forecaster=rollback_model,
    y=y,
    cv=cv,
    metric="mean_absolute_error",
    verbose=False,
    show_progress=False,
)

validation_time = time.time() - start_time

print(f"\nValidation completed in {validation_time:.2f} seconds")
print(f"Rollback Model MAE: {metric_values['mean_absolute_error'].iloc[0]:.2f}")

# Compare to current production model performance (simulated)
current_production_mae = 15.0  # Degraded performance
rollback_mae = metric_values["mean_absolute_error"].iloc[0]

print(f"\nProduction Comparison:")
print(f"Current Model MAE:  {current_production_mae:.2f}")
print(f"Rollback Model MAE: {rollback_mae:.2f}")
print(f"Improvement:        {current_production_mae - rollback_mae:.2f} ({(1 - rollback_mae/current_production_mae)*100:.1f}%)")

# Emergency decision criteria
if rollback_mae < current_production_mae * 0.8:  # 20% improvement threshold
    print(f"\n✓ RECOMMENDATION: APPROVE rollback")
    print(f"  Significant improvement detected")
    print(f"  Proceed with rollback to restore service")
else:
    print(f"\n✗ RECOMMENDATION: REJECT rollback")
    print(f"  Insufficient improvement")
    print(f"  Investigate root cause instead of rollback")

# Quick diagnostic: Where is the rollback model failing?
errors = (y.loc[predictions.index] - predictions["pred"]).abs()
worst_period_start = errors.idxmax()
print(f"\nWorst prediction period: {worst_period_start}")
print(f"Error magnitude: {errors.max():.2f}")
```

Key emergency validation elements:

1. Speed priority: Minimal computation for rapid decision
2. Rollback testing: Quick assessment of previous model version
3. Clear decision criteria: Quantitative thresholds for action
4. Diagnostic information: Identifies failure modes
5. Production context: Balances speed vs thoroughness appropriately

### Computational Efficiency Comparison

OneStepAheadFold vs TimeSeriesFold computational cost:

- OneStepAheadFold: 1 model training + 1 prediction pass
- TimeSeriesFold (10 folds, refit=True): 10 model trainings + 10 prediction passes
- TimeSeriesFold (10 folds, refit=False): 1 model training + 10 prediction passes

For expensive models or large datasets, OneStepAheadFold can be 10-100x faster than TimeSeriesFold with refit=True.

## Grid Search

::: spotforecast2.model_selection.grid_search

## spotoptim Search

::: spotforecast2.model_selection.spotoptim_search

## Bayesian Search

::: spotforecast2.model_selection.bayesian_search

## Random Search

::: spotforecast2.model_selection.random_search
