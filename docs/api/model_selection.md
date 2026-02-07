# Model Selection Module

## Validation

::: spotforecast2.model_selection.validation.backtesting_forecaster

### Probabilistic Forecasting with Residuals

When using probabilistic forecasting methods (`interval` parameter), the forecaster requires residuals for generating prediction intervals:

- In-sample residuals: Automatically stored when `use_in_sample_residuals=True` (default). The `backtesting_forecaster` function handles this by setting `store_in_sample_residuals=True` during training.

- Out-of-sample residuals: Used when `use_in_sample_residuals=False`. These must be precomputed using the forecaster's `set_out_sample_residuals(y_true, y_pred)` method before calling `backtesting_forecaster`.

- Binned residuals: When `use_binned_residuals=True` (default), residuals are selected based on predicted values for improved interval accuracy. This requires the forecaster to have a binner configured during initialization.

For conformal prediction (`interval_method='conformal'`), the method automatically uses the appropriate residuals based on the `use_in_sample_residuals` setting.

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


## Time Series Cross-Validation

::: spotforecast2.model_selection.split_ts_cv.TimeSeriesFold

## One Step Ahead Fold

::: spotforecast2.model_selection.split_one_step.OneStepAheadFold

## Grid Search

::: spotforecast2.model_selection.grid_search

## Bayesian Search

::: spotforecast2.model_selection.bayesian_search

## Random Search

::: spotforecast2.model_selection.random_search
