# ENTSO-E Energy Forecasting Guide

This guide provides comprehensive examples for using `spotforecast2` with ENTSO-E energy data.
Examples are organized from beginner to advanced, with each code snippet backed by automated tests.

## Prerequisites

Before running these examples, ensure you have:

1. `spotforecast2` installed: `pip install spotforecast2`
2. An ENTSO-E API key (optional for training examples)

---

## Configuration

### Default Configuration

The simplest way to get started is using the default configuration:

```python
from spotforecast2 import Config

config = Config()
print(config.API_COUNTRY_CODE)  # 'DE'
print(config.predict_size)      # 24
print(config.random_state)      # 314159
```

### Custom Configuration

Customize parameters for your specific use case:

```python
from spotforecast2 import Config
import pandas as pd

config = Config(
    api_country_code='FR',
    predict_size=48,
    refit_size=14,
    random_state=42
)
print(config.API_COUNTRY_CODE)  # 'FR'
print(config.predict_size)      # 48
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_country_code` | str | "DE" | ISO country code for ENTSO-E API |
| `predict_size` | int | 24 | Number of hours to predict ahead |
| `refit_size` | int | 7 | Number of days between model refits |
| `train_size` | Timedelta | 3 years | Training data window |
| `random_state` | int | 314159 | Random seed for reproducibility |
| `periods` | List[Period] | 5 periods | Cyclical feature encodings |

### Accessing Period Configurations

View the cyclical encoding periods:

```python
from spotforecast2 import Config

config = Config()
for period in config.periods:
    print(f"{period.name}: {period.n_periods} basis functions")
```

---

## Feature Engineering

### Period Dataclass

Periods define cyclical time features using radial basis functions:

```python
from spotforecast2_safe.data import Period

daily = Period(
    name='daily',
    n_periods=12,
    column='hour',
    input_range=(1, 24)
)
print(daily.name)        # 'daily'
print(daily.n_periods)   # 12
```

### RepeatingBasisFunction

Transform time features into smooth cyclical encodings:

```python
from spotforecast2_safe.preprocessing import RepeatingBasisFunction
import pandas as pd

rbf = RepeatingBasisFunction(
    n_periods=12,
    column='hour',
    input_range=(1, 24)
)

df = pd.DataFrame({'hour': range(1, 25)})
features = rbf.transform(df)
print(features.shape)  # (24, 12)
```

### ExogBuilder

Build complete exogenous feature sets including holidays and weekends:

```python
from spotforecast2_safe.preprocessing import ExogBuilder
from spotforecast2_safe.data import Period
import pandas as pd

periods = [
    Period(name='daily', n_periods=12, column='hour', input_range=(1, 24)),
    Period(name='weekly', n_periods=7, column='dayofweek', input_range=(0, 6)),
]

builder = ExogBuilder(periods=periods, country_code='DE')
X = builder.build(
    pd.Timestamp('2025-01-01', tz='UTC'),
    pd.Timestamp('2025-01-02', tz='UTC')
)
print(X.shape)  # (25, 21) - 12 + 7 + 2 (holiday, weekend)
```

### Using Config with ExogBuilder

Combine configuration and feature building:

```python
from spotforecast2 import Config
from spotforecast2_safe.preprocessing import ExogBuilder
import pandas as pd

config = Config()
builder = ExogBuilder(
    periods=config.periods,
    country_code=config.API_COUNTRY_CODE
)
X = builder.build(
    pd.Timestamp('2025-12-31', tz='UTC'),
    pd.Timestamp('2026-01-01', tz='UTC')
)
print(f"Generated {X.shape[1]} features for {X.shape[0]} hours")
```

---

## Data Preprocessing

### Linear Interpolation

Handle missing values in time series data:

```python
from spotforecast2_safe.preprocessing import LinearlyInterpolateTS
import pandas as pd
import numpy as np

ts = pd.Series(
    [1.0, np.nan, 3.0, np.nan, 5.0],
    index=pd.date_range('2025-01-01', periods=5, freq='h')
)

interpolator = LinearlyInterpolateTS()
ts_clean = interpolator.fit_transform(ts)

print(ts_clean.values)  # [1.0, 2.0, 3.0, 4.0, 5.0]
```

---

## Forecaster Models

### LightGBM Forecaster

Create a LightGBM-based recursive forecaster:

```python
from spotforecast2.tasks.task_entsoe import ForecasterRecursiveLGBM, config

model = ForecasterRecursiveLGBM(iteration=1)

print(model.name)             # 'lgbm'
print(model.random_state)     # 314159 (from config)
print(len(model.preprocessor.periods))  # 5 (from config)
```

### XGBoost Forecaster

Create an XGBoost-based recursive forecaster:

```python
from spotforecast2.tasks.task_entsoe import ForecasterRecursiveXGB, config

model = ForecasterRecursiveXGB(iteration=1, lags=24)

print(model.name)  # 'xgb'
```

### Custom Configuration Forecaster

Override default configuration values:

```python
from spotforecast2.tasks.task_entsoe import ForecasterRecursiveLGBM
from spotforecast2_safe.data import Period

custom_periods = [
    Period(name='hourly', n_periods=24, column='hour', input_range=(1, 24)),
]

model = ForecasterRecursiveLGBM(
    iteration=1,
    lags=48,
    periods=custom_periods,
    country_code='FR',
    random_state=42
)

print(len(model.preprocessor.periods))  # 1
print(model.preprocessor.country_code)  # 'FR'
```

---

## Using the Python API (Notebooks & Quarto)

### Full Prediction Pipeline

For users working in Jupyter Notebooks or Quarto, the entire ENTSO-E pipeline can be executed using the Python API. This approach is highly recommended for safety-critical research as it allows for precise control over time windows and hyperparameters.

```python
import pandas as pd
import os
from spotforecast2_safe.downloader.entsoe import download_new_data
from spotforecast2_safe.manager.trainer import handle_training as handle_training_safe
from spotforecast2_safe.manager.predictor import get_model_prediction as get_model_prediction_safe
from spotforecast2.manager.plotter import make_plot
from spotforecast2.tasks.task_entsoe import ForecasterRecursiveLGBM

# 1. Setup Time Windows (Last 3 years until last month)
now = pd.Timestamp.now(tz='UTC').floor('D')
current_month_start = now.replace(day=1)
last_month_start = (current_month_start - pd.Timedelta(days=1)).replace(day=1)

# 2. Download Data (Optional, requires ENTSOE_API_KEY)
api_key = os.environ.get("ENTSOE_API_KEY")
if api_key:
    download_new_data(api_key=api_key, start="202301010000")

# 3. Configure and Train
# Explicit parameters override global configuration for reproducibility
model_class = ForecasterRecursiveLGBM
model_name = "lgbm_advanced"

handle_training_safe(
    model_class=model_class,
    model_name=model_name,
    train_size=pd.Timedelta(days=3 * 365),
    end_dev=last_month_start.strftime("%Y-%m-%d %H:%M%z"),
)

# 4. Generate Predictions for the forecast horizon
# The predictor will automatically load the model trained above
predictions = get_model_prediction_safe(
    model_name=model_name,
    predict_size=24 * 31
)

# 5. Visualize Results
if predictions:
    make_plot(predictions)
```

---

## File Paths

### Data Home Directory

Access the data storage location:

```python
from spotforecast2_safe.data import get_data_home

data_home = get_data_home()
print(data_home)  # ~/spotforecast2_data or SPOTFORECAST2_DATA
```

---

## CLI Commands

### Download Data

```bash
# Download with API key
uv run spotforecast2-entsoe download --api-key YOUR_API_KEY 202301010000

# Download with date range
uv run spotforecast2-entsoe download 202301010000 202312312300

# Force re-download
uv run spotforecast2-entsoe download --force 202301010000
```

### Train Models

```bash
# Train LightGBM model
uv run spotforecast2-entsoe train lgbm

# Train XGBoost model
uv run spotforecast2-entsoe train xgb

# Force retraining
uv run spotforecast2-entsoe train lgbm --force
```

### Generate Predictions

```bash
# Predict with default model (lgbm)
uv run spotforecast2-entsoe predict

# Predict with specific model
uv run spotforecast2-entsoe predict lgbm
uv run spotforecast2-entsoe predict xgb

# Predict and generate plot
uv run spotforecast2-entsoe predict --plot
```

### Merge Data Files

```bash
uv run spotforecast2-entsoe merge
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ENTSOE_API_KEY` | ENTSO-E API key for data downloads |
| `SPOTFORECAST2_DATA` | Custom data directory (default: ~/spotforecast2_data) |

---

## Testing

All examples in this guide are validated by automated tests:

```bash
# Run documentation example tests
uv run pytest tests/test_docs_entsoe_examples.py -v

# Run all ENTSO-E tests
uv run pytest tests/test_tasks_entsoe.py -v
```

---

## See Also

- [Tasks Overview](tasks.md) - All available CLI commands
- [API Reference](api/preprocessing.md) - Detailed API documentation
- [Model Persistence](processing/model_persistence.md) - Saving and loading models
