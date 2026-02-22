# Task Scripts

`spotforecast2` provides command-line task scripts for common forecasting workflows. These scripts are registered as console entry points and can be invoked directly via `uv run` or after package installation.

## Available Commands

| Command | Description |
|---------|-------------|
| `spotforecast2-entsoe` | ENTSO-E energy forecasting pipeline (download, train, predict) |
| `spotforecast-demo` | Demonstration task comparing baseline, covariate, and custom models |
| `spotforecast-n2o1` | N-to-1 forecasting with weighted aggregation |
| `spotforecast-n2o1-df` | N-to-1 forecasting using a DataFrame input |
| `spotforecast-n2o1-cov` | N-to-1 forecasting with exogenous covariates |
| `spotforecast-n2o1-cov-df` | N-to-1 forecasting with covariates and DataFrame input |

---

## ENTSO-E Task

The `spotforecast2-entsoe` command provides a unified CLI for the ENTSO-E energy forecasting pipeline.

### Subcommands

```bash
# Download data from ENTSO-E
uv run spotforecast2-entsoe download --api-key YOUR_API_KEY 202301010000

# Train a model (lgbm or xgb)
uv run spotforecast2-entsoe train lgbm --force

# Generate predictions and plot (defaults to lgbm)
uv run spotforecast2-entsoe predict --plot

# Generate predictions with explicit model selection
uv run spotforecast2-entsoe predict lgbm --plot
uv run spotforecast2-entsoe predict xgb --plot

# Merge raw data files
uv run spotforecast2-entsoe merge
```

### Download arguments and time format

The positional argument 202301010000 is a UTC timestamp in the format YYYYMMDDHHMM.
It represents the start of the download window. You can provide either one timestamp
(start only) or two timestamps (start and end).

```bash
# Start only (end defaults to now, UTC)
uv run spotforecast2-entsoe download 202301010000

# Start and end (UTC)
uv run spotforecast2-entsoe download 202301010000 202312312300
```

Hidden arguments and defaults for download:

- --api-key or ENTSOE_API_KEY environment variable
- --force to re-download even if files already exist
- data home controlled by SPOTFORECAST2_DATA (default is ~/spotforecast2_data)

### Configuration

The ENTSO-E task uses a configuration class that can be customized programmatically.
All configuration parameters have sensible defaults but can be overridden when needed.

#### Using Default Configuration

```python
from spotforecast2 import Config

# Create default configuration instance
config = Config()

# Access configuration values
print(config.API_COUNTRY_CODE)  # 'DE'
print(config.predict_size)      # 24
print(config.train_size)        # Timedelta(days=1095)
```

#### Custom Configuration

```python
from spotforecast2 import Config
import pandas as pd

# Create custom configuration
custom_config = Config(
    api_country_code='DE',
    predict_size=48,
    refit_size=14,
    train_size=pd.Timedelta(days=365),
    random_state=42
)

# Use in your code
print(custom_config.API_COUNTRY_CODE)  # 'DE'
print(custom_config.predict_size)      # 48
```

#### Available Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_country_code` | str | "DE" | ISO country code for ENTSO-E API |
| `predict_size` | int | 24 | Number of hours to predict ahead |
| `refit_size` | int | 7 | Number of days between model refits |
| `train_size` | Timedelta | 3 years | Training data window |
| `end_train_default` | str | "2025-12-31 00:00+00:00" | Default training end date |
| `delta_val` | Timedelta | 10 weeks | Validation window size |
| `random_state` | int | 314159 | Random seed for reproducibility |
| `n_hyperparameters_trials` | int | 20 | Hyperparameter tuning trials |
| `lags_consider` | List[int] | [1..23] | Lag values for features |
| `periods` | List[Period] | 5 periods | Cyclical feature encodings |

For more details, see the [ConfigEntsoe API documentation](reference/manager.qmd).

### Time intervals for download, training, prediction, validation, and testing

Download interval is defined by the start/end timestamps passed to the download command.

Training, prediction, validation, and testing intervals are configured via the Config class.
The CLI uses default configuration values which can be modified programmatically:

- training end time: config.end_train_default (defaults to "2025-12-31 00:00+00:00")
- training window size: config.train_size (defaults to 3 years)
- prediction window: config.predict_size * config.refit_size hours

Validation and testing are derived from the prediction window:

- validation metrics use the first 24 hours of the prediction window
- testing metrics use the full prediction window

!!! note "Customizing Configuration"
    To use custom configuration values, you'll need to modify the task script
    to create a Config instance with your desired parameters. See the
    [Configuration](#configuration) section above for examples.

!!! tip "API Key Management"
    Store your ENTSO-E API key in the `ENTSOE_API_KEY` environment variable to avoid passing it on every command:
    ```bash
    export ENTSOE_API_KEY="your-api-key-here"
    echo $ENTSOE_API_KEY
    uv run spotforecast2-entsoe download 202301010000
    ```

### Visualize Results

The prediction plot shows the following graphs:

* **Total system load (actual)**: The real-time electricity demand (consumption) within the bidding zone. This includes network losses but excludes consumption for pumped storage and generating auxiliaries.
* **Total system load (model prediction)**: The demand forecast generated by the `spotforecast2` machine learning model (e.g., LightGBM or XGBoost) based on historical data and exogenous features.
* **Benchmark Forecast (e.g. ENTSOE)**: The reference forecast provided by the Transmission System Operators (TSOs) via the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).
* **Actual (last week)**: The actual system load from exactly one week ago at the same time, which serves as a seasonal baseline comparison.


The prediction plot is saved as an HTML file named `index.html` in the data home directory.
By default this is `~/spotforecast2_data/index.html` or the path defined by `SPOTFORECAST2_DATA`.

```bash
# Default location on macOS/Linux
open ~/spotforecast2_data/index.html

# If you use a custom data home
open "$SPOTFORECAST2_DATA/index.html"
```

Check the CLI logs for the exact path (look for "Plot saved to ...").

---

## Demo Task

The `spotforecast-demo` command runs a comparison of three forecasting approaches:

1. **Baseline**: Standard N-to-1 recursive forecaster
2. **Covariate-enhanced**: Includes weather, holidays, and cyclical features
3. **Custom LightGBM**: Optimized hyperparameters

```bash
# Run with default settings
uv run spotforecast-demo

# Force retraining and save plot
uv run spotforecast-demo --force_train true --html task_demo_plot.html
```

---

## N-to-1 Forecasting Tasks

These tasks implement multi-output time series forecasting with weighted aggregation.

### Basic N-to-1

```bash
uv run spotforecast-n2o1
```

### N-to-1 with DataFrame Input

```bash
uv run spotforecast-n2o1-df
```

### N-to-1 with Covariates

Includes weather data, holiday indicators, and cyclical time features.

```bash
uv run spotforecast-n2o1-cov
```

### N-to-1 with Covariates and DataFrame

```bash
uv run spotforecast-n2o1-cov-df
```

---

## Configuration

All tasks use sensible defaults but can be customized via:

- **Environment variables** (e.g., `ENTSOE_API_KEY`)
- **Command-line arguments** (use `--help` for details)
- **Configuration files** stored in `~/spotforecast2_models/`

```bash
# View available options for any command
uv run spotforecast-demo --help
uv run spotforecast2-entsoe predict --help
```

---

## Model Persistence

Trained models are saved to `~/spotforecast2_models/<task_name>/` by default. This allows:

- **Incremental retraining**: Only retrain when models are stale
- **Reproducibility**: Models are versioned by task and timestamp
- **Auditability**: Full training logs are stored alongside models

!!! warning "Safety-Critical Consideration"
    In production environments, always verify model checksums and training timestamps before deployment.

---

## Testing

The task scripts are covered by comprehensive safety-critical tests to ensure reliability in production environments.

### Running Tests

Run all ENTSO-E task tests:

```bash
uv run pytest tests/test_tasks_entsoe.py -v
```

Run specific test categories:

```bash
# Run only safety-critical tests
uv run pytest tests/test_tasks_entsoe.py::TestSafetyCriticalEntsoe -v

# Run parameter validation tests
uv run pytest tests/test_tasks_entsoe.py::TestSafetyCriticalEntsoe::test_train_lgbm_model_parameter_correctness -v

# Run with coverage
uv run pytest tests/test_tasks_entsoe.py --cov=spotforecast2.tasks.task_entsoe --cov-report=html
```

### Test Categories

The test suite includes:

- **Parameter Validation**: Ensures correct parameter passing between CLI and internal functions
- **Error Handling**: Validates graceful degradation and meaningful error messages
- **Data Validation**: Tests boundary conditions and edge cases
- **Integration Tests**: Verifies end-to-end functionality
- **Regression Tests**: Protects against known historical bugs
- **Model Selection Safety**: Prevents model mismatch in production pipelines

!!! tip "Continuous Testing"
    Run tests before deployment in production environments:
    ```bash
    uv run pytest tests/ -v --tb=short
    ```
