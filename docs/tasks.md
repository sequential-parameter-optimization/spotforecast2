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

!!! tip "API Key Management"
    Store your ENTSO-E API key in the `ENTSOE_API_KEY` environment variable to avoid passing it on every command:
    ```bash
    export ENTSOE_API_KEY="your-api-key-here"
    echo $ENTSOE_API_KEY
    uv run spotforecast2-entsoe download 202301010000
    ```

### Visualize Results

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
