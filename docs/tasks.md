# Task Scripts

`spotforecast2` provides command-line task scripts for common forecasting workflows. These scripts are registered as console entry points and can be invoked directly via `uv run` or after package installation.

## Available Commands

| Command | Description |
|---------|-------------|
| `spotforecast-entsoe` | ENTSO-E energy forecasting pipeline (download, train, predict) |
| `spotforecast-demo` | Demonstration task comparing baseline, covariate, and custom models |
| `spotforecast-n2o1` | N-to-1 forecasting with weighted aggregation |
| `spotforecast-n2o1-df` | N-to-1 forecasting using a DataFrame input |
| `spotforecast-n2o1-cov` | N-to-1 forecasting with exogenous covariates |
| `spotforecast-n2o1-cov-df` | N-to-1 forecasting with covariates and DataFrame input |

---

## ENTSO-E Task

The `spotforecast-entsoe` command provides a unified CLI for the ENTSO-E energy forecasting pipeline.

### Subcommands

```bash
# Download data from ENTSO-E
uv run spotforecast-entsoe download --api-key YOUR_API_KEY 202301010000

# Train a model (lgbm or xgb)
uv run spotforecast-entsoe train lgbm --force

# Generate predictions and plot
uv run spotforecast-entsoe predict --plot

# Merge raw data files
uv run spotforecast-entsoe merge
```

!!! tip "API Key Management"
    Store your ENTSO-E API key in the `ENTSOE_API_KEY` environment variable to avoid passing it on every command:
    ```bash
    export ENTSOE_API_KEY="your-api-key-here"
    uv run spotforecast-entsoe download 202301010000
    ```

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
uv run spotforecast-entsoe predict --help
```

---

## Model Persistence

Trained models are saved to `~/spotforecast2_models/<task_name>/` by default. This allows:

- **Incremental retraining**: Only retrain when models are stale
- **Reproducibility**: Models are versioned by task and timestamp
- **Auditability**: Full training logs are stored alongside models

!!! warning "Safety-Critical Consideration"
    In production environments, always verify model checksums and training timestamps before deployment.
