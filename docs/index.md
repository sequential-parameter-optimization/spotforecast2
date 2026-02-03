# Welcome to spotforecast2

**spotforecast2** is a Python package for forecasting, combining the power of `sklearn`, `spotoptim`and `skforecast` with specialized utilities for "spot" forecasting.

## Installation

* Download from GitHub

* Sync using uv
```bash
uv sync
```

## Features

- **Data Fetching**: Easy access to time series data.
- **Preprocessing**: Robust tools for curating, cleaning, and splitting data.
- **Forecasting**: A rich set of forecasting strategies (constantly extended).
- **Model Selection**: `spotoptim` and `optuna` search for hyperparameter tuning.
- **Weather Integration**: Utilities for fetching and using weather data in forecasts.

## Attributions

Parts of the code are ported from skforecast to reduce external dependencies.
Many thanks to the skforecast team for their great work!