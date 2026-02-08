# Welcome to spotforecast2

[![Version](https://img.shields.io/badge/version-{{ version_badge }}-blue.svg)](https://github.com/sequential-parameter-optimization/spotforecast2/releases)
[![GitHub](https://img.shields.io/badge/GitHub-spotforecast2-181717?logo=github)](https://github.com/sequential-parameter-optimization/spotforecast2)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

**spotforecast2** is a Python package for forecasting, combining the power of `sklearn`, `spotoptim`and `skforecast` with specialized utilities for "spot" forecasting.

## Quick Links

- 📦 [GitHub Repository](https://github.com/sequential-parameter-optimization/spotforecast2)
- 📚 [API Reference](api/data.md)
- 🚀 Current Version: **{{ version }}**

## Installation

* Download from GitHub

```bash
git clone https://github.com/sequential-parameter-optimization/spotforecast2.git
cd spotforecast2
```

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