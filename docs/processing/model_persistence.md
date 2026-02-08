# Model Persistence Guide

## Overview

This guide explains how to use the model persistence feature in spotforecast2, which provides scikit-learn-style caching of trained forecasters to disk.

**Key Feature**: Model persistence is fully enabled with support for sample weight functions, providing significant speedup for repeated predictions!

## Installation & Setup

No additional installation needed! The implementation uses joblib (already in requirements) and the built-in `WeightFunction` class.

## Quick Start

### First Run - Training and Caching
```python
from spotforecast2_safe.processing.n2n_predict_with_covariates import n2n_predict_with_covariates

# Models are trained and cached automatically
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    verbose=True  # Shows: "Training X forecasters..." and "Saving X trained forecasters..."
)
```

### Second Run - Loading from Cache
```python
# Models are loaded from cache (much faster!)
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    verbose=True  # Shows: "All X forecasters loaded from cache"
)
```

### Force Retraining
```python
# Force retraining - ignore cache, retrain all models
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    force_train=True,  # Ignore cache, retrain all
    verbose=True
)
```

### Custom Cache Location
```python
# Use custom directory for models
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    model_dir="/path/to/models",  # Default: None (uses ~/spotforecast2_cache/forecasters)
    verbose=True
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_train` | bool | True | Force retraining, ignore cache |
| `model_dir` | str/Path | None | Cache directory location. If None, uses `get_cache_home()/forecasters` |

## Performance


**Default Cache Directory:**
- Location: `~/spotforecast2_cache/forecasters/`
- Environment Variable: `SPOTFORECAST2_CACHE` (overrides default directory)
- Models are stored in the format: `model_dir/forecaster_{target_name}.joblib`

## Verbose Output Examples

### All Models Loaded from Cache
```
[8/9] Loading or training recursive forecasters with exogenous variables...
  Attempting to load cached models...
  ✓ Loaded forecaster for power from ./forecaster_models/forecaster_power.joblib
  ✓ Loaded forecaster for energy from ./forecaster_models/forecaster_energy.joblib
  ...
  ✓ All 10 forecasters loaded from cache
```

### Partial Cache - Loading and Training
```
[8/9] Loading or training recursive forecasters with exogenous variables...
  Attempting to load cached models...
  ✓ Loaded forecaster for power from ./forecaster_models/forecaster_power.joblib
  ✓ Loaded 1 forecasters, will train 1 new ones
  Training forecaster for energy...
    ✓ Forecaster trained for energy
  Saving 1 trained forecasters to disk...
  ✓ Saved forecaster for energy to ./forecaster_models/forecaster_energy.joblib
  ✓ Total forecasters available: 2
```

### Force Retraining
```
[8/9] Loading or training recursive forecasters with exogenous variables...
  Force retraining all 2 forecasters...
  Training forecaster for power...
    ✓ Forecaster trained for power
  Training forecaster for energy...
    ✓ Forecaster trained for energy
  Saving 2 trained forecasters to disk...
  ✓ Saved forecaster for power to ./forecaster_models/forecaster_power.joblib
  ✓ Saved forecaster for energy to ./forecaster_models/forecaster_energy.joblib
  ✓ Total forecasters available: 2
```

## Key Implementation Details

### WeightFunction Class

The `WeightFunction` class enables model persistence with sample weights:

```python
from spotforecast2.preprocessing import WeightFunction

# Create picklable weight function
weights_series = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
weight_func = WeightFunction(weights_series)

# Use with forecaster - automatically persisted to disk!
forecaster = ForecasterRecursive(
    estimator=estimator,
    weight_func=weight_func
)
```

**Calling WeightFunction**:

```python
import pandas as pd
from spotforecast2.preprocessing import WeightFunction

weights = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
weight_func = WeightFunction(weights)

# For single index value
weight = weight_func(0)  # Returns: 1.0

# For multiple index values
weights = weight_func(pd.Index([0, 1, 2]))  # Returns: array([1.0, 0.9, 0.8])
```

**Benefits**:
- ✅ Fully picklable (works with joblib)
- ✅ No external dependencies
- ✅ No closure limitations
- ✅ Follows sklearn conventions

This approach ensures all trained models with sample weights can be persisted to disk without any external dependencies.

### Smart Caching Strategy

The system implements intelligent selective retraining:

1. **Cache Lookup** (if `force_train=False`)
   - Check if model cache directory exists
   - Attempt to load all target models from disk
   - Identify which targets are missing

2. **Selective Training**
   - Train only missing models (not cached)
   - Keep loaded models in memory
   - Saves significant computation time

3. **Auto-Save**
   - Newly trained models automatically saved to disk
   - Maintains cache consistency
   - No manual save required

4. **Force Retraining** (if `force_train=True`)
   - Clears cache directory
   - Trains all models from scratch
   - Useful for model updates or validation

## Working with Models

### Helper Functions (Advanced Usage)

For advanced use cases, you can directly use the persistence helper functions:

```python
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    _ensure_model_dir,
    _get_model_filepath,
    _save_forecasters,
    _load_forecasters,
    _model_directory_exists
)

# Create/ensure model directory exists
model_dir = _ensure_model_dir("./my_models")

# Get path for a specific model
path = _get_model_filepath(model_dir, "power")
# Returns: my_models/forecaster_power.joblib

# Load cached models
forecasters, missing = _load_forecasters(
    ["power", "energy", "temperature"],
    model_dir,
    verbose=True
)
# Returns: (loaded_forecasters_dict, missing_targets_list)

# Save models to disk
saved_paths = _save_forecasters(
    {"power": forecaster_obj, "energy": forecaster_obj},
    model_dir,
    verbose=True
)

# Check if cache directory exists
if _model_directory_exists(model_dir):
    print("Cache directory found")
```

### Programmatic Configuration

```python
import os
from spotforecast2_safe.data import get_cache_home

# Get default cache location
cache_dir = get_cache_home()

# Or set environment variable
os.environ['SPOTFORECAST2_CACHE'] = '/custom/cache/path'
cache_dir = get_cache_home()  # Now uses custom path

# Use in forecasting
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    model_dir=str(cache_dir / "forecasters"),
    verbose=True
)
```

## Implementation Files

**Core Implementation**:
- `src/spotforecast2/processing/n2n_predict_with_covariates.py` - Main forecasting function with persistence
- `src/spotforecast2/preprocessing/imputation.py` - WeightFunction class
- `src/spotforecast2/utils/forecaster_config.py` - Weight function initialization

**Test Files**:
- `tests/test_model_persistence.py` (35 unit tests)
- `tests/test_n2n_persistence_integration.py` (12 integration tests)
- `tests/test_weight_function_pickle.py` (6 pickling tests)
- `tests/test_cache_home.py` (14 cache home tests)

## Testing

```bash
# Run persistence tests
uv run pytest tests/test_model_persistence.py -v

# Run documentation example tests
uv run pytest tests/test_docs_model_persistence_examples.py -v

# Run integration tests
uv run pytest tests/test_n2n_persistence_integration.py -v

# Run weight function pickling tests
uv run pytest tests/test_weight_function_pickle.py -v

# Run all persistence-related tests
uv run pytest tests/test_model_persistence.py tests/test_docs_model_persistence_examples.py tests/test_n2n_persistence_integration.py tests/test_weight_function_pickle.py -v

# Quick check (all tests should pass)
uv run pytest tests/test_model_persistence.py tests/test_docs_model_persistence_examples.py tests/test_n2n_persistence_integration.py tests/test_weight_function_pickle.py --tb=no -q
```

**Documentation validation**: All examples in this guide are validated by `tests/test_docs_model_persistence_examples.py` with 43 comprehensive pytest cases.

## Troubleshooting

### Q: Models not loading?

**A:** Check that the `model_dir` path is correct and accessible:
```bash
# Verify models exist in the directory
ls ./forecaster_models/

# Check file permissions
ls -la ./forecaster_models/
```

Use `force_train=True` to rebuild the cache if needed:
```python
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    force_train=True,  # Rebuild cache
    model_dir="./forecaster_models",
    verbose=True
)
```

### Q: Slow on first run?

**A:** Training takes 5-10 minutes depending on data size and number of models. This is normal - models are then cached for fast reuse. Subsequent runs will be 1-2 seconds.

### Q: Want to clear cache?

**A:** Delete the model directory:
```bash
rm -rf ./forecaster_models/
```

Or set `force_train=True` to rebuild:
```python
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    force_train=True,
    model_dir="./forecaster_models",
    verbose=True
)
```

### Q: Models taking up too much space?

**A:** Each model is ~1-5 MB compressed with joblib. You can:
- Delete `model_dir` to free space: `rm -rf ./forecaster_models/`
- Use a different location: Set `model_dir` to a location with more space
- Set `force_train=True` to rebuild only if needed

### Q: How do I use custom estimators with persistence?

**A:** Custom estimators work with persistence as long as they're pickle-compatible (most scikit-learn compatible estimators are):

```python
from lightgbm import LGBMRegressor

custom_estimator = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    estimator=custom_estimator,
    force_train=False,  # Use cache if available
    model_dir="./models",
    verbose=True
)
```

## Technical Details

### Implementation

The model persistence feature uses **joblib** for serialization, following scikit-learn conventions:
- **Format**: Binary compressed files with `.joblib` extension
- **Compression**: joblib compress=3 (good balance of speed and size)
- **Location**: Configurable directory (default: `./forecaster_models/`)
- **Naming**: `forecaster_{target_name}.joblib`

#### Weight Function Pickling

The implementation uses a `WeightFunction` class to ensure sample weights can be pickled. This solves a common problem where local functions with closures cannot be serialized:

```python
from spotforecast2.preprocessing import WeightFunction
import pandas as pd

# Weights created from missing data analysis
weights = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])

# Wrap in WeightFunction (picklable, unlike local functions!)
weight_func = WeightFunction(weights)

# Can be pickled and saved to disk
import pickle
pickled = pickle.dumps(weight_func)

# Use with ForecasterRecursive
forecaster = ForecasterRecursive(
    estimator=estimator,
    lags=24,
    weight_func=weight_func  # Fully picklable!
)
```

This approach ensures all trained models with sample weights can be persisted to disk without any external dependencies.

### Smart Caching Strategy

The system implements intelligent selective retraining:

1. **Cache Lookup** (if `force_train=False`)
   - Check if model cache directory exists
   - Attempt to load all target models from disk
   - Identify which targets are missing

2. **Selective Training**
   - Train only missing models (not cached)
   - Keep loaded models in memory
   - Saves significant computation time

3. **Auto-Save**
   - Newly trained models automatically saved to disk
   - Maintains cache consistency
   - No manual save required

4. **Force Retraining** (if `force_train=True`)
   - Clears cache directory
   - Trains all models from scratch
   - Useful for model updates or validation

### API Compatibility

✅ **Backward Compatible** - All new parameters have defaults
✅ **Drop-in Replacement** - Works with existing code
✅ **No Breaking Changes** - Safe to upgrade

## See Also

- [API Reference - Forecasting](../api/forecaster.md)
- [API Reference - Data](../api/data.md)
- [Preprocessing Guide](../api/preprocessing.md)

