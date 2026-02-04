# Model Persistence Guide

## Overview

This guide explains how to use the model persistence feature in spotforecast2, which provides scikit-learn-style caching of trained forecasters to disk.

## Quick Start

### First Run - Training and Caching
```python
from spotforecast2.processing.n2n_predict_with_covariates import n2n_predict_with_covariates

# Models are trained and cached automatically
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    verbose=True
)
# Time: ~5-10 minutes
```

### Second Run - Loading from Cache
```python
# Models are loaded from cache (much faster!)
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    verbose=True
)
# Time: ~1-2 seconds
```

### Force Retraining
```python
# Force retraining - ignore cache, retrain all models
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    force_train=True,
    verbose=True
)
# Time: ~5-10 minutes
```

### Custom Cache Location
```python
# Use custom directory for models
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    model_dir="/path/to/models",  # Default: "./forecaster_models"
    verbose=True
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_train` | bool | False | Force retraining, ignore cache |
| `model_dir` | str/Path | "./forecaster_models" | Cache directory location |

## Performance Impact

Typical execution times when forecasting for 10 target variables:

| Scenario | Time | Speedup |
|----------|------|---------|
| First run (train + save) | 5-10 min | Baseline |
| Subsequent runs (load) | 1-2 sec | **150-600x faster** |
| Partial cache (5 cached, 5 new) | 2-3 min | ~2x faster |
| Force retrain (all models) | 5-10 min | Full retraining |

## Storage Usage

Each model is approximately 1-5 MB when compressed:
- 10 models: ~10-50 MB total
- 50 models: ~50-250 MB total

Models are stored in the format: `model_dir/forecaster_{target_name}.joblib`

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

## Advanced Usage

### Working with Helper Functions

For advanced use cases, you can directly use the persistence helper functions:

```python
from spotforecast2.processing.n2n_predict_with_covariates import (
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

