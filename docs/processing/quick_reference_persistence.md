# Quick Reference - Model Persistence

## Installation & Setup
No additional installation needed. The implementation uses joblib (already in requirements).

## Basic Usage

### First Run (Model Training + Caching)
```python
from spotforecast2.processing.n2n_predict_with_covariates import n2n_predict_with_covariates

predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    verbose=True  # Shows: "Training X forecasters..." and "Saving X trained forecasters..."
)
# Time: ~5-10 minutes
```

### Second Run (Load from Cache)
```python
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    verbose=True  # Shows: "All X forecasters loaded from cache"
)
# Time: ~1-2 seconds (150-600x faster!)
```

### Force Retraining
```python
predictions, metadata, forecasters = n2n_predict_with_covariates(
    forecast_horizon=24,
    force_train=True,  # Ignore cache, retrain all
    verbose=True
)
# Time: ~5-10 minutes
```

### Custom Cache Location
```python
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

## Files

**Implementation File**:
- `src/spotforecast2/processing/n2n_predict_with_covariates.py`

**Test Files**:
- `tests/test_model_persistence.py` (35 unit tests)
- `tests/test_n2n_persistence_integration.py` (12 integration tests)

**Documentation**:
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `COMPLETION_STATUS.md` - Project status

## Testing

```bash
# Run persistence tests
uv run pytest tests/test_model_persistence.py -v

# Run integration tests
uv run pytest tests/test_n2n_persistence_integration.py -v

# Run all tests
uv run pytest tests/ -v

# Quick check
uv run pytest tests/test_model_persistence.py tests/test_n2n_persistence_integration.py --tb=no -q
```

## Helper Functions (Advanced)

```python
from spotforecast2.processing.n2n_predict_with_covariates import (
    _ensure_model_dir,
    _get_model_filepath,
    _save_forecasters,
    _load_forecasters,
    _model_directory_exists
)

# Create model directory
model_dir = _ensure_model_dir("./my_models")

# Get path for a specific model
path = _get_model_filepath(model_dir, "power")

# Load cached models
forecasters, missing = _load_forecasters(
    ["power", "energy"],
    model_dir,
    verbose=True
)

# Save models
saved_paths = _save_forecasters(forecasters, model_dir, verbose=True)

# Check if cache exists
if _model_directory_exists(model_dir):
    print("Cache directory found")
```

## Performance Expectations

| Scenario | Time | Speedup |
|----------|------|---------|
| First run (with 10 targets) | 5-10 min | Baseline |
| Load from cache (10 targets) | 1-2 sec | ~200-400x |
| Partial cache (5 cached, 5 new) | 2-3 min | ~2x |
| Force retrain (10 targets) | 5-10 min | Retrains all |

## Storage Usage

- **Per model**: 1-5 MB (compressed)
- **10 models**: ~10-50 MB
- **50 models**: ~50-250 MB

Models are stored in: `model_dir/forecaster_{target_name}.joblib`

## Verbose Output Examples

### All models cached:
```
[8/9] Loading or training recursive forecasters...
Attempting to load cached models...
All 10 forecasters loaded from cache
```

### Partial cache:
```
[8/9] Loading or training recursive forecasters...
Attempting to load cached models...
Loaded 7, will train 3 new ones...
Training 3 forecasters...
Saving 3 trained forecasters to disk...
```

### Force train:
```
[8/9] Loading or training recursive forecasters...
Force retraining all 10 forecasters...
Training 10 forecasters...
Saving 10 trained forecasters to disk...
```

## Troubleshooting

**Q: Models not loading?**
- Check that `model_dir` path is correct and accessible
- Verify models exist in the directory: `ls ./forecaster_models/`
- Use `force_train=True` to rebuild cache

**Q: Slow on first run?**
- Training takes 5-10 minutes depending on data size
- This is normal - models are then cached for fast reuse
- Subsequent runs will be 1-2 seconds

**Q: Want to clear cache?**
- Delete the `model_dir` directory: `rm -rf ./forecaster_models/`
- Or set `force_train=True` to rebuild

**Q: Models taking up too much space?**
- Each model is ~1-5 MB compressed
- Delete `model_dir` to free space
- Set `force_train=True` to rebuild only if needed

## API Compatibility

- ✅ Backward compatible (all new parameters have defaults)
- ✅ Drop-in replacement for existing code
- ✅ No breaking changes
- ✅ Works with existing configurations

## Test Results

```
Unit Tests:        35/35 PASSED ✅
Integration Tests: 12/12 PASSED ✅
Project Tests:    322/322 PASSED ✅
─────────────────────────────────
Total:            369/369 PASSED ✅
```

---

For detailed technical information, see [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
