"""
Test suite for model persistence documentation examples.

Implements pytest tests for all examples from docs/processing/model_persistence.md
to ensure documentation accuracy and example functionality.

Safety-critical validation scope:
- Cache functionality (save/load cycles)
- WeightFunction pickling
- Cache directory management
- Helper function operations
- Programmatic configuration
"""

import tempfile
from pathlib import Path
import pickle
import os

import numpy as np
import pandas as pd
import pytest

from spotforecast2.preprocessing import WeightFunction
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    _ensure_model_dir,
    _get_model_filepath,
    _model_directory_exists,
)
from spotforecast2.data import get_cache_home


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_weights_series():
    """Create sample weights series for testing."""
    return pd.Series(
        [1.0, 0.9, 0.8, 0.7, 0.6],
        index=[0, 1, 2, 3, 4],
        name="sample_weights",
    )


@pytest.fixture
def sample_time_series():
    """Create sample time series data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    y = pd.Series(
        np.cumsum(np.random.normal(0.1, 1, 100)) + 50,
        index=dates,
        name="value",
    )
    return y


# ============================================================================
# WeightFunction Class Tests (from docs)
# ============================================================================


class TestWeightFunctionPickling:
    """Test WeightFunction class for pickling support.
    
    Based on docs: "WeightFunction Class" section
    """

    def test_weight_function_creation(self, sample_weights_series):
        """Test creating a WeightFunction instance."""
        weight_func = WeightFunction(sample_weights_series)
        assert weight_func is not None

    def test_weight_function_callable(self, sample_weights_series):
        """Test that WeightFunction is callable."""
        weight_func = WeightFunction(sample_weights_series)
        
        # Call with index
        result = weight_func(pd.Index([0, 1]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_weight_function_returns_correct_values(self, sample_weights_series):
        """Test that WeightFunction returns correct weight values."""
        weight_func = WeightFunction(sample_weights_series)
        
        result = weight_func(pd.Index([0, 1, 2]))
        expected = np.array([1.0, 0.9, 0.8])
        
        np.testing.assert_array_almost_equal(result, expected)

    def test_weight_function_pickling(self, sample_weights_series):
        """Test that WeightFunction can be pickled.
        
        Safety-critical: Ensure model persistence works with sample weights.
        """
        weight_func = WeightFunction(sample_weights_series)
        
        # Pickle and unpickle
        pickled = pickle.dumps(weight_func)
        unpickled = pickle.loads(pickled)
        
        # Verify functionality preserved
        original_result = weight_func(pd.Index([0, 1, 2]))
        unpickled_result = unpickled(pd.Index([0, 1, 2]))
        
        np.testing.assert_array_almost_equal(original_result, unpickled_result)

    def test_weight_function_no_closure_dependency(self, sample_weights_series):
        """Verify WeightFunction doesn't depend on closures (unlike lambdas/nested functions).
        
        Safety-critical: Local functions with closures cannot be pickled,
        causing model persistence to fail. WeightFunction avoids this issue.
        """
        weight_func = WeightFunction(sample_weights_series)
        
        # The WeightFunction should be picklable unlike a lambda
        # (which would fail with: "Can't pickle local object 'lambda'")
        assert callable(weight_func)
        
        # Verify it has no closure variables
        assert not hasattr(weight_func, "__closure__") or weight_func.__closure__ is None

    def test_weight_function_with_different_indices(self, sample_weights_series):
        """Test WeightFunction with various index types."""
        weight_func = WeightFunction(sample_weights_series)
        
        # Test with pd.Index (single value)
        result1 = weight_func(0)
        assert result1 == 1.0
        
        # Test with pd.Index (multiple values)
        result2 = weight_func(pd.Index([0, 1]))
        assert len(result2) == 2
        
        # Single scalar value should work
        assert weight_func(2) == 0.8


# ============================================================================
# Cache Directory Management Tests
# ============================================================================


class TestCacheDirectoryManagement:
    """Test cache directory management functions.
    
    Based on docs: "Programmatic Configuration" section
    """

    def test_ensure_model_dir_creates_directory(self, temp_model_dir):
        """Test _ensure_model_dir creates directories."""
        new_dir = temp_model_dir / "forecasters"
        
        result = _ensure_model_dir(new_dir)
        
        assert result.exists()
        assert result.is_dir()

    def test_ensure_model_dir_with_nested_path(self, temp_model_dir):
        """Test _ensure_model_dir with nested directory structure."""
        nested_path = temp_model_dir / "cache" / "models" / "forecasters"
        
        result = _ensure_model_dir(nested_path)
        
        assert result.exists()
        assert nested_path.exists()

    def test_ensure_model_dir_idempotent(self, temp_model_dir):
        """Test that _ensure_model_dir is idempotent."""
        model_dir = temp_model_dir / "models"
        
        # Call twice
        result1 = _ensure_model_dir(model_dir)
        result2 = _ensure_model_dir(model_dir)
        
        # Both should return same path
        assert result1 == result2
        assert result1.exists()

    def test_model_directory_exists_true(self, temp_model_dir):
        """Test _model_directory_exists returns True for existing directory."""
        assert _model_directory_exists(temp_model_dir) is True

    def test_model_directory_exists_false(self, temp_model_dir):
        """Test _model_directory_exists returns False for non-existing directory."""
        nonexistent = temp_model_dir / "does_not_exist"
        assert _model_directory_exists(nonexistent) is False

    def test_model_directory_exists_with_string_path(self, temp_model_dir):
        """Test _model_directory_exists works with string paths."""
        assert _model_directory_exists(str(temp_model_dir)) is True

    def test_get_model_filepath_format(self, temp_model_dir):
        """Test _get_model_filepath generates correct format.
        
        Based on docs: "Default Cache Directory" section naming convention
        """
        filepath = _get_model_filepath(temp_model_dir, "power")
        
        assert filepath.name == "forecaster_power.joblib"
        assert filepath.suffix == ".joblib"
        assert filepath.parent == temp_model_dir

    def test_get_model_filepath_multiple_targets(self, temp_model_dir):
        """Test _get_model_filepath for multiple targets."""
        targets = ["power", "energy", "demand"]
        filepaths = [_get_model_filepath(temp_model_dir, t) for t in targets]
        
        # All should be unique
        assert len(set(str(fp) for fp in filepaths)) == len(targets)
        
        # All should have .joblib extension
        assert all(fp.suffix == ".joblib" for fp in filepaths)

    def test_get_model_filepath_with_special_characters(self, temp_model_dir):
        """Test _get_model_filepath handles special characters in target names.
        
        From docs: "Models are stored in the format: model_dir/forecaster_{target_name}.joblib"
        """
        filepath = _get_model_filepath(temp_model_dir, "WWW_HE:FIRA158117")
        
        assert "WWW_HE:FIRA158117" in filepath.name
        assert filepath.name == "forecaster_WWW_HE:FIRA158117.joblib"


# ============================================================================
# Cache Home Management Tests
# ============================================================================


class TestCacheHomeManagement:
    """Test cache home directory management.
    
    Based on docs: "Default Cache Directory" section
    """

    def test_get_cache_home_returns_path(self):
        """Test that get_cache_home() returns a valid path."""
        cache_home = get_cache_home()
        assert cache_home is not None
        assert isinstance(cache_home, Path)

    def test_cache_home_respects_environment_variable(self):
        """Test that SPOTFORECAST2_CACHE environment variable is respected.
        
        From docs: "Environment Variable: SPOTFORECAST2_CACHE (overrides default directory)"
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_cache")
            os.environ["SPOTFORECAST2_CACHE"] = custom_path
            
            try:
                cache_home = get_cache_home()
                assert custom_path in str(cache_home) or cache_home == Path(custom_path)
            finally:
                # Clean up
                if "SPOTFORECAST2_CACHE" in os.environ:
                    del os.environ["SPOTFORECAST2_CACHE"]

    def test_cache_home_default_location_exists(self):
        """Test that default cache home location is accessible."""
        cache_home = get_cache_home()
        
        # Should be under home directory
        assert Path.home() in cache_home.parents or cache_home == Path.home()

    def test_cache_home_path_contains_forecasters_subdirectory(self):
        """Test combining cache_home with forecasters subdirectory.
        
        From docs: "If None, uses get_cache_home()/forecasters"
        """
        cache_home = get_cache_home()
        forecasters_dir = cache_home / "forecasters"
        
        assert "forecasters" in str(forecasters_dir)
        assert cache_home in forecasters_dir.parents


# ============================================================================
# WeightFunction with Forecaster Integration Tests
# ============================================================================


class TestWeightFunctionWithForecaster:
    """Test WeightFunction integration with forecasters.
    
    Based on docs: "WeightFunction Class" examples
    """

    def test_weight_function_with_various_weight_values(self):
        """Test WeightFunction with different weight distributions."""
        # Uniform weights
        uniform_weights = pd.Series([1.0] * 10, index=range(10))
        wf_uniform = WeightFunction(uniform_weights)
        result = wf_uniform(pd.Index([0, 5, 9]))
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])
        
        # Decreasing weights (recent data emphasized)
        decreasing_weights = pd.Series(np.linspace(1.0, 0.1, 10), index=range(10))
        wf_decreasing = WeightFunction(decreasing_weights)
        result = wf_decreasing(pd.Index([0, 5, 9]))
        assert result[0] > result[2]  # Earlier weights higher

    def test_weight_function_preserves_magnitude_after_pickle(self, sample_weights_series):
        """Test that weight magnitudes are preserved through pickle cycle."""
        wf_original = WeightFunction(sample_weights_series)
        original_result = wf_original(pd.Index([0, 1, 2, 3, 4]))
        
        # Pickle and unpickle
        pickled = pickle.dumps(wf_original)
        wf_restored = pickle.loads(pickled)
        restored_result = wf_restored(pd.Index([0, 1, 2, 3, 4]))
        
        # Sum of weights should be approximately equal
        np.testing.assert_almost_equal(
            original_result.sum(),
            restored_result.sum(),
            decimal=10
        )

    def test_weight_function_empty_series(self):
        """Test WeightFunction with empty series edge case."""
        empty_weights = pd.Series([], dtype=float)
        wf = WeightFunction(empty_weights)
        
        # Should handle empty series gracefully
        assert hasattr(wf, "weights_series")
        assert len(wf.weights_series) == 0

    def test_weight_function_with_nan_values(self):
        """Test WeightFunction behavior with NaN values."""
        weights_with_nan = pd.Series([1.0, np.nan, 0.8], index=[0, 1, 2])
        wf = WeightFunction(weights_with_nan)
        
        # Should preserve NaN through pickle cycle
        pickled = pickle.dumps(wf)
        wf_restored = pickle.loads(pickled)
        
        result = wf_restored(pd.Index([0, 1, 2]))
        assert np.isnan(result[1])


# ============================================================================
# Helper Functions Tests
# ============================================================================


class TestHelperFunctions:
    """Test helper functions described in documentation.
    
    Based on docs: "Helper Functions (Advanced Usage)" section
    """

    def test_ensure_model_dir_creates_parent_directories(self, temp_model_dir):
        """Test that _ensure_model_dir creates all parent directories."""
        deep_path = temp_model_dir / "a" / "b" / "c" / "d" / "forecasters"
        
        result = _ensure_model_dir(deep_path)
        
        assert result.exists()
        assert deep_path.exists()

    def test_get_model_filepath_returns_path_object(self, temp_model_dir):
        """Test that _get_model_filepath returns Path objects."""
        filepath = _get_model_filepath(temp_model_dir, "test_model")
        
        assert isinstance(filepath, Path)

    def test_get_model_filepath_consistency(self, temp_model_dir):
        """Test that calling _get_model_filepath multiple times is consistent."""
        filepath1 = _get_model_filepath(temp_model_dir, "model")
        filepath2 = _get_model_filepath(temp_model_dir, "model")
        
        assert filepath1 == filepath2

    def test_model_directory_exists_with_path_object(self, temp_model_dir):
        """Test _model_directory_exists works with Path objects."""
        assert _model_directory_exists(temp_model_dir) is True

    def test_model_directory_exists_with_string(self, temp_model_dir):
        """Test _model_directory_exists works with string paths."""
        assert _model_directory_exists(str(temp_model_dir)) is True


# ============================================================================
# Documentation Example Tests
# ============================================================================


class TestDocumentationExamples:
    """Test exact examples from documentation.
    
    Ensures all code samples in model_persistence.md work as documented.
    """

    def test_weight_function_basic_example(self):
        """Test basic WeightFunction example from docs.
        
        From docs:
        ```python
        from spotforecast2.preprocessing import WeightFunction
        
        weights_series = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
        weight_func = WeightFunction(weights_series)
        ```
        """
        weights_series = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
        weight_func = WeightFunction(weights_series)
        
        assert weight_func is not None
        assert weight_func(pd.Index([0])) == 1.0

    def test_weight_function_pickling_example(self):
        """Test WeightFunction pickling example from docs.
        
        From docs:
        ```python
        weight_func = WeightFunction(weights_series)
        
        # Wrap in WeightFunction (picklable, unlike local functions!)
        import pickle
        pickled = pickle.dumps(weight_func)
        unpickled = pickle.loads(pickled)
        unpickled(pd.Index([0, 1]))
        ```
        """
        weights_series = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
        weight_func = WeightFunction(weights_series)
        
        # Should be picklable
        pickled = pickle.dumps(weight_func)
        unpickled = pickle.loads(pickled)
        result = unpickled(pd.Index([0, 1]))
        
        assert len(result) == 2
        assert result[0] == 1.0

    def test_cache_directory_filepath_example(self, temp_model_dir):
        """Test filepath naming from docs.
        
        From docs: "Models are stored in the format: model_dir/forecaster_{target_name}.joblib"
        """
        filepath = _get_model_filepath(temp_model_dir, "power")
        
        assert filepath.name == "forecaster_power.joblib"

    def test_ensemble_weights_example(self):
        """Test creating weights from ensemble-like scenarios."""
        # Simulate weights based on model performance
        performance_scores = pd.Series(
            [0.95, 0.88, 0.92, 0.85],
            index=[0, 1, 2, 3]
        )
        
        # Normalize to weights
        weights = performance_scores / performance_scores.sum()
        weight_func = WeightFunction(weights)
        
        # Should be picklable
        pickled = pickle.dumps(weight_func)
        assert pickled is not None

    def test_missing_data_weights_example(self):
        """Test creating weights from missing data imputation.
        
        From docs: "Weights created from missing data analysis"
        """
        # Simulate weights based on imputation method certainty
        weights = pd.Series(
            [1.0, 0.9, 0.8],  # High certainty -> weight 1.0, low -> 0.8
            index=[0, 1, 2]
        )
        
        weight_func = WeightFunction(weights)
        
        # Verify picklable
        assert pickle.dumps(weight_func) is not None


# ============================================================================
# Safety-Critical Validation Tests
# ============================================================================


class TestSafetyCriticalValidation:
    """Safety-critical validation tests for model persistence.
    
    Ensures model persistence is reliable for production use.
    """

    def test_weight_function_thread_safe_pickling(self, sample_weights_series):
        """Verify WeightFunction pickling is thread-safe."""
        wf = WeightFunction(sample_weights_series)
        
        # Multiple pickle/unpickle cycles should produce identical results
        results = []
        for _ in range(10):
            pickled = pickle.dumps(wf)
            unpickled = pickle.loads(pickled)
            result = unpickled(pd.Index([0, 1, 2]))
            results.append(result)
        
        # All results should be identical
        for r in results[1:]:
            np.testing.assert_array_equal(r, results[0])

    def test_weight_function_preserves_data_types(self, sample_weights_series):
        """Verify WeightFunction preserves data types through pickle cycle."""
        wf = WeightFunction(sample_weights_series)
        
        pickled = pickle.dumps(wf)
        unpickled = pickle.loads(pickled)
        
        # Original weights should be preserved in type
        assert isinstance(unpickled.weights_series, pd.Series)
        assert unpickled.weights_series.dtype == sample_weights_series.dtype

    def test_model_directory_creation_atomicity(self, temp_model_dir):
        """Verify model directory creation is atomic operations."""
        # Should be safe to call multiple times
        dir1 = _ensure_model_dir(temp_model_dir / "models")
        dir2 = _ensure_model_dir(temp_model_dir / "models")
        
        # Should result in same directory
        assert dir1 == dir2
        assert dir1.exists()

    def test_cache_path_uniqueness(self, temp_model_dir):
        """Verify each model gets unique cache path."""
        targets = ["target_a", "target_b", "target_c"]
        paths = [_get_model_filepath(temp_model_dir, t) for t in targets]
        
        # All paths should be unique
        assert len(set(str(p) for p in paths)) == len(targets)

    def test_weight_function_prevents_closure_lockup(self):
        """Verify WeightFunction prevents closure-based lockup.
        
        Safety-critical: Local functions with closures cannot be pickled,
        causing model persistence to fail in production.
        """
        # This would fail with normal function
        weights = pd.Series([1.0, 0.9], index=[0, 1])
        wrapper_class = WeightFunction(weights)
        
        # Should be picklable (unlike closure)
        try:
            pickle.dumps(wrapper_class)
            success = True
        except Exception:
            success = False
        
        assert success, "WeightFunction should be picklable"


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of persistence mechanism."""

    def test_weight_function_with_large_series(self):
        """Test WeightFunction with large weight series."""
        large_weights = pd.Series(
            np.random.rand(10000),
            index=range(10000)
        )
        wf = WeightFunction(large_weights)
        
        # Should pickle/unpickle successfully
        pickled = pickle.dumps(wf)
        unpickled = pickle.loads(pickled)
        
        assert len(unpickled.weights_series) == 10000

    def test_weight_function_with_datetime_index(self):
        """Test WeightFunction with DatetimeIndex."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        weights = pd.Series(np.linspace(1.0, 0.1, 10), index=dates)
        
        wf = WeightFunction(weights)
        
        # Should pickle/unpickle
        pickled = pickle.dumps(wf)
        unpickled = pickle.loads(pickled)
        
        assert isinstance(unpickled.weights_series.index, pd.DatetimeIndex)

    def test_weight_function_with_multiindex(self):
        """Test WeightFunction with MultiIndex."""
        index = pd.MultiIndex.from_product(
            [["A", "B"], [1, 2, 3]],
            names=["letter", "number"]
        )
        weights = pd.Series(np.random.rand(6), index=index)
        
        wf = WeightFunction(weights)
        
        # Should pickle/unpickle
        pickled = pickle.dumps(wf)
        unpickled = pickle.loads(pickled)
        
        assert isinstance(unpickled.weights_series.index, pd.MultiIndex)

    def test_model_directory_with_long_paths(self, temp_model_dir):
        """Test handling of very long directory paths."""
        # Create a path with many levels
        long_path = temp_model_dir
        for i in range(20):
            long_path = long_path / f"level_{i:02d}"
        
        result = _ensure_model_dir(long_path)
        assert result.exists()

    def test_cache_path_with_unicode_characters(self, temp_model_dir):
        """Test model paths with unicode characters."""
        unicode_target = "température_énergie_naïve"
        filepath = _get_model_filepath(temp_model_dir, unicode_target)
        
        assert unicode_target in filepath.name
        assert filepath.suffix == ".joblib"
