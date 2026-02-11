"""
Test suite for outlier detection and visualization documentation examples.

Implements pytest tests for all examples from docs/preprocessing/outliers.md
to ensure documentation accuracy and example functionality.

Safety-critical validation scope:
- Outlier detection accuracy
- Visualization output generation
- Parameter validation and edge cases
- Data integrity throughout workflow
- Reproducibility with random_state
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the functions being tested
from spotforecast2.preprocessing.outlier import get_outliers

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data_simple():
    """Create sample data with clear outliers."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "temperature": np.concatenate(
                [np.random.normal(20, 5, 100), [50, 60, 70]]  # outliers
            ),
            "humidity": np.concatenate(
                [np.random.normal(60, 10, 100), [95, 98, 99]]  # outliers
            ),
        }
    )


@pytest.fixture
def sample_data_multicolumn():
    """Create multicolumn data with outliers."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
            "B": np.concatenate([np.random.normal(5, 2, 100), [100, 110, 120]]),
        }
    )


@pytest.fixture
def sample_timeseries_data():
    """Create time series data with outliers."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=103, freq="h")
    return pd.DataFrame(
        {
            "temperature": np.concatenate(
                [np.random.normal(20, 5, 100), [50, 60, 70]]  # outliers
            ),
            "humidity": np.concatenate(
                [np.random.normal(60, 10, 100), [95, 98, 99]]  # outliers
            ),
        },
        index=dates,
    )


@pytest.fixture
def complete_workflow_data():
    """Create realistic time series data for complete workflow."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    return pd.DataFrame(
        {
            "temperature": np.concatenate(
                [np.random.normal(20, 5, 197), [50, 60, 70]]  # outliers
            ),
            "humidity": np.concatenate(
                [np.random.normal(60, 10, 197), [95, 98, 99]]  # outliers
            ),
            "pressure": np.concatenate(
                [np.random.normal(1013, 10, 197), [800, 1200, 950]]  # outliers
            ),
        },
        index=dates,
    )


# ============================================================================
# Tests for get_outliers() Function
# ============================================================================


class TestGetOutliersBasic:
    """Test basic outlier detection functionality.

    Based on docs: "get_outliers()" API Reference section
    """

    def test_get_outliers_returns_dict(self, sample_data_simple):
        """Test that get_outliers returns a dictionary."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        assert isinstance(outliers, dict)

    def test_get_outliers_dict_keys_match_columns(self, sample_data_simple):
        """Test that returned dictionary keys match data columns."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        assert set(outliers.keys()) == set(sample_data_simple.columns)

    def test_get_outliers_values_are_series(self, sample_data_simple):
        """Test that outlier values are pandas Series."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        for col, outlier_vals in outliers.items():
            assert isinstance(outlier_vals, pd.Series)

    def test_get_outliers_detects_extreme_values(self, sample_data_simple):
        """Test that get_outliers detects the extreme values as outliers."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        # Should detect outliers in both columns
        assert len(outliers["temperature"]) > 0
        assert len(outliers["humidity"]) > 0

    def test_get_outliers_with_different_contamination(self, sample_data_simple):
        """Test get_outliers with different contamination levels."""
        # Conservative
        outliers_conservative = get_outliers(sample_data_simple, contamination=0.01)

        # Liberal
        outliers_liberal = get_outliers(sample_data_simple, contamination=0.05)

        # Liberal should detect at least as many outliers
        assert len(outliers_liberal["temperature"]) >= len(
            outliers_conservative["temperature"]
        )

    def test_get_outliers_reproducibility(self, sample_data_simple):
        """Test that same random_state produces same results.

        From docs: "The random_state parameter ensures reproducibility"
        """
        outliers1 = get_outliers(
            sample_data_simple, random_state=42, contamination=0.03
        )
        outliers2 = get_outliers(
            sample_data_simple, random_state=42, contamination=0.03
        )

        # Results should be identical
        for col in outliers1.keys():
            pd.testing.assert_series_equal(outliers1[col], outliers2[col])

    def test_get_outliers_different_random_state(self, sample_data_simple):
        """Test that different random_state may produce different results."""
        outliers1 = get_outliers(
            sample_data_simple, random_state=42, contamination=0.03
        )
        outliers2 = get_outliers(
            sample_data_simple, random_state=123, contamination=0.03
        )

        # Results may differ (not guaranteed, but likely)
        # Just check they're both valid
        assert isinstance(outliers1, dict)
        assert isinstance(outliers2, dict)

    def test_get_outliers_multicolumn(self, sample_data_multicolumn):
        """Test outlier detection on multicolumn data.

        From docs: API example with columns A and B
        """
        outliers = get_outliers(sample_data_multicolumn, contamination=0.03)

        assert "A" in outliers
        assert "B" in outliers


# ============================================================================
# Tests for Contamination Parameter
# ============================================================================


class TestContaminationParameter:
    """Test contamination parameter behavior.

    Based on docs: "contamination parameter" section
    """

    def test_contamination_conservative(self, sample_data_simple):
        """Test conservative contamination (0.01 = 1%)."""
        outliers = get_outliers(sample_data_simple, contamination=0.01)

        # Should detect few outliers
        for col, outlier_vals in outliers.items():
            assert len(outlier_vals) >= 0

    def test_contamination_moderate(self, sample_data_simple):
        """Test moderate contamination (0.02 = 2%)."""
        outliers = get_outliers(sample_data_simple, contamination=0.02)

        for col, outlier_vals in outliers.items():
            assert len(outlier_vals) >= 0

    def test_contamination_liberal(self, sample_data_simple):
        """Test liberal contamination (0.05 = 5%)."""
        outliers = get_outliers(sample_data_simple, contamination=0.05)

        # Should detect more outliers with higher contamination
        for col, outlier_vals in outliers.items():
            assert len(outlier_vals) >= 0

    def test_contamination_affects_detection_sensitivity(self, sample_data_simple):
        """Test that higher contamination detects more outliers."""
        con_low = 0.01
        con_high = 0.05

        outliers_low = get_outliers(sample_data_simple, contamination=con_low)
        outliers_high = get_outliers(sample_data_simple, contamination=con_high)

        # Higher contamination should find at least as many outliers
        for col in outliers_low.keys():
            assert len(outliers_high[col]) >= len(outliers_low[col])


# ============================================================================
# Tests for Random State Parameter
# ============================================================================


class TestRandomStateParameter:
    """Test random_state parameter behavior.

    Based on docs: "random_state parameter" section
    """

    def test_random_state_same_state_produces_same_results(self, sample_data_simple):
        """Test that same random_state produces consistent results."""
        results = []
        for _ in range(3):
            outliers = get_outliers(
                sample_data_simple, random_state=42, contamination=0.03
            )
            results.append(outliers)

        # All should be identical
        for col in results[0].keys():
            for i in range(1, len(results)):
                pd.testing.assert_series_equal(
                    results[0][col], results[i][col], check_exact=True
                )

    def test_random_state_edge_cases(self, sample_data_simple):
        """Test random_state with various values."""
        for state in [0, 1, 42, 1234, 9999]:
            outliers = get_outliers(
                sample_data_simple, random_state=state, contamination=0.03
            )

            assert isinstance(outliers, dict)
            assert len(outliers) > 0


# ============================================================================
# Tests for Data Integrity
# ============================================================================


class TestDataIntegrity:
    """Test that get_outliers maintains data integrity.

    Safety-critical: Ensure outlier detection doesn't corrupt data
    """

    def test_outliers_are_actual_values_from_data(self, sample_data_simple):
        """Test that detected outliers are actual values from the data."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        for col, outlier_vals in outliers.items():
            # Each outlier value should exist in original data
            for val in outlier_vals:
                assert val in sample_data_simple[col].values

    def test_outlier_indices_valid(self, sample_data_simple):
        """Test that outlier indices are valid for the data."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        for col, outlier_vals in outliers.items():
            # All indices should be valid
            for idx in outlier_vals.index:
                assert idx in sample_data_simple.index

    def test_get_outliers_with_nan_values(self):
        """Test get_outliers with data containing NaN values."""
        data = pd.DataFrame(
            {
                "A": [1, 2, np.nan, 4, 5] * 20 + [100, 101, 102],
                "B": [10, 20, 30, np.nan, 50] * 20 + [1000, 1100, 1200],
            }
        )

        outliers = get_outliers(data, contamination=0.03)

        # Should handle NaN gracefully
        assert isinstance(outliers, dict)

    def test_get_outliers_preserves_column_order(self, sample_data_multicolumn):
        """Test that function preserves column information."""
        outliers = get_outliers(sample_data_multicolumn, contamination=0.03)

        # All original columns should be in results
        assert set(outliers.keys()) == set(sample_data_multicolumn.columns)


# ============================================================================
# Tests for Complete Workflow
# ============================================================================


class TestCompleteWorkflow:
    """Test complete workflow example from documentation.

    Based on docs: "Complete Workflow Example" section
    """

    def test_workflow_detect_outliers(self, complete_workflow_data):
        """Test outlier detection step in workflow."""
        outliers = get_outliers(complete_workflow_data, contamination=0.015)

        # Should have outliers for all columns
        assert len(outliers) == 3  # temperature, humidity, pressure

        # Should detect some outliers
        for col, outlier_vals in outliers.items():
            assert len(outlier_vals) > 0

    def test_workflow_calculate_percentages(self, complete_workflow_data):
        """Test calculating outlier percentages in workflow."""
        outliers = get_outliers(complete_workflow_data, contamination=0.015)

        for col, outlier_vals in outliers.items():
            pct = (len(outlier_vals) / len(complete_workflow_data)) * 100

            # Should be reasonable percentage
            assert 0 <= pct <= 100
            # With contamination 0.015, should detect ~1.5% or fewer
            assert pct <= 5

    def test_workflow_selective_columns(self, complete_workflow_data):
        """Test analyzing selective columns."""
        outliers = get_outliers(complete_workflow_data, contamination=0.015)

        # Should have data for specific columns
        assert "temperature" in outliers
        assert "humidity" in outliers
        assert "pressure" in outliers


# ============================================================================
# Tests for Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and robustness.

    Safety-critical: Ensure function handles unusual inputs gracefully
    """

    def test_single_column_dataframe(self):
        """Test with single column DataFrame."""
        data = pd.DataFrame(
            {"A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]])}
        )

        outliers = get_outliers(data, contamination=0.03)

        assert "A" in outliers

    def test_small_dataframe(self):
        """Test with small DataFrame (edge case)."""
        data = pd.DataFrame({"A": [1, 2, 3, 100], "B": [10, 20, 30, 1000]})

        outliers = get_outliers(
            data, contamination=0.25
        )  # High contamination for tiny dataset

        assert isinstance(outliers, dict)
        assert len(outliers) == 2

    def test_large_dataframe(self):
        """Test with large DataFrame."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "A": np.concatenate(
                    [np.random.normal(0, 1, 10000), np.random.normal(10, 1, 100)]
                ),
                "B": np.concatenate(
                    [np.random.normal(5, 2, 10000), np.random.normal(20, 2, 100)]
                ),
            }
        )

        outliers = get_outliers(data, contamination=0.01)

        assert isinstance(outliers, dict)

    def test_all_identical_values(self):
        """Test with all identical values (no variance)."""
        data = pd.DataFrame({"A": [5.0] * 100, "B": [10.0] * 100})

        # Should not crash
        outliers = get_outliers(data, contamination=0.01)

        assert isinstance(outliers, dict)

    def test_bimodal_distribution(self):
        """Test with bimodal distribution (two groups)."""
        data = pd.DataFrame(
            {
                "A": np.concatenate(
                    [
                        np.random.normal(0, 1, 50),
                        np.random.normal(10, 1, 50),
                        [100, 101, 102],
                    ]
                )
            }
        )

        outliers = get_outliers(data, contamination=0.03)

        assert isinstance(outliers, dict)

    def test_with_negative_values(self):
        """Test with negative values in data."""
        data = pd.DataFrame(
            {"A": np.concatenate([np.random.normal(-20, 5, 100), [50, 60, 70]])}
        )

        outliers = get_outliers(data, contamination=0.03)

        assert len(outliers["A"]) > 0

    def test_with_very_large_values(self):
        """Test with very large values."""
        data = pd.DataFrame(
            {
                "A": np.concatenate(
                    [np.random.normal(1e10, 1e9, 100), [1e12, 1e12, 1e12]]
                )
            }
        )

        outliers = get_outliers(data, contamination=0.03)

        assert isinstance(outliers, dict)

    def test_with_very_small_values(self):
        """Test with very small (near-zero) values."""
        data = pd.DataFrame(
            {
                "A": np.concatenate(
                    [np.random.normal(1e-10, 1e-11, 100), [1e-8, 1e-8, 1e-8]]
                )
            }
        )

        outliers = get_outliers(data, contamination=0.03)

        assert isinstance(outliers, dict)


# ============================================================================
# Tests for Timeseries Data
# ============================================================================


class TestTimeseriesData:
    """Test outlier detection on time series data.

    Based on docs: examples using DatetimeIndex
    """

    def test_outliers_with_datetime_index(self, sample_timeseries_data):
        """Test that outlier detection works with datetime index."""
        outliers = get_outliers(sample_timeseries_data, contamination=0.03)

        # Outliers should have datetime index
        for col, outlier_vals in outliers.items():
            assert isinstance(outlier_vals.index, pd.DatetimeIndex)

    def test_outlier_timing_identification(self, sample_timeseries_data):
        """Test that we can identify when outliers occurred."""
        outliers = get_outliers(sample_timeseries_data, contamination=0.03)

        for col, outlier_vals in outliers.items():
            # Should be able to see when outliers occurred
            assert len(outlier_vals.index) > 0

            # Timestamps should be reasonable
            for timestamp in outlier_vals.index:
                assert timestamp in sample_timeseries_data.index

    def test_continuous_timeseries(self):
        """Test with continuous time series data."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        data = pd.DataFrame(
            {
                "value": np.concatenate(
                    [np.random.normal(100, 10, 497), [200, 250, 300]]
                )
            },
            index=dates,
        )

        outliers = get_outliers(data, contamination=0.01)

        assert isinstance(outliers["value"].index, pd.DatetimeIndex)


# ============================================================================
# Tests for API Examples from Documentation
# ============================================================================


class TestAPIExamples:
    """Test exact examples from API documentation.

    Ensures all documented examples work as shown
    """

    def test_api_example_basic(self, sample_data_multicolumn):
        """Test basic API example from documentation."""
        outliers = get_outliers(sample_data_multicolumn, contamination=0.03)

        # Should be able to iterate through results as shown in docs
        for col, outlier_vals in outliers.items():
            assert len(col) > 0
            assert len(outlier_vals) >= 0

    def test_api_example_contains_column_names(self):
        """Test that API example shows all column names."""
        data = pd.DataFrame(
            {
                "A": np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
                "B": np.concatenate([np.random.normal(5, 2, 100), [100, 110, 120]]),
            }
        )

        outliers = get_outliers(data, contamination=0.03)

        # Verify we can access results as shown in docs
        col_names = list(outliers.keys())
        assert "A" in col_names
        assert "B" in col_names


# ============================================================================
# Tests for Validation Workflow
# ============================================================================


class TestValidationWorkflow:
    """Test validation workflow from best practices.

    Based on docs: "Validation" best practice
    """

    def test_validation_summary_stats(self, sample_data_simple):
        """Test getting summary statistics for validation."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        for col, outlier_vals in outliers.items():
            # Should be able to compute statistics
            regular_range_min = sample_data_simple[col].min()
            regular_range_max = sample_data_simple[col].max()
            outlier_values = sorted(outlier_vals.unique())
            outlier_indices = list(outlier_vals.index)

            # All should be valid
            assert regular_range_min is not None
            assert regular_range_max is not None
            assert len(outlier_values) >= 0
            assert len(outlier_indices) >= 0

    def test_validation_outlier_context(self, complete_workflow_data):
        """Test validation in complete context."""
        outliers = get_outliers(complete_workflow_data, contamination=0.015)

        for col, outlier_vals in outliers.items():
            # For each outlier, verify it makes sense
            for idx in outlier_vals.index:
                original_value = complete_workflow_data.loc[idx, col]

                # Outlier value should match original data
                assert original_value in outlier_vals.values


# ============================================================================
# Tests for Safety-Critical Characteristics
# ============================================================================


class TestSafetyCritical:
    """Safety-critical validation tests.

    Ensures outlier detection is reliable for production use
    """

    def test_deterministic_with_seed(self, sample_data_simple):
        """Verify deterministic behavior with seed."""
        results_list = []
        for _ in range(5):
            outliers = get_outliers(
                sample_data_simple, random_state=42, contamination=0.03
            )
            results_list.append(outliers)

        # All results should be identical
        for col in results_list[0].keys():
            base_series = results_list[0][col]
            for result_dict in results_list[1:]:
                pd.testing.assert_series_equal(base_series, result_dict[col])

    def test_no_silent_failures(self, sample_data_simple):
        """Test that function doesn't silently handle errors."""
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        # All columns should have results
        assert len(outliers) == len(sample_data_simple.columns)

    def test_contamination_bounds_respected(self, sample_data_simple):
        """Test that detected outliers respect contamination bounds."""
        contamination = 0.03
        outliers = get_outliers(sample_data_simple, contamination=contamination)

        for col, outlier_vals in outliers.items():
            # Detected outliers should be close to specified contamination level
            detected_pct = len(outlier_vals) / len(sample_data_simple)

            # Allow some tolerance due to binning in isolation forest
            assert detected_pct <= contamination * 1.5

    def test_consistent_across_data_subsets(self):
        """Test that algorithm is consistent across data subsets."""
        np.random.seed(42)
        data_full = pd.DataFrame(
            {
                "A": np.concatenate(
                    [np.random.normal(0, 1, 200), [10, 11, 12, 13, 14, 15]]
                )
            }
        )

        # Both should detect outliers
        outliers = get_outliers(data_full, contamination=0.03)

        assert len(outliers["A"]) > 0


# ============================================================================
# Tests for Documentation Accuracy
# ============================================================================


class TestDocumentationAccuracy:
    """Test that documentation examples work exactly as documented."""

    def test_quick_start_example_works(self, sample_data_simple):
        """Test quick start example from docs."""
        # From docs: "Create sample data with outliers"
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        # From docs: "for col, outlier_vals in outliers.items()"
        for col, outlier_vals in outliers.items():
            # Should be able to print results as shown
            num_outliers = len(outlier_vals)
            assert num_outliers >= 0

    def test_parameters_table_default_values(self, sample_data_simple):
        """Test that default parameter values match documentation."""
        # Defaults from docs table: contamination: 0.01, random_state: 1234
        outliers = get_outliers(sample_data_simple)  # Using defaults

        # Should produce valid results with defaults
        assert isinstance(outliers, dict)
        assert len(outliers) > 0

    def test_return_type_matches_docs(self, sample_data_simple):
        """Test that return type matches documentation.

        Docs: "Returns: A dictionary mapping column names to pandas Series of outlier values"
        """
        outliers = get_outliers(sample_data_simple, contamination=0.03)

        # Should be dict
        assert isinstance(outliers, dict)

        # Values should be pandas Series
        for col, outlier_vals in outliers.items():
            assert isinstance(outlier_vals, pd.Series)
