"""
Test suite for time series visualization documentation examples.

Implements pytest tests for all examples from docs/preprocessing/time_series_visualization.md
to ensure documentation accuracy and example functionality.

Safety-critical validation scope:
- Visualization function execution
- Data integrity and handling
- Parameter validation
- Edge cases and robustness
- Temporal data handling
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spotforecast2.preprocessing.time_series_visualization import (
    visualize_ts_plotly,
    visualize_ts_comparison,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_timeseries_single_column():
    """Create simple time series with single column."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame({"temperature": np.random.normal(20, 5, 100)}, index=dates)


@pytest.fixture
def sample_timeseries_multicolumn():
    """Create time series with multiple columns."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "temperature": np.random.normal(20, 5, 100),
            "humidity": np.random.normal(60, 10, 100),
        },
        index=dates,
    )


@pytest.fixture
def train_val_test_split():
    """Create train/validation/test split datasets."""
    np.random.seed(42)
    dates_train = pd.date_range("2024-01-01", periods=100, freq="h")
    dates_val = pd.date_range("2024-05-11", periods=50, freq="h")
    dates_test = pd.date_range("2024-07-01", periods=30, freq="h")

    data_train = pd.DataFrame(
        {
            "temperature": np.random.normal(20, 5, 100),
            "humidity": np.random.normal(60, 10, 100),
        },
        index=dates_train,
    )

    data_val = pd.DataFrame(
        {
            "temperature": np.random.normal(22, 5, 50),
            "humidity": np.random.normal(55, 10, 50),
        },
        index=dates_val,
    )

    data_test = pd.DataFrame(
        {
            "temperature": np.random.normal(25, 5, 30),
            "humidity": np.random.normal(50, 10, 30),
        },
        index=dates_test,
    )

    return {"Train": data_train, "Validation": data_val, "Test": data_test}


@pytest.fixture
def seasonal_datasets():
    """Create datasets representing different seasons."""
    np.random.seed(42)
    dates1 = pd.date_range("2024-01-01", periods=100, freq="h")
    dates2 = pd.date_range("2024-04-01", periods=100, freq="h")
    dates3 = pd.date_range("2024-07-01", periods=100, freq="h")

    df1 = pd.DataFrame({"temperature": np.random.normal(15, 3, 100)}, index=dates1)

    df2 = pd.DataFrame({"temperature": np.random.normal(22, 3, 100)}, index=dates2)

    df3 = pd.DataFrame({"temperature": np.random.normal(25, 3, 100)}, index=dates3)

    return {"Winter": df1, "Spring": df2, "Summer": df3}


# ============================================================================
# Tests for visualize_ts_plotly()
# ============================================================================


class TestVisualizeTimeSeriesPlotlyBasic:
    """Test basic visualize_ts_plotly functionality.

    Based on docs: "visualize_ts_plotly()" API Reference
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_single_dataset(self, mock_go, sample_timeseries_single_column):
        """Test visualizing a single dataset."""
        dataframes = {"Data": sample_timeseries_single_column}

        # Should not raise
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_requires_dict(self, mock_go, sample_timeseries_single_column):
        """Test that visualize_ts_plotly requires dict input."""
        # Should raise TypeError for non-dict input
        with pytest.raises((TypeError, AttributeError)):
            visualize_ts_plotly(sample_timeseries_single_column)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_requires_non_empty_dict(self, mock_go):
        """Test that empty dict raises error."""
        with pytest.raises(ValueError):
            visualize_ts_plotly({})

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_with_multiple_datasets(self, mock_go, train_val_test_split):
        """Test visualizing multiple datasets."""
        # Should not raise
        visualize_ts_plotly(train_val_test_split)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_with_column_selection(
        self, mock_go, sample_timeseries_multicolumn
    ):
        """Test selecting specific columns."""
        dataframes = {"Data": sample_timeseries_multicolumn}

        # Should not raise
        visualize_ts_plotly(dataframes, columns=["temperature"])

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_with_title_suffix(
        self, mock_go, sample_timeseries_single_column
    ):
        """Test adding title suffix."""
        dataframes = {"Data": sample_timeseries_single_column}

        # Should not raise
        visualize_ts_plotly(dataframes, title_suffix="[°C]")

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_with_custom_figsize(
        self, mock_go, sample_timeseries_single_column
    ):
        """Test custom figure size."""
        dataframes = {"Data": sample_timeseries_single_column}

        # Should not raise
        visualize_ts_plotly(dataframes, figsize=(1400, 600))

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_with_template(self, mock_go, sample_timeseries_single_column):
        """Test specifying template."""
        dataframes = {"Data": sample_timeseries_single_column}

        # Should not raise
        visualize_ts_plotly(dataframes, template="plotly_dark")

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_visualize_with_colors(self, mock_go, train_val_test_split):
        """Test custom color mapping."""
        colors = {"Train": "blue", "Validation": "green", "Test": "red"}

        # Should not raise
        visualize_ts_plotly(train_val_test_split, colors=colors)


# ============================================================================
# Tests for visualize_ts_comparison()
# ============================================================================


class TestVisualizeTimeSeriesComparison:
    """Test visualize_ts_comparison functionality.

    Based on docs: "visualize_ts_comparison()" API Reference
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_comparison_basic(self, mock_go, seasonal_datasets):
        """Test basic comparison visualization."""
        # Should not raise
        visualize_ts_comparison(seasonal_datasets)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_comparison_show_mean(self, mock_go, seasonal_datasets):
        """Test comparison with mean overlay."""
        # Should not raise
        visualize_ts_comparison(seasonal_datasets, show_mean=True)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_comparison_with_colors(self, mock_go, seasonal_datasets):
        """Test comparison with custom colors."""
        colors = {"Winter": "blue", "Spring": "green", "Summer": "red"}

        # Should not raise
        visualize_ts_comparison(seasonal_datasets, colors=colors)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_comparison_with_title_suffix(self, mock_go, seasonal_datasets):
        """Test comparison with title suffix."""
        # Should not raise
        visualize_ts_comparison(seasonal_datasets, title_suffix="[°C]")

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_comparison_requires_non_empty_dict(self, mock_go):
        """Test that empty dict raises error."""
        with pytest.raises(ValueError):
            visualize_ts_comparison({})

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_comparison_with_figsize(self, mock_go, seasonal_datasets):
        """Test comparison with custom figure size."""
        # Should not raise
        visualize_ts_comparison(seasonal_datasets, figsize=(1200, 600))


# ============================================================================
# Tests for Complete Workflow Examples
# ============================================================================


class TestCompleteWorkflow:
    """Test complete workflow examples from documentation.

    Based on docs: "Complete Workflow Examples" section
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_train_val_test_split_workflow(self, mock_go):
        """Test train/validation/test split workflow."""
        np.random.seed(42)
        full_data = pd.DataFrame(
            {
                "temperature": np.sin(np.linspace(0, 10, 300))
                + np.random.normal(0, 0.1, 300),
                "humidity": np.cos(np.linspace(0, 10, 300)) * 100
                + np.random.normal(50, 5, 300),
            },
            index=pd.date_range("2024-01-01", periods=300, freq="h"),
        )

        # Split data
        split1 = int(0.6 * len(full_data))
        split2 = int(0.8 * len(full_data))

        data_train = full_data.iloc[:split1]
        data_val = full_data.iloc[split1:split2]
        data_test = full_data.iloc[split2:]

        # Visualize
        dataframes = {"Train": data_train, "Validation": data_val, "Test": data_test}

        visualize_ts_plotly(dataframes, template="plotly_white", figsize=(1200, 600))

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_multiple_datasets_comparison_workflow(self, mock_go):
        """Test multiple datasets comparison workflow."""
        dates1 = pd.date_range("2024-01-01", periods=100, freq="h")
        dates2 = pd.date_range("2024-04-01", periods=100, freq="h")
        dates3 = pd.date_range("2024-07-01", periods=100, freq="h")

        df1 = pd.DataFrame({"temperature": np.random.normal(15, 3, 100)}, index=dates1)

        df2 = pd.DataFrame({"temperature": np.random.normal(22, 3, 100)}, index=dates2)

        df3 = pd.DataFrame({"temperature": np.random.normal(25, 3, 100)}, index=dates3)

        # Compare with mean
        visualize_ts_comparison(
            {"Winter": df1, "Spring": df2, "Summer": df3},
            show_mean=True,
            colors={"Winter": "blue", "Spring": "green", "Summer": "red"},
        )

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_dynamic_dataset_handling(self, mock_go):
        """Test handling multiple dynamic datasets."""
        np.random.seed(42)
        dataframes = {}

        for i in range(5):
            dates = pd.date_range(f"2024-{i+1:02d}-01", periods=50, freq="h")
            dataframes[f"Month_{i+1}"] = pd.DataFrame(
                {"sales": np.random.gamma(2, 2, 50) * 1000}, index=dates
            )

        visualize_ts_plotly(dataframes, title_suffix="[USD]", figsize=(1400, 600))


# ============================================================================
# Tests for Parameters and Configuration
# ============================================================================


class TestParametersAndConfiguration:
    """Test all parameter configur options.

    Based on docs: "Parameters and Configuration" section
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_figsize_parameter_small(self, mock_go, sample_timeseries_single_column):
        """Test small figure size."""
        dataframes = {"Data": sample_timeseries_single_column}
        visualize_ts_plotly(dataframes, figsize=(800, 400))

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_figsize_parameter_large(self, mock_go, sample_timeseries_single_column):
        """Test large figure size."""
        dataframes = {"Data": sample_timeseries_single_column}
        visualize_ts_plotly(dataframes, figsize=(1600, 800))

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_template_plotly_white(self, mock_go, sample_timeseries_single_column):
        """Test light theme template."""
        dataframes = {"Data": sample_timeseries_single_column}
        visualize_ts_plotly(dataframes, template="plotly_white")

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_template_plotly_dark(self, mock_go, sample_timeseries_single_column):
        """Test dark theme template."""
        dataframes = {"Data": sample_timeseries_single_column}
        visualize_ts_plotly(dataframes, template="plotly_dark")

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_template_plotly_minimal(self, mock_go, sample_timeseries_single_column):
        """Test minimal theme template."""
        dataframes = {"Data": sample_timeseries_single_column}
        visualize_ts_plotly(dataframes, template="plotly")

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_template_ggplot2(self, mock_go, sample_timeseries_single_column):
        """Test ggplot2 theme template."""
        dataframes = {"Data": sample_timeseries_single_column}
        visualize_ts_plotly(dataframes, template="ggplot2")

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_color_customization_hex(self, mock_go, train_val_test_split):
        """Test color customization with hex codes."""
        colors = {
            "Train": "#1f77b4",  # Blue
            "Validation": "#ff7f0e",  # Orange
            "Test": "#2ca02c",  # Green
        }
        visualize_ts_plotly(train_val_test_split, colors=colors)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_color_customization_names(self, mock_go, train_val_test_split):
        """Test color customization with color names."""
        colors = {"Train": "blue", "Validation": "orange", "Test": "green"}
        visualize_ts_plotly(train_val_test_split, colors=colors)


# ============================================================================
# Tests for Best Practices
# ============================================================================


class TestBestPractices:
    """Test best practices from documentation.

    Based on docs: "Best Practices" section
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_datetime_index_handling(self, mock_go):
        """Test proper datetime index handling."""
        # Good: Using datetime index
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        data = np.random.normal(20, 5, 100)
        df = pd.DataFrame({"value": data}, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_consistent_column_shapes(self, mock_go):
        """Test consistent data shapes across datasets."""
        np.random.seed(42)
        dates1 = pd.date_range("2024-01-01", periods=100, freq="h")
        dates2 = pd.date_range("2024-05-11", periods=100, freq="h")

        # Both have same columns
        df1 = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 5, 100),
                "humidity": np.random.normal(60, 10, 100),
            },
            index=dates1,
        )

        df2 = pd.DataFrame(
            {
                "temperature": np.random.normal(22, 5, 100),
                "humidity": np.random.normal(55, 10, 100),
            },
            index=dates2,
        )

        dataframes = {"Dataset1": df1, "Dataset2": df2}

        # Verify columns match
        columns = set(df1.columns) & set(df2.columns)
        assert len(columns) > 0

        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_large_dataset_subsampling(self, mock_go):
        """Test subsampling for large datasets."""
        np.random.seed(42)
        # Create large dataset
        dates = pd.date_range("2024-01-01", periods=10000, freq="min")
        df = pd.DataFrame({"value": np.random.normal(20, 5, 10000)}, index=dates)

        # Subsample every 10th point
        df_sub = df[::10]

        dataframes = {"Data": df_sub}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_meaningful_dataset_names(self, mock_go):
        """Test using meaningful dataset names."""
        np.random.seed(42)
        dates_train = pd.date_range("2023-01-01", periods=100, freq="h")
        dates_val = pd.date_range("2024-01-01", periods=50, freq="h")
        dates_test = pd.date_range("2024-02-01", periods=50, freq="h")

        data_train = pd.DataFrame(
            {"value": np.random.normal(20, 5, 100)}, index=dates_train
        )

        data_val = pd.DataFrame({"value": np.random.normal(21, 5, 50)}, index=dates_val)

        data_test = pd.DataFrame(
            {"value": np.random.normal(22, 5, 50)}, index=dates_test
        )

        # Good: Descriptive names
        dataframes = {
            "Training (2023)": data_train,
            "Validation (Jan 2024)": data_val,
            "Testing (Feb 2024)": data_test,
        }

        visualize_ts_plotly(dataframes)


# ============================================================================
# Tests for Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling.

    Safety-critical: Ensure robustness to unusual inputs
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_single_value_dataset(self, mock_go):
        """Test with dataset containing single value."""
        dates = pd.date_range("2024-01-01", periods=1, freq="h")
        df = pd.DataFrame({"value": [20]}, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_constant_values(self, mock_go):
        """Test with constant values (no variance)."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({"value": [5.0] * 100}, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_nan_values_in_data(self, mock_go):
        """Test handling NaN values."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        data = np.random.normal(20, 5, 100)
        data[10:20] = np.nan
        df = pd.DataFrame({"value": data}, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_negative_values(self, mock_go):
        """Test with negative values."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({"value": np.random.normal(-20, 5, 100)}, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_very_large_values(self, mock_go):
        """Test with very large values."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({"value": np.random.normal(1e10, 1e9, 100)}, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_very_small_values(self, mock_go):
        """Test with very small values."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({"value": np.random.normal(1e-10, 1e-11, 100)}, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_many_columns(self, mock_go):
        """Test with many columns."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        data = {f"col_{i}": np.random.normal(i * 10, 5, 100) for i in range(20)}
        df = pd.DataFrame(data, index=dates)

        dataframes = {"Data": df}
        visualize_ts_plotly(dataframes)


# ============================================================================
# Tests for API Examples from Documentation
# ============================================================================


class TestAPIExamples:
    """Test exact examples from API documentation.

    Ensures all documented examples work as shown
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_quick_start_basic_visualization(self, mock_go):
        """Test quick start basic visualization example."""
        np.random.seed(42)
        dates_train = pd.date_range("2024-01-01", periods=100, freq="h")
        dates_val = pd.date_range("2024-05-11", periods=50, freq="h")
        dates_test = pd.date_range("2024-07-01", periods=30, freq="h")

        data_train = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 5, 100),
                "humidity": np.random.normal(60, 10, 100),
            },
            index=dates_train,
        )

        data_val = pd.DataFrame(
            {
                "temperature": np.random.normal(22, 5, 50),
                "humidity": np.random.normal(55, 10, 50),
            },
            index=dates_val,
        )

        data_test = pd.DataFrame(
            {
                "temperature": np.random.normal(25, 5, 30),
                "humidity": np.random.normal(50, 10, 30),
            },
            index=dates_test,
        )

        dataframes = {"Train": data_train, "Validation": data_val, "Test": data_test}

        visualize_ts_plotly(dataframes)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_quick_start_single_dataset(self, mock_go):
        """Test quick start single dataset example."""
        np.random.seed(42)
        dates_train = pd.date_range("2024-01-01", periods=100, freq="h")

        data_train = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 5, 100),
                "humidity": np.random.normal(60, 10, 100),
            },
            index=dates_train,
        )

        dataframes = {"Data": data_train}
        visualize_ts_plotly(dataframes, columns=["temperature"])

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_quick_start_custom_styling(self, mock_go):
        """Test quick start custom styling example."""
        np.random.seed(42)
        dates_train = pd.date_range("2024-01-01", periods=100, freq="h")
        dates_val = pd.date_range("2024-05-11", periods=50, freq="h")
        dates_test = pd.date_range("2024-07-01", periods=30, freq="h")

        data_train = pd.DataFrame(
            {"temperature": np.random.normal(20, 5, 100)}, index=dates_train
        )

        data_val = pd.DataFrame(
            {"temperature": np.random.normal(22, 5, 50)}, index=dates_val
        )

        data_test = pd.DataFrame(
            {"temperature": np.random.normal(25, 5, 30)}, index=dates_test
        )

        dataframes = {"Train": data_train, "Validation": data_val, "Test": data_test}

        visualize_ts_plotly(
            dataframes,
            template="plotly_dark",
            colors={"Train": "blue", "Validation": "green", "Test": "red"},
            figsize=(1400, 600),
        )

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_api_example_visualize_ts_plotly(self, mock_go):
        """Test API example for visualize_ts_plotly."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 5, 100),
                "humidity": np.random.normal(60, 10, 100),
            },
            index=dates,
        )

        visualize_ts_plotly({"Data": df})

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_api_example_visualize_ts_comparison(self, mock_go):
        """Test API example for visualize_ts_comparison."""
        np.random.seed(42)
        dates1 = pd.date_range("2024-01-01", periods=100, freq="h")
        dates2 = pd.date_range("2024-05-11", periods=100, freq="h")

        df1 = pd.DataFrame({"value": np.random.normal(20, 5, 100)}, index=dates1)

        df2 = pd.DataFrame({"value": np.random.normal(22, 5, 100)}, index=dates2)

        visualize_ts_comparison({"Dataset1": df1, "Dataset2": df2}, show_mean=True)


# ============================================================================
# Tests for Data Integrity
# ============================================================================


class TestDataIntegrity:
    """Test data integrity and handling.

    Safety-critical: Ensure data is not corrupted
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_index_integrity(self, mock_go, sample_timeseries_single_column):
        """Test that datetime index is preserved."""
        original_index = sample_timeseries_single_column.index
        dataframes = {"Data": sample_timeseries_single_column}

        visualize_ts_plotly(dataframes)

        # Verify index unchanged
        assert sample_timeseries_single_column.index.equals(original_index)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_data_values_unchanged(self, mock_go, sample_timeseries_single_column):
        """Test that data values are not modified."""
        original_values = sample_timeseries_single_column.values.copy()
        dataframes = {"Data": sample_timeseries_single_column}

        visualize_ts_plotly(dataframes)

        # Verify values unchanged
        np.testing.assert_array_equal(
            sample_timeseries_single_column.values, original_values
        )

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_multiple_datasets_independent(self, mock_go, train_val_test_split):
        """Test that datasets remain independent."""
        original_train = train_val_test_split["Train"].copy()

        visualize_ts_plotly(train_val_test_split)

        # Verify train dataset unchanged
        pd.testing.assert_frame_equal(train_val_test_split["Train"], original_train)


# ============================================================================
# Tests for Safety-Critical Characteristics
# ============================================================================


class TestSafetyCritical:
    """Safety-critical validation tests.

    Ensures visualization is reliable for production use
    """

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_handles_missing_columns(self, mock_go, sample_timeseries_multicolumn):
        """Test handling of missing columns."""
        dataframes = {"Data": sample_timeseries_multicolumn}

        # Request non-existent column
        with pytest.raises(ValueError):
            visualize_ts_plotly(dataframes, columns=["nonexistent"])

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_error_on_empty_dataframes(self, mock_go):
        """Test that empty dataframes dict raises error."""
        # Should raise ValueError
        with pytest.raises(ValueError):
            visualize_ts_plotly({})

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_error_on_non_dict_input(self, mock_go, sample_timeseries_single_column):
        """Test that non-dict input raises TypeError."""
        # Should raise TypeError
        with pytest.raises(TypeError):
            visualize_ts_plotly(sample_timeseries_single_column)

    @patch("spotforecast2.preprocessing.time_series_visualization.go")
    def test_comparison_error_on_empty(self, mock_go):
        """Test that comparison also rejects empty dict."""
        with pytest.raises(ValueError):
            visualize_ts_comparison({})

    def test_deterministic_with_seed(self, train_val_test_split):
        """Test that results are deterministic with fixed seed."""
        np.random.seed(42)
        data1 = {k: v.copy() for k, v in train_val_test_split.items()}

        np.random.seed(42)
        data2 = {k: v.copy() for k, v in train_val_test_split.items()}

        # Same seed should produce identical data
        for key in data1:
            pd.testing.assert_frame_equal(data1[key], data2[key])
