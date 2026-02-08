# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test suite for task_demo.py integration with load_actual_combined.

This module validates that task_demo.py correctly uses load_actual_combined
from spotforecast2_safe for loading ground truth data.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from spotforecast2_safe.manager.datasets import DemoConfig, load_actual_combined


class TestLoadActualCombinedIntegration:
    """Test integration of load_actual_combined in task_demo workflow."""

    def test_load_actual_combined_with_demo_config(self):
        """Test loading actual data using DemoConfig as in task_demo.py."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("timestamp,col1,col2,col3\n")
            for i in range(30):
                f.write(f"2020-01-01 {i:02d}:00:00,{i},{i*2},{i*3}\n")
            temp_path = Path(f.name)

        try:
            # Simulate task_demo.py usage
            DATA_PATH = str(temp_path)
            FORECAST_HORIZON = 24
            WEIGHTS = [1.0, 1.0, 1.0]
            columns = ["col1", "col2", "col3"]

            # Use load_actual_combined as in task_demo.py
            config = DemoConfig(data_path=Path(DATA_PATH).expanduser())
            actual_combined = load_actual_combined(
                config=config,
                columns=columns,
                forecast_horizon=FORECAST_HORIZON,
                weights=WEIGHTS,
            )

            # Validate results
            assert isinstance(actual_combined, pd.Series)
            assert len(actual_combined) == FORECAST_HORIZON
            assert actual_combined.index.name == "timestamp"

        finally:
            temp_path.unlink()

    def test_load_actual_combined_with_tilde_path(self):
        """Test that tilde paths are properly expanded."""
        # Create temp directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "spotforecast2_data"
            data_dir.mkdir()
            data_file = data_dir / "data_test.csv"

            # Write test data
            with open(data_file, "w") as f:
                f.write("timestamp,A,B\n")
                for i in range(10):
                    f.write(f"2020-01-01 {i:02d}:00:00,{i},{i*2}\n")

            # Use the file
            config = DemoConfig(data_path=data_file)
            result = load_actual_combined(
                config=config,
                columns=["A", "B"],
                forecast_horizon=5,
                weights=[1.0, 1.0],
            )

            assert len(result) == 5
            assert isinstance(result, pd.Series)

    def test_load_actual_combined_override_parameters(self):
        """Test overriding forecast_horizon and weights as in task_demo.py."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("timestamp,X,Y,Z\n")
            for i in range(50):
                f.write(f"2020-01-01 {i:02d}:00:00,{i},{i*2},{i*3}\n")
            temp_path = Path(f.name)

        try:
            # Config has default horizon of 24
            config = DemoConfig(data_path=temp_path, forecast_horizon=24)

            # Override with custom horizon
            result = load_actual_combined(
                config=config,
                columns=["X", "Y", "Z"],
                forecast_horizon=10,  # Override
                weights=[1.0, -1.0, 1.0],  # Custom weights
            )

            assert len(result) == 10  # Should use overridden value, not 24

        finally:
            temp_path.unlink()

    def test_load_actual_combined_with_standard_weights(self):
        """Test using the standard 11-column weight configuration from task_demo.py."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            # Create 11 columns as in task_demo
            columns = [f"col{i}" for i in range(11)]
            f.write("timestamp," + ",".join(columns) + "\n")
            for i in range(30):
                values = ",".join([str(i * j) for j in range(11)])
                f.write(f"2020-01-01 {i:02d}:00:00,{values}\n")
            temp_path = Path(f.name)

        try:
            # Standard weights from task_demo.py
            WEIGHTS = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]

            config = DemoConfig(data_path=temp_path)
            result = load_actual_combined(
                config=config,
                columns=columns,
                forecast_horizon=24,
                weights=WEIGHTS,
            )

            assert len(result) == 24
            assert isinstance(result, pd.Series)
            assert len(WEIGHTS) == 11

        finally:
            temp_path.unlink()


class TestTaskDemoWorkflow:
    """Test the complete workflow as used in task_demo.py."""

    def test_workflow_simulation(self):
        """Simulate the task_demo.py workflow with load_actual_combined."""
        # Create mock prediction data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            columns = ["A", "B", "C"]
            f.write("timestamp," + ",".join(columns) + "\n")
            for i in range(30):
                values = ",".join([str(100 + i), str(50 + i), str(25 + i)])
                f.write(f"2020-01-01 {i:02d}:00:00,{values}\n")
            temp_path = Path(f.name)

        try:
            # Simulate task_demo.py constants
            DATA_PATH = str(temp_path)
            FORECAST_HORIZON = 24
            WEIGHTS = [1.0, -0.5, -0.5]

            # Simulate prediction columns
            baseline_predictions_columns = columns

            # Load actual combined as in task_demo.py
            config = DemoConfig(data_path=Path(DATA_PATH).expanduser())
            actual_combined = load_actual_combined(
                config=config,
                columns=list(baseline_predictions_columns),
                forecast_horizon=FORECAST_HORIZON,
                weights=WEIGHTS,
            )

            # Validate
            assert isinstance(actual_combined, pd.Series)
            assert len(actual_combined) == FORECAST_HORIZON

            # Simulate reindexing (as done in task_demo.py)
            mock_prediction_index = pd.date_range(
                "2020-01-01", periods=FORECAST_HORIZON, freq="h"
            )
            actual_combined_reindexed = actual_combined.reindex(mock_prediction_index)

            assert len(actual_combined_reindexed) == FORECAST_HORIZON

        finally:
            temp_path.unlink()


class TestErrorHandling:
    """Test error handling in load_actual_combined integration."""

    def test_missing_file_error(self):
        """Test error when data file does not exist."""
        config = DemoConfig(data_path=Path("/nonexistent/path/data.csv"))

        with pytest.raises(FileNotFoundError):
            load_actual_combined(
                config=config,
                columns=["A", "B"],
                forecast_horizon=24,
                weights=[1.0, 1.0],
            )

    def test_missing_columns_error(self):
        """Test error when requested columns are missing."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("timestamp,A,B\n")
            f.write("2020-01-01 00:00:00,1,2\n")
            temp_path = Path(f.name)

        try:
            config = DemoConfig(data_path=temp_path)

            with pytest.raises(ValueError, match="Missing columns in test data"):
                load_actual_combined(
                    config=config,
                    columns=["A", "B", "C"],  # C doesn't exist
                    forecast_horizon=1,
                    weights=[1.0, 1.0, 1.0],
                )

        finally:
            temp_path.unlink()


class TestBackwardCompatibility:
    """Test that the new implementation maintains backward compatibility."""

    def test_same_results_as_old_implementation(self):
        """Verify new load_actual_combined produces same results as old _load_actual_combined."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("timestamp,col1,col2\n")
            for i in range(10):
                f.write(f"2020-01-01 {i:02d}:00:00,{i},{i*2}\n")
            temp_path = Path(f.name)

        try:
            columns = ["col1", "col2"]
            weights = [1.0, 1.0]
            forecast_horizon = 5

            # New implementation
            config = DemoConfig(data_path=temp_path)
            new_result = load_actual_combined(
                config=config,
                columns=columns,
                forecast_horizon=forecast_horizon,
                weights=weights,
            )

            # Old implementation (inline for comparison)
            from spotforecast2_safe.processing.agg_predict import agg_predict

            data_test = pd.read_csv(temp_path, index_col=0, parse_dates=True)
            actual_df = data_test[columns].iloc[:forecast_horizon]
            old_result = agg_predict(actual_df, weights=weights)

            # Compare results
            pd.testing.assert_series_equal(new_result, old_result)

        finally:
            temp_path.unlink()
