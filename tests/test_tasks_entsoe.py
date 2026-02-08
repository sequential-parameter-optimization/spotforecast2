# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from spotforecast2.tasks.task_entsoe import (
    main,
    Period,
    ExogBuilder,
    RepeatingBasisFunction,
    ForecasterRecursiveLGBM,
    ForecasterRecursiveXGB
)

class TestTaskEntsoe(unittest.TestCase):
    """Tests for the task_entsoe script."""

    def test_rbf_transformer(self):
        """Test RepeatingBasisFunction simplified implementation."""
        rbf = RepeatingBasisFunction(n_periods=12, column="hour", input_range=(1, 24))
        df = pd.DataFrame({"hour": range(1, 25)})
        
        # Test transform
        out = rbf.transform(df)
        self.assertEqual(out.shape, (24, 12))
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 1))

    def test_exog_builder(self):
        """Test ExogBuilder creates expected columns."""
        periods = [Period(name="daily", n_periods=12, column="hour", input_range=(1, 24))]
        builder = ExogBuilder(periods=periods)
        
        start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2026-01-02 00:00", tz="UTC")
        
        X = builder.build(start, end)
        
        expected_cols = ["is_weekend"]
        # Plus 12 RBF columns: daily_0 ... daily_11
        for i in range(12):
             expected_cols.append(f"daily_{i}")
        for col in expected_cols:
            self.assertIn(col, X.columns)
        
        self.assertEqual(X.shape[0], 25) # 24 hours + 1 endpoint? freq="h" includes end?
        # pd.date_range includes end by default if matches freq. 00:00 to 00:00 next day is 25 points.

    @patch("spotforecast2.tasks.task_entsoe.download_new_data")
    def test_cli_download(self, mock_download):
        """Test download subcommand."""
        test_args = ["task_entsoe.py", "download", "--api-key", "test_key", "202601010000"]
        with patch.object(sys, "argv", test_args):
            main()
            
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        self.assertEqual(call_args.kwargs["api_key"], "test_key")
        self.assertEqual(call_args.kwargs["start"], "202601010000")

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_cli_train(self, mock_handle_training):
        """Test train subcommand."""
        test_args = ["task_entsoe.py", "train", "lgbm", "--force"]
        with patch.object(sys, "argv", test_args):
            main()
            
        mock_handle_training.assert_called_once()
        self.assertEqual(mock_handle_training.call_args.kwargs["model_type"], "lgbm")
        self.assertTrue(mock_handle_training.call_args.kwargs["force"])
        # Check if model class passed is correct
        model_class = mock_handle_training.call_args.kwargs["model_class"]
        self.assertEqual(model_class, ForecasterRecursiveLGBM)

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    @patch("spotforecast2.tasks.task_entsoe.make_plot")
    def test_cli_predict(self, mock_make_plot, mock_get_pred):
        """Test predict subcommand."""
        mock_get_pred.return_value = {"some": "data"}
        
        test_args = ["task_entsoe.py", "predict", "--plot"]
        with patch.object(sys, "argv", test_args):
            main()
            
        mock_get_pred.assert_called_once()
        mock_make_plot.assert_called_once_with({"some": "data"})

if __name__ == "__main__":
    unittest.main()
