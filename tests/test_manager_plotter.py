# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import plotly.graph_objects as go
from spotforecast2.manager.plotter import make_plot, PredictionFigure

class TestPlotter(unittest.TestCase):
    """Tests for the plotter manager."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Setup mock prediction package
        dates_train = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
        dates_future = pd.date_range("2026-01-02", periods=24, freq="h", tz="UTC")
        
        self.mock_pkg = {
            "train_actual": pd.Series(range(24), index=dates_train),
            "future_actual": pd.Series(range(24, 48), index=dates_future),
            "train_pred": pd.Series(range(24), index=dates_train),
            "future_pred": pd.Series(range(24, 48), index=dates_future),
            "future_forecast": pd.Series(range(24, 48), index=dates_future),
            "metrics_train": {"mae": 1.0, "mape": 0.1},
            "metrics_future": {"mae": 2.0, "mape": 0.2},
            "metrics_future_one_day": {"mae": 1.5, "mape": 0.15},
            "metrics_forecast": {"mae": 3.0, "mape": 0.3},
            "metrics_forecast_one_day": {"mae": 2.5, "mape": 0.25},
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_prediction_figure_creation(self):
        """Test that PredictionFigure generates a valid plotly figure."""
        pf = PredictionFigure(self.mock_pkg)
        fig = pf.make_plot()
        
        self.assertIsInstance(fig, go.Figure)
        # Check number of traces (train_actual + future_actual concat, train_pred + future_pred concat, forecast, last week shift)
        # Actually concat is done internally, so traces: Actual, Prediction, Forecast, Last Week = 4 traces
        self.assertEqual(len(fig.data), 4)
        
        # Verify annotations
        self.assertEqual(len(fig.layout.annotations), 1)
        self.assertIn("End of Training", fig.layout.annotations[0].text)

    @patch("spotforecast2.manager.plotter.get_data_home")
    def test_make_plot_success(self, mock_get_home):
        """Test make_plot returns figure and saves to default path."""
        mock_get_home.return_value = self.test_dir
        
        fig = make_plot(self.mock_pkg)
        
        self.assertIsInstance(fig, go.Figure)
        expected_path = self.test_dir / "index.html"
        self.assertTrue(expected_path.exists())

    def test_make_plot_custom_path(self):
        """Test make_plot saves to custom path."""
        custom_path = self.test_dir / "subfolder" / "plot.html"
        
        fig = make_plot(self.mock_pkg, output_path=custom_path)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(custom_path.exists())

if __name__ == "__main__":
    unittest.main()
