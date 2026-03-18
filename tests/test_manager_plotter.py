# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import plotly.graph_objects as go
from spotforecast2.manager.plotter import make_plot, PredictionFigure


def _make_pkg(
    train_hours: int = 48,
    future_hours: int = 24,
    include_forecast: bool = True,
    test_actual: pd.Series | None = None,
    empty_future_actual: bool = False,
):
    """Build a minimal prediction package for testing."""
    dates_train = pd.date_range("2026-01-01", periods=train_hours, freq="h", tz="UTC")
    dates_future = pd.date_range("2026-01-03", periods=future_hours, freq="h", tz="UTC")

    future_actual = (
        pd.Series(dtype="float64")
        if empty_future_actual
        else pd.Series(range(future_hours), index=dates_future)
    )

    pkg = {
        "train_actual": pd.Series(range(train_hours), index=dates_train),
        "future_actual": future_actual,
        "train_pred": pd.Series(range(train_hours), index=dates_train),
        "future_pred": pd.Series(range(future_hours), index=dates_future),
        "metrics_train": {"mae": 1.0, "mape": 0.1},
        "metrics_future": {"mae": 2.0, "mape": 0.2},
        "metrics_future_one_day": {"mae": 1.5, "mape": 0.15},
    }
    if include_forecast:
        pkg["future_forecast"] = pd.Series(range(future_hours), index=dates_future)
        pkg["metrics_forecast"] = {"mae": 3.0, "mape": 0.3}
        pkg["metrics_forecast_one_day"] = {"mae": 2.5, "mape": 0.25}
    if test_actual is not None:
        pkg["test_actual"] = test_actual
    return pkg


class TestPredictionFigureBackwardCompat(unittest.TestCase):
    """Existing behaviour is preserved for callers that don't pass new params."""

    def setUp(self):
        self.mock_pkg = _make_pkg()

    def test_returns_figure(self):
        fig = PredictionFigure(self.mock_pkg).make_plot()
        self.assertIsInstance(fig, go.Figure)

    def test_trace_count_with_forecast(self):
        # actual, pred, benchmark, test_actual (absent), last_week = 4
        fig = PredictionFigure(self.mock_pkg).make_plot()
        self.assertEqual(len(fig.data), 4)

    def test_trace_count_without_forecast(self):
        pkg = _make_pkg(include_forecast=False)
        fig = PredictionFigure(pkg).make_plot()
        self.assertEqual(len(fig.data), 3)  # actual, pred, last_week

    def test_end_of_training_annotation(self):
        fig = PredictionFigure(self.mock_pkg).make_plot()
        self.assertEqual(len(fig.layout.annotations), 1)
        self.assertIn("End of Training", fig.layout.annotations[0].text)

    def test_default_title(self):
        fig = PredictionFigure(self.mock_pkg).make_plot()
        self.assertEqual(fig.layout.title.text, "Energy Demand Prediction")


class TestPredictionFigureTitle(unittest.TestCase):
    """title parameter is forwarded to the figure layout."""

    def test_custom_title(self):
        pkg = _make_pkg()
        fig = PredictionFigure(pkg, title="Forecast for Target A").make_plot()
        self.assertEqual(fig.layout.title.text, "Forecast for Target A")

    def test_empty_title(self):
        pkg = _make_pkg()
        fig = PredictionFigure(pkg, title="").make_plot()
        self.assertEqual(fig.layout.title.text, "")


class TestPredictionFigureTestActual(unittest.TestCase):
    """test_actual key in the package adds a fifth green trace."""

    def setUp(self):
        dates_future = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
        self.test_actual = pd.Series(range(24), index=dates_future)

    def test_trace_added_when_test_actual_present(self):
        pkg = _make_pkg(include_forecast=False, test_actual=self.test_actual)
        fig = PredictionFigure(pkg).make_plot()
        # actual, pred, test_actual, last_week = 4
        self.assertEqual(len(fig.data), 4)

    def test_test_actual_trace_name(self):
        pkg = _make_pkg(include_forecast=False, test_actual=self.test_actual)
        fig = PredictionFigure(pkg).make_plot()
        names = [t.name for t in fig.data]
        self.assertIn("Actual (test / ground truth)", names)

    def test_no_test_actual_trace_when_absent(self):
        pkg = _make_pkg(include_forecast=False)
        fig = PredictionFigure(pkg).make_plot()
        names = [t.name for t in fig.data]
        self.assertNotIn("Actual (test / ground truth)", names)

    def test_empty_test_actual_not_added(self):
        empty = pd.Series(dtype="float64")
        pkg = _make_pkg(include_forecast=False, test_actual=empty)
        fig = PredictionFigure(pkg).make_plot()
        names = [t.name for t in fig.data]
        self.assertNotIn("Actual (test / ground truth)", names)


class TestPredictionFigureGenuineFutureMode(unittest.TestCase):
    """Genuine-future mode: future_actual is empty; forecast must still be visible."""

    def setUp(self):
        self.pkg = _make_pkg(include_forecast=False, empty_future_actual=True)

    def test_returns_figure(self):
        fig = PredictionFigure(self.pkg).make_plot()
        self.assertIsInstance(fig, go.Figure)

    def test_xaxis_extends_to_forecast(self):
        """X-axis max must reach beyond end_training into the forecast window."""
        fig = PredictionFigure(self.pkg).make_plot()
        end_training = self.pkg["train_actual"].index.max()
        future_end = self.pkg["future_pred"].index.max()
        x_range = fig.layout.xaxis.range
        # max of the x-axis range must be after end_training
        x_max = pd.Timestamp(x_range[1])
        self.assertGreater(x_max, end_training)
        # and must reach (at least) the last forecast timestamp
        self.assertGreaterEqual(x_max, future_end)


class TestPredictionFigureTrainingDataClipping(unittest.TestCase):
    """Training traces are clipped so that only the last 1-day slice is rendered."""

    def test_clipped_trace_shorter_than_full_training(self):
        # 200 training hours; only the last 24 h + 1 h buffer are visible
        pkg = _make_pkg(train_hours=200, include_forecast=False)
        fig = PredictionFigure(pkg).make_plot()
        # First trace is "Total system load (actual)" — must be clipped
        actual_trace_len = len(fig.data[0].x)
        self.assertLess(actual_trace_len, 200)

    def test_clipped_trace_covers_visible_window(self):
        pkg = _make_pkg(train_hours=200, include_forecast=False)
        fig = PredictionFigure(pkg).make_plot()
        end_training = pkg["train_actual"].index.max()
        min_range = end_training - pd.Timedelta(days=1)
        first_x = pd.Timestamp(fig.data[0].x[0])
        self.assertGreaterEqual(first_x, min_range)


class TestMakePlot(unittest.TestCase):
    """make_plot() wrapper: saves HTML and forwards title."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.mock_pkg = _make_pkg()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2.manager.plotter.get_data_home")
    def test_make_plot_saves_to_default_path(self, mock_get_home):
        mock_get_home.return_value = self.test_dir
        fig = make_plot(self.mock_pkg)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue((self.test_dir / "index.html").exists())

    def test_make_plot_saves_to_custom_path(self):
        custom_path = self.test_dir / "subfolder" / "plot.html"
        fig = make_plot(self.mock_pkg, output_path=custom_path)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(custom_path.exists())

    def test_make_plot_forwards_title(self):
        custom_path = self.test_dir / "out.html"
        fig = make_plot(self.mock_pkg, output_path=custom_path, title="My Title")
        self.assertEqual(fig.layout.title.text, "My Title")


if __name__ == "__main__":
    unittest.main()
