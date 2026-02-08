# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Module for generating interactive prediction plots.

This module provides the `PredictionFigure` class and `make_plot` function
to visualize time series forecasting results, including actual values,
predictions, and performance metrics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from spotforecast2_safe.data.fetch_data import get_data_home

logger = logging.getLogger(__name__)


class PredictionFigure:
    """
    Encapsulates the generation of an interactive Plotly figure for predictions.

    Args:
        prediction_package: A dictionary containing prediction data and metrics.
            Expected keys include:
            - 'train_actual': pd.Series
            - 'future_actual': pd.Series
            - 'train_pred': pd.Series
            - 'future_pred': pd.Series
            - 'future_forecast': pd.Series (e.g., benchmark/ENTSOE)
            - 'metrics_train': Dict[str, float]
            - 'metrics_future': Dict[str, float]
            - 'metrics_future_one_day': Dict[str, float]
            - 'metrics_forecast': Dict[str, float]
            - 'metrics_forecast_one_day': Dict[str, float]
    """

    def __init__(self, prediction_package: Dict[str, Any]):
        self.prediction_package = prediction_package
        self.fig = go.Figure()

    def make_plot(self) -> go.Figure:
        """
        Generate the Plotly figure with traces and annotations.

        Returns:
            The generated plotly.graph_objects.Figure.
        """
        # Combine data for plotting
        y_actual = pd.concat(
            [
                self.prediction_package["train_actual"],
                self.prediction_package["future_actual"],
            ]
        )
        y_pred = pd.concat(
            [
                self.prediction_package["train_pred"],
                self.prediction_package["future_pred"],
            ]
        )
        y_forecast = self.prediction_package.get("future_forecast")

        # Calculate shifting for 'last week' comparison if possible
        y_last_week = y_actual.shift(7 * 24)

        # Calculate prediction horizon
        future_pred_idx = self.prediction_package["future_pred"].index
        total_hours_prediction = (
            future_pred_idx.max() - future_pred_idx.min() + pd.Timedelta(hours=1)
        ).total_seconds() // 3600

        end_training = self.prediction_package["train_actual"].index.max()

        # Legend construction
        actual_legend = "Total system load (actual)<br>"

        m_train = self.prediction_package.get("metrics_train", {})
        m_future_1d = self.prediction_package.get("metrics_future_one_day", {})
        m_future = self.prediction_package.get("metrics_future", {})

        pred_legend = (
            "Total system load (model prediction)<br>"
            f"Training: MAE={m_train.get('mae', 0):.0f}, MAPE={m_train.get('mape', 0):.2f}<br>"
            f"Prediction (24h): MAE={m_future_1d.get('mae', 0):.0f}, MAPE={m_future_1d.get('mape', 0):.2f}<br>"
            f"Prediction ({total_hours_prediction:.0f}h): MAE={m_future.get('mae', 0):.0f}, MAPE={m_future.get('mape', 0):.2f}"
        )

        # Add traces
        self.fig.add_trace(
            go.Scatter(
                x=y_actual.index, y=y_actual, mode="lines+markers", name=actual_legend
            )
        )
        self.fig.add_trace(
            go.Scatter(x=y_pred.index, y=y_pred, mode="lines+markers", name=pred_legend)
        )

        if y_forecast is not None:
            m_f_1d = self.prediction_package.get("metrics_forecast_one_day", {})
            m_f = self.prediction_package.get("metrics_forecast", {})
            forecast_legend = (
                "Benchmark Forecast (e.g., ENTSOE)<br>"
                f"Benchmark (24h): MAE={m_f_1d.get('mae', 0):.0f}, MAPE={m_f_1d.get('mape', 0):.2f}<br>"
                f"Benchmark ({total_hours_prediction:.0f}h): MAE={m_f.get('mae', 0):.0f}, MAPE={m_f.get('mape', 0):.2f}"
            )
            self.fig.add_trace(
                go.Scatter(
                    x=y_forecast.index,
                    y=y_forecast,
                    mode="lines+markers",
                    name=forecast_legend,
                )
            )

        self.fig.add_trace(
            go.Scatter(
                x=y_last_week.index,
                y=y_last_week,
                mode="lines+markers",
                line=dict(dash="dash"),
                name="Actual (last week)",
            )
        )

        # Layout adjustments
        self.fig.update_layout(
            template="plotly_white",
            title="Energy Demand Prediction",
            autosize=True,
            width=None,
            height=700,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
            ),
            xaxis=dict(title="Time (UTC)"),
            yaxis=dict(title="Load [MW]"),
        )

        # Focus range
        max_range = y_actual.index.max()
        min_range = end_training - pd.Timedelta(days=1)
        self.fig.update_xaxes(range=[min_range, max_range])

        # Vertical line for end of training
        self.fig.add_vline(
            x=end_training,
            line_width=2,
            line_color="black",
            line_dash="dash",
        )
        self.fig.add_annotation(
            x=end_training,
            text="End of Training",
            showarrow=False,
            yref="paper",
            y=1.05,
        )

        return self.fig


def make_plot(
    prediction_package: Dict[str, Any], output_path: Optional[Union[str, Path]] = None
) -> go.Figure:
    """
    Generate and optionally save an interactive prediction plot.

    Args:
        prediction_package: Dictionary of results (actuals, preds, metrics).
        output_path: Path to save the HTML file. If None, it defaults to
            'index.html' in the package's data home directory.

    Returns:
        The generated Plotly Figure object.

    Examples:
        >>> from spotforecast2.manager.plotter import make_plot
        >>> # fig = make_plot(results)
    """
    predictor_fig = PredictionFigure(prediction_package)
    fig = predictor_fig.make_plot()

    if output_path is None:
        data_home = get_data_home()
        output_path = data_home / "index.html"
    else:
        output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        logger.info("Plot saved to %s", output_path)
    except Exception as e:
        logger.error("Failed to save plot to %s: %s", output_path, e)

    return fig
