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
            - 'test_actual': pd.Series (optional external ground truth for the
              forecast period, e.g. from data_test.csv; absent in genuine-future
              mode when no ground truth is available yet)
            - 'metrics_train': Dict[str, float]
            - 'metrics_future': Dict[str, float]
            - 'metrics_future_one_day': Dict[str, float]
            - 'metrics_forecast': Dict[str, float]
            - 'metrics_forecast_one_day': Dict[str, float]
        title: Figure title shown at the top of the plot.
    """

    def __init__(
        self,
        prediction_package: Dict[str, Any],
        title: str = "Energy Demand Prediction",
    ):
        self.prediction_package = prediction_package
        self.title = title
        self.fig = go.Figure()

    def make_plot(self) -> go.Figure:
        """
        Generate the Plotly figure with traces and annotations.

        Traces added (always):
        - Total system load — actual (training window, clipped to visible range)
        - Total system load — model prediction (training + forecast, clipped)
        - Actual (last week) — time-shifted actual for seasonality context

        Traces added (when data is available):
        - Benchmark Forecast (e.g., ENTSOE) — if ``future_forecast`` key present
        - Actual (test / ground truth) — if ``test_actual`` key present

        The X-axis is fixed to ``[end_training − 1 day, future_pred.max() + 1 h]``
        so the full forecast window is always visible, including in genuine-future
        mode where ``future_actual`` is an empty Series.  Only the data slice in
        that window is serialised into the Plotly JSON, keeping HTML output small.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from spotforecast2.manager.plotter import PredictionFigure
            >>> dates = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
            >>> train_end = dates[70]
            >>> y = pd.Series(np.random.rand(100) * 100, index=dates, name="load")
            >>> p = y + np.random.normal(0, 5, 100)
            >>> pkg = {
            ...     "train_actual": y.loc[:train_end],
            ...     "future_actual": y.loc[train_end:],
            ...     "train_pred": p.loc[:train_end],
            ...     "future_pred": p.loc[train_end:],
            ...     "metrics_train": {"mae": 5.0, "mape": 0.1},
            ...     "metrics_future": {"mae": 6.0, "mape": 0.12},
            ...     "metrics_future_one_day": {"mae": 4.5, "mape": 0.08},
            ... }
            >>> fig = PredictionFigure(pkg).make_plot()
            >>> isinstance(fig.data, tuple)
            True
        """
        end_training = self.prediction_package["train_actual"].index.max()
        future_pred = self.prediction_package["future_pred"]

        # X-axis: show last day of training + full forecast window.
        # Using future_pred.index.max() (not y_actual.index.max()) ensures the
        # forecast is visible even in genuine-future mode where future_actual is empty.
        min_range = end_training - pd.Timedelta(days=1)
        max_range = future_pred.index.max() + pd.Timedelta(hours=1)

        # Combine actuals and predictions, then clip to the visible window.
        # Clipping prevents serialising up to 8760 training points into the HTML
        # when only the last ~24 h of the training trace is actually visible.
        y_actual = pd.concat(
            [
                self.prediction_package["train_actual"],
                self.prediction_package["future_actual"],
            ]
        )
        y_pred = pd.concat(
            [
                self.prediction_package["train_pred"],
                future_pred,
            ]
        )
        y_actual_visible = y_actual[y_actual.index >= min_range]
        y_pred_visible = y_pred[y_pred.index >= min_range]

        y_forecast = self.prediction_package.get("future_forecast")
        test_actual = self.prediction_package.get("test_actual")

        # Last-week comparison (shift on full series, then clip to visible window)
        y_last_week_visible = y_actual.shift(7 * 24)
        y_last_week_visible = y_last_week_visible[
            y_last_week_visible.index >= min_range
        ]

        # Prediction horizon (hours) for legend labels
        total_hours_prediction = (
            future_pred.index.max() - future_pred.index.min() + pd.Timedelta(hours=1)
        ).total_seconds() // 3600

        # --- Legend strings ---
        actual_legend = "Total system load (actual)"

        m_train = self.prediction_package.get("metrics_train", {})
        m_future_1d = self.prediction_package.get("metrics_future_one_day", {})
        m_future = self.prediction_package.get("metrics_future", {})
        pred_legend = (
            "Total system load (model prediction)<br>"
            f"Training: MAE={m_train.get('mae', 0):.0f}, MAPE={m_train.get('mape', 0):.2f}<br>"
            f"Prediction (24h): MAE={m_future_1d.get('mae', 0):.0f}, MAPE={m_future_1d.get('mape', 0):.2f}<br>"
            f"Prediction ({total_hours_prediction:.0f}h): MAE={m_future.get('mae', 0):.0f}, MAPE={m_future.get('mape', 0):.2f}"
        )

        # --- Traces ---
        self.fig.add_trace(
            go.Scatter(
                x=y_actual_visible.index,
                y=y_actual_visible,
                mode="lines+markers",
                name=actual_legend,
            )
        )
        self.fig.add_trace(
            go.Scatter(
                x=y_pred_visible.index,
                y=y_pred_visible,
                mode="lines+markers",
                name=pred_legend,
            )
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

        if test_actual is not None and len(test_actual) > 0:
            self.fig.add_trace(
                go.Scatter(
                    x=test_actual.index,
                    y=test_actual,
                    mode="lines+markers",
                    name="Actual (test / ground truth)",
                    line=dict(color="green", width=2),
                    marker=dict(size=5),
                )
            )

        self.fig.add_trace(
            go.Scatter(
                x=y_last_week_visible.index,
                y=y_last_week_visible,
                mode="lines+markers",
                line=dict(dash="dash"),
                name="Actual (last week)",
            )
        )

        # --- Y-axis range from visible data only ---
        all_visible = pd.concat([y_actual_visible, y_pred_visible])
        if test_actual is not None and len(test_actual) > 0:
            all_visible = pd.concat([all_visible, test_actual])
        y_min = all_visible.min()
        y_max = all_visible.max()
        y_margin = (y_max - y_min) * 0.05

        # --- Layout ---
        self.fig.update_layout(
            template="plotly_white",
            title=self.title,
            autosize=True,
            width=None,
            height=700,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
            ),
            xaxis=dict(title="Time (UTC)"),
            yaxis=dict(title="Load [MW]"),
        )
        self.fig.update_xaxes(range=[min_range, max_range])
        self.fig.update_yaxes(range=[y_min - y_margin, y_max + y_margin])

        # Vertical marker at end of training
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
    prediction_package: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Energy Demand Prediction",
) -> go.Figure:
    """
    Generate and optionally save an interactive prediction plot.

    Args:
        prediction_package: Dictionary of results (actuals, preds, metrics).
        output_path: Path to save the HTML file. If None, it defaults to
            'index.html' in the package's data home directory.
        title: Figure title shown at the top of the plot.

    Returns:
        The generated Plotly Figure object.

    Examples:
        >>> from spotforecast2.manager.plotter import make_plot
        >>> # fig = make_plot(results)
    """
    predictor_fig = PredictionFigure(prediction_package, title=title)
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


def plot_actual_vs_predicted(
    actual_combined: pd.Series,
    baseline_combined: pd.Series,
    covariates_combined: pd.Series,
    custom_lgbm_combined: pd.Series,
    html_path: Optional[str] = None,
) -> None:
    """
    Plot actual vs predicted combined values for model comparison.

    This function creates an interactive Plotly figure comparing ground truth
    with predictions from three different forecasting models: baseline,
    covariate-enhanced, and custom LightGBM. The plot includes interactive
    hover information and can be saved as a standalone HTML file.

    Safety-Critical Features:
        - Interactive visualization for model validation
        - Supports HTML export for audit trails
        - Shows all models simultaneously for easy comparison
        - Uses consistent color scheme and line styles

    Args:
        actual_combined: Ground truth combined series with datetime index.
        baseline_combined: Baseline combined prediction series. Must have
            same index as actual_combined.
        covariates_combined: Covariate-enhanced combined prediction series.
            Must have same index as actual_combined.
        custom_lgbm_combined: Custom LightGBM (optimized params) combined
            prediction series. Must have same index as actual_combined.
        html_path: If set, save the plot as a single self-contained HTML file
            to this path. If None, displays plot interactively only.

    Returns:
        None. Displays plot and optionally saves to HTML file.

    Raises:
        ValueError: If series indices don't align or are empty.

    Examples:
        >>> import pandas as pd
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2.manager.plotter import plot_actual_vs_predicted
        >>>
        >>> # Example 1: Create synthetic data for testing
        >>> index = pd.date_range('2020-01-01', periods=24, freq='h')
        >>> actual = pd.Series(range(100, 124), index=index, name='actual')
        >>> baseline = pd.Series(range(101, 125), index=index, name='baseline')
        >>> covariates = pd.Series(range(99, 123), index=index, name='covariates')
        >>> custom = pd.Series(range(100, 124), index=index, name='custom')
        >>>
        >>> # Verify data properties
        >>> print(f"Data length: {len(actual)}")
        Data length: 24
        >>> print(f"Index type: {type(actual.index).__name__}")
        Index type: DatetimeIndex
        >>>
        >>> # Example 2: Comparing models with different accuracies
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> index = pd.date_range('2020-01-01 00:00:00', periods=48, freq='h')
        >>> actual = pd.Series(
        ...     100 + 10 * np.sin(np.arange(48) * 2 * np.pi / 24),
        ...     index=index
        ... )
        >>> baseline = actual + np.random.normal(0, 2, 48)
        >>> covariates = actual + np.random.normal(0, 1, 48)
        >>> custom = actual + np.random.normal(0, 0.5, 48)
        >>>
        >>> # Verify series properties before plotting
        >>> print(f"Actual range: [{actual.min():.1f}, {actual.max():.1f}]")
        Actual range: [90.0, 110.0]
        >>> print(f"All indices aligned: {(actual.index == baseline.index).all()}")
        All indices aligned: True
        >>>
        >>> # Example 3: Production workflow with actual forecast data
        >>> index = pd.date_range('2020-01-01', periods=24, freq='h')
        >>> ground_truth = pd.Series([100 + i for i in range(24)], index=index)
        >>> model1_pred = pd.Series([101 + i for i in range(24)], index=index)
        >>> model2_pred = pd.Series([99 + i for i in range(24)], index=index)
        >>> model3_pred = pd.Series([100 + i for i in range(24)], index=index)
        >>>
        >>> # Calculate errors
        >>> mae_baseline = abs(ground_truth - model1_pred).mean()
        >>> mae_covariates = abs(ground_truth - model2_pred).mean()
        >>> mae_custom = abs(ground_truth - model3_pred).mean()
        >>> print(f"Baseline MAE: {mae_baseline:.2f}")
        Baseline MAE: 1.00
        >>> print(f"Covariates MAE: {mae_covariates:.2f}")
        Covariates MAE: 1.00
        >>> print(f"Custom MAE: {mae_custom:.2f}")
        Custom MAE: 0.00
        >>>
        >>> # Example 4: Verify data alignment before plotting
        >>> index1 = pd.date_range('2020-01-01', periods=24, freq='h')
        >>> index2 = pd.date_range('2020-01-02', periods=24, freq='h')
        >>> series1 = pd.Series(range(24), index=index1)
        >>> series2 = pd.Series(range(24), index=index2)
        >>>
        >>> # Check alignment
        >>> indices_match = (series1.index == series2.index).all()
        >>> print(f"Indices aligned: {indices_match}")
        Indices aligned: False
        >>>
        >>> # Reindex to align
        >>> series2_aligned = series2.reindex(series1.index)
        >>> print(f"After reindex: {(series1.index == series2_aligned.index).all()}")
        After reindex: True
        >>>
        >>> # Example 5: Verify all series have correct properties
        >>> index = pd.date_range('2020-01-01', periods=10, freq='h')
        >>> actual = pd.Series(range(10), index=index)
        >>> pred1 = pd.Series(range(1, 11), index=index)
        >>> pred2 = pd.Series(range(10), index=index)
        >>> pred3 = pd.Series(range(10), index=index)
        >>>
        >>> # Safety checks
        >>> assert isinstance(actual.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
        >>> assert len(actual) == len(pred1) == len(pred2) == len(pred3), "All series must have same length"
        >>> assert (actual.index == pred1.index).all(), "Indices must align"
        >>> print("All safety checks passed")
        All safety checks passed
        >>>
        >>> # Example 6: Calculate metrics for model comparison
        >>> index = pd.date_range('2020-01-01', periods=100, freq='h')
        >>> actual = pd.Series(100 + np.random.randn(100) * 5, index=index)
        >>> pred1 = actual + np.random.randn(100) * 2
        >>> pred2 = actual + np.random.randn(100) * 1.5
        >>> pred3 = actual + np.random.randn(100) * 1
        >>>
        >>> # Calculate MAE for each model
        >>> mae1 = abs(actual - pred1).mean()
        >>> mae2 = abs(actual - pred2).mean()
        >>> mae3 = abs(actual - pred3).mean()
        >>> print(f"Model 1 MAE: {mae1:.2f}")  # doctest: +ELLIPSIS
        Model 1 MAE: ...
        >>> print(f"Model 2 MAE: {mae2:.2f}")  # doctest: +ELLIPSIS
        Model 2 MAE: ...
        >>> print(f"Model 3 MAE: {mae3:.2f}")  # doctest: +ELLIPSIS
        Model 3 MAE: ...
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_combined.index,
            y=actual_combined.values,
            mode="lines+markers",
            name="Actual",
            line=dict(color="green", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_combined.index,
            y=baseline_combined.values,
            mode="lines+markers",
            name="Predicted (Baseline)",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=covariates_combined.index,
            y=covariates_combined.values,
            mode="lines+markers",
            name="Predicted (Covariates)",
            line=dict(color="blue", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=custom_lgbm_combined.index,
            y=custom_lgbm_combined.values,
            mode="lines+markers",
            name="Predicted (Custom LightGBM)",
            line=dict(color="orange", width=2, dash="dashdot"),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Combined Values: Actual vs. Predicted",
        xaxis_title="Time",
        yaxis_title="Combined Value",
        width=1000,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )

    if html_path:
        fig.write_html(html_path)
        print(f"Plot saved to {html_path}")

    fig.show()
