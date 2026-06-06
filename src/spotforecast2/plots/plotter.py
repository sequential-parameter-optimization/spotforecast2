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

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        from spotforecast2.plots.plotter import PredictionFigure

        rng = np.random.default_rng(0)
        train_idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        future_idx = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
        pkg = {
            "train_actual": pd.Series(rng.uniform(80, 120, 48), index=train_idx),
            "train_pred": pd.Series(rng.uniform(80, 120, 48), index=train_idx),
            "future_actual": pd.Series(rng.uniform(80, 120, 24), index=future_idx),
            "future_pred": pd.Series(rng.uniform(80, 120, 24), index=future_idx),
            "metrics_train": {"mae": 2.5, "mape": 0.02},
            "metrics_future": {"mae": 3.0, "mape": 0.03},
            "metrics_future_one_day": {"mae": 2.8, "mape": 0.025},
        }
        pf = PredictionFigure(pkg, title="Demo Forecast")
        fig = pf.make_plot()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # at least actual, prediction, last-week traces
        print(f"Figure has {len(fig.data)} traces, title: '{pf.title}'")
        ```
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
            ```{python}
            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go

            from spotforecast2.plots.plotter import PredictionFigure

            rng = np.random.default_rng(0)
            dates = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
            train_end = dates[70]
            y = pd.Series(rng.uniform(0, 100, 100), index=dates, name="load")
            p = y + rng.normal(0, 5, 100)
            pkg = {
                "train_actual": y.loc[:train_end],
                "future_actual": y.loc[train_end:],
                "train_pred": p.loc[:train_end],
                "future_pred": p.loc[train_end:],
                "metrics_train": {"mae": 5.0, "mape": 0.1},
                "metrics_future": {"mae": 6.0, "mape": 0.12},
                "metrics_future_one_day": {"mae": 4.5, "mape": 0.08},
            }
            fig = PredictionFigure(pkg).make_plot()
            assert isinstance(fig, go.Figure)
            print(f"Traces: {len(fig.data)}, type: {type(fig).__name__}")
            ```
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

        # Last-week comparison: shift training-actual timestamps forward by 7 days
        # so the trace extends through the prediction horizon, not just to end_training.
        # Using index-based shifting (+ Timedelta) rather than positional shift() ensures
        # the trace covers both the visible training tail AND the full forecast window.
        y_last_week = self.prediction_package["train_actual"].copy()
        y_last_week.index = y_last_week.index + pd.Timedelta(weeks=1)
        y_last_week_visible = y_last_week[
            (y_last_week.index >= min_range) & (y_last_week.index <= max_range)
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
            yaxis=dict(title="Load"),
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

    def save_to_file(
        self,
        path: Union[str, Path],
        **kwargs: Any,
    ) -> Path:
        """Write the figure to a static image file (PNG, SVG, PDF, …).

        Thin wrapper around `plotly.graph_objects.Figure.write_image()`.
        Requires the ``kaleido`` package (declared in
        ``pyproject.toml``).

        Args:
            path: Destination file path.  The format is inferred from the
                extension (``.png``, ``.svg``, ``.pdf``, …).
            **kwargs: Forwarded to ``fig.write_image`` (e.g. ``width``,
                ``height``, ``scale``).

        Returns:
            The resolved `pathlib.Path` of the written file.

        Raises:
            FileNotFoundError: If the parent directory does not exist.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path

            import numpy as np
            import pandas as pd

            from spotforecast2.plots.plotter import PredictionFigure

            rng = np.random.default_rng(0)
            train_idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
            future_idx = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
            pkg = {
                "train_actual": pd.Series(rng.standard_normal(48), index=train_idx),
                "train_pred": pd.Series(rng.standard_normal(48), index=train_idx),
                "future_actual": pd.Series(rng.standard_normal(24), index=future_idx),
                "future_pred": pd.Series(rng.standard_normal(24), index=future_idx),
                "metrics_train": {"mae": 1.0, "mape": 0.1},
                "metrics_future": {"mae": 1.0, "mape": 0.1},
                "metrics_future_one_day": {"mae": 1.0, "mape": 0.1},
            }
            fig = PredictionFigure(pkg)
            fig.make_plot()
            with tempfile.TemporaryDirectory() as tmp:
                out = fig.save_to_file(Path(tmp) / "out.png")
                print(out.exists())
            ```
        """
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")
        # kaleido v1 cannot serialise pandas Timestamp axis values directly
        # (orjson rejects them).  Round-trip through Plotly's own JSON
        # encoder, which coerces Timestamps to ISO strings.
        import plotly.io as pio

        pio.from_json(self.fig.to_json()).write_image(str(path), **kwargs)
        return path

    def write_to_file(
        self,
        path: Union[str, Path],
        *,
        template_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Render the figure into a standalone HTML page via a Jinja2 template.

        The default template wraps the Plotly figure in a minimal HTML
        skeleton with the figure title as ``<title>``.  Provide
        ``template_path`` to substitute a custom Jinja2 template that
        receives the variable ``fig`` (an HTML fragment) and ``title``.

        Args:
            path: Destination HTML file path.
            template_path: Optional path to a custom Jinja2 template.

        Returns:
            The resolved `pathlib.Path` of the written file.

        Raises:
            FileNotFoundError: If the parent directory or
                ``template_path`` does not exist.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path

            import numpy as np
            import pandas as pd

            from spotforecast2.plots.plotter import PredictionFigure

            rng = np.random.default_rng(0)
            train_idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
            future_idx = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
            pkg = {
                "train_actual": pd.Series(rng.standard_normal(48), index=train_idx),
                "train_pred": pd.Series(rng.standard_normal(48), index=train_idx),
                "future_actual": pd.Series(rng.standard_normal(24), index=future_idx),
                "future_pred": pd.Series(rng.standard_normal(24), index=future_idx),
                "metrics_train": {"mae": 1.0, "mape": 0.1},
                "metrics_future": {"mae": 1.0, "mape": 0.1},
                "metrics_future_one_day": {"mae": 1.0, "mape": 0.1},
            }
            fig = PredictionFigure(pkg, title="Demo")
            fig.make_plot()
            with tempfile.TemporaryDirectory() as tmp:
                out = fig.write_to_file(Path(tmp) / "report.html")
                print(out.suffix)
            ```
        """
        from importlib import resources

        from jinja2 import Environment, select_autoescape

        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

        if template_path is not None:
            tpl_path = Path(template_path)
            if not tpl_path.exists():
                raise FileNotFoundError(f"Template file does not exist: {tpl_path}")
            template_source = tpl_path.read_text(encoding="utf-8")
        else:
            template_source = (
                resources.files("spotforecast2.plots.assets")
                .joinpath("prediction_report.html.j2")
                .read_text(encoding="utf-8")
            )

        fig_html = self.fig.to_html(full_html=False, include_plotlyjs="cdn")
        env = Environment(autoescape=select_autoescape(default=True))
        rendered = env.from_string(template_source).render(
            fig=fig_html, title=self.title
        )
        path.write_text(rendered, encoding="utf-8")
        return path


def make_plot(
    prediction_package: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Energy Demand Prediction",
    save: bool = True,
) -> go.Figure:
    """
    Generate and optionally save an interactive prediction plot.

    Args:
        prediction_package: Dictionary of results (actuals, preds, metrics).
        output_path: Path to save the HTML file. If None, it defaults to
            'index.html' in the package's data home directory.
        title: Figure title shown at the top of the plot.
        save: If True (default), write the figure to ``output_path``.  Pass
            ``False`` to obtain a figure without any disk side-effect (used by
            the multitask pipeline, where many figures are produced per run
            and the auto-write would otherwise overwrite the same file
            repeatedly).

    Returns:
        The generated Plotly Figure object.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        from spotforecast2.plots.plotter import make_plot

        rng = np.random.default_rng(1)
        train_idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        future_idx = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
        pkg = {
            "train_actual": pd.Series(rng.uniform(80, 120, 48), index=train_idx),
            "train_pred": pd.Series(rng.uniform(80, 120, 48), index=train_idx),
            "future_actual": pd.Series(rng.uniform(80, 120, 24), index=future_idx),
            "future_pred": pd.Series(rng.uniform(80, 120, 24), index=future_idx),
            "metrics_train": {"mae": 2.5, "mape": 0.02},
            "metrics_future": {"mae": 3.0, "mape": 0.03},
            "metrics_future_one_day": {"mae": 2.8, "mape": 0.025},
        }
        fig = make_plot(pkg, save=False)
        assert isinstance(fig, go.Figure)
        print(f"Figure type: {type(fig).__name__}, traces: {len(fig.data)}")
        ```
    """
    predictor_fig = PredictionFigure(prediction_package, title=title)
    fig = predictor_fig.make_plot()

    if not save:
        return fig

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
        ```{python}
        import numpy as np
        import pandas as pd

        from spotforecast2.plots.plotter import plot_actual_vs_predicted

        rng = np.random.default_rng(42)
        index = pd.date_range("2020-01-01", periods=24, freq="h")
        actual = pd.Series(
            100 + 10 * np.sin(np.arange(24) * 2 * np.pi / 24), index=index
        )
        baseline = actual + rng.normal(0, 2, 24)
        covariates = actual + rng.normal(0, 1, 24)
        custom = actual + rng.normal(0, 0.5, 24)

        assert isinstance(actual.index, pd.DatetimeIndex)
        assert (actual.index == baseline.index).all()
        mae_baseline = abs(actual - baseline).mean()
        mae_custom = abs(actual - custom).mean()
        print(f"Baseline MAE: {mae_baseline:.2f}, Custom MAE: {mae_custom:.2f}")
        assert mae_custom < mae_baseline  # tighter noise → lower error
        ```
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


def plot_with_outliers(
    df_pipeline: pd.DataFrame,
    df_pipeline_original: pd.DataFrame,
    config: Any,
    targets: Optional[list[str]] = None,
) -> None:
    """Interactive time series plot with outliers and optional bounds.

    This function generates an interactive Plotly figure that visualizes the
    time series data from the pipeline, highlighting any detected outliers.
    Regular data points are shown in light grey, while outliers are marked in
    red. When ``config.bounds`` is set, two horizontal reference lines in
    lightblue are added per plot — one for the lower bound and one for the
    upper bound — to indicate the acceptable value range for that target.

    The plot title includes the percentage of outliers detected for each
    target variable.

    The resolved list of target column names must be passed explicitly via
    *targets*.  Callers inside the pipeline (e.g. ``PlottingMixin``) obtain
    this list from ``task.run_state.targets`` after ``prepare_data`` has run.
    A ``SimpleNamespace``/dict-like *config* that carries its own ``targets``
    attribute is still accepted for backwards-compatible standalone usage —
    the explicit *targets* argument takes precedence when provided.

    Args:
        df_pipeline (pd.DataFrame): The processed DataFrame from the pipeline,
            which may contain NaN values where outliers have been detected and
            removed.
        df_pipeline_original (pd.DataFrame): The original DataFrame before
            outlier removal.
        config: Configuration object carrying ``bounds`` (optional list of
            ``(lower, upper)`` tuples, one per target, in the same order as
            *targets*).  ``config.targets`` is used as a fallback when the
            *targets* argument is ``None`` (legacy path).
        targets: Resolved list of target column names.  When ``None`` the
            function falls back to ``config.targets`` (legacy callers that
            pass a ``SimpleNamespace`` with ``targets`` set).

    Returns:
        None. Displays one interactive Plotly figure per target variable.

    Examples:
        ```{python}
        import pandas as pd
        import numpy as np
        from types import SimpleNamespace
        from spotforecast2.plots.plotter import plot_with_outliers
        # Create synthetic data
        dates = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
        data = pd.DataFrame({
            "target1": np.random.rand(100) * 100,
            "target2": np.random.rand(100) * 50,
        }, index=dates)
        # Introduce outliers
        data.loc[dates[10], "target1"] = 300  # Outlier in target1
        data.loc[dates[20], "target2"] = 150  # Outlier in target2
        df_pipeline = data.copy()
        df_pipeline.loc[[dates[10], dates[20]], ["target1", "target2"]] = np.nan
        # Config with bounds; targets passed explicitly
        config = SimpleNamespace(bounds=[(-10, 200), (0, 100)])
        plot_with_outliers(df_pipeline, data, config, targets=["target1", "target2"])
        ```
    """
    bounds = getattr(config, "bounds", None)
    _targets = targets if targets is not None else config.targets

    for i, target in enumerate(_targets):
        fig = go.Figure()

        # Plot Regular Data (lightgrey)
        fig.add_trace(
            go.Scatter(
                x=df_pipeline.index,
                y=df_pipeline[target],
                mode="lines",
                name="Regular Data",
                line=dict(color="lightgrey"),
            )
        )

        # Plot Outliers (Red)
        outliers = df_pipeline_original.loc[df_pipeline[target].isna(), target]
        if not outliers.empty:
            fig.add_trace(
                go.Scatter(
                    x=outliers.index,
                    y=outliers,
                    mode="markers",
                    name="Outliers",
                    marker=dict(color="red", size=8, symbol="x"),
                )
            )

        # Add lower and upper bound lines when available
        if bounds is not None and i < len(bounds):
            lower_bound, upper_bound = bounds[i]
            fig.add_hline(
                y=lower_bound,
                line=dict(color="lightblue", width=2, dash="dash"),
                annotation_text=f"Lower bound: {lower_bound}",
                annotation_position="bottom right",
                annotation=dict(font=dict(color="steelblue")),
            )
            fig.add_hline(
                y=upper_bound,
                line=dict(color="lightblue", width=2, dash="dash"),
                annotation_text=f"Upper bound: {upper_bound}",
                annotation_position="top right",
                annotation=dict(font=dict(color="steelblue")),
            )

        pct_outliers = (len(outliers) / len(df_pipeline)) * 100

        fig.update_layout(
            title=f"{target} Time Series inc. Manually Labelled Outliers ({pct_outliers:.2f}%)",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.show()
