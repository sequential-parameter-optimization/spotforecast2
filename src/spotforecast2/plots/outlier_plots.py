# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from spotforecast2_safe.preprocessing.outlier import get_outliers


def visualize_outliers_hist(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    columns: Optional[list[str]] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
    figsize: tuple[int, int] = (10, 5),
    bins: int = 50,
    **kwargs: Any,
) -> None:
    """Visualize outliers in DataFrame using stacked histograms.

    Creates a histogram for each specified column, displaying both regular data
    and detected outliers in different colors. Uses IsolationForest for outlier
    detection.

    Args:
        data: The DataFrame with cleaned data (outliers may be NaN).
        data_original: The original DataFrame before outlier detection.
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        contamination: The estimated proportion of outliers in the dataset.
            Default: 0.01.
        random_state: Random seed for reproducibility. Default: 1234.
        figsize: Figure size as (width, height). Default: (10, 5).
        bins: Number of histogram bins. Default: 50.
        **kwargs: Additional keyword arguments passed to plt.hist() (e.g., color,
            alpha, edgecolor, etc.).

    Returns:
        None. Displays matplotlib figures.

    Raises:
        ValueError: If data or data_original is empty, or if specified columns
            don't exist.
        ImportError: If matplotlib is not installed.

    Examples:
        ```{python}
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for doc rendering
        import numpy as np
        import pandas as pd
        from spotforecast2.plots.outlier_plots import visualize_outliers_hist

        rng = np.random.default_rng(0)
        normal_vals = rng.normal(loc=20.0, scale=2.0, size=28)
        outlier_vals = [60.0, 65.0]  # two obvious outliers
        data_original = pd.DataFrame(
            {"temperature": np.concatenate([normal_vals, outlier_vals])}
        )
        data_cleaned = data_original.copy()

        # Renders a stacked histogram; outliers shown in red
        visualize_outliers_hist(
            data_cleaned,
            data_original,
            columns=["temperature"],
            contamination=0.07,
            figsize=(6, 3),
            bins=15,
            alpha=0.7,
        )
        print("visualize_outliers_hist completed without error")
        ```
    """
    if data.empty or data_original.empty:
        raise ValueError("Input data is empty")

    columns_to_plot = columns if columns is not None else data.columns

    # Validate columns exist
    missing_cols = set(columns_to_plot) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    # Detect outliers
    outliers = get_outliers(
        data_original,
        data_original=data_original,
        contamination=contamination,
        random_state=random_state,
    )

    for col in columns_to_plot:
        # Get inliers (non-NaN values in cleaned data)
        inliers = data[col].dropna()

        # Get outlier values
        outlier_vals = outliers[col]

        # Calculate percentage
        pct_outliers = (len(outlier_vals) / len(data_original)) * 100

        # Create figure
        plt.figure(figsize=figsize)
        plt.hist(
            [inliers, outlier_vals],
            bins=bins,
            stacked=True,
            color=["lightgrey", "red"],
            label=["Regular Data", "Outliers"],
            **kwargs,
        )
        plt.grid(True, alpha=0.3)
        plt.title(f"{col} Distribution with Outliers ({pct_outliers:.2f}%)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()


def visualize_outliers_plotly_scatter(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    columns: Optional[list[str]] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
    **kwargs: Any,
) -> None:
    """Visualize outliers in time series using Plotly scatter plots.

    Creates an interactive time series plot for each specified column, showing
    regular data as a line and detected outliers as scatter points. Uses
    IsolationForest for outlier detection.

    Args:
        data: The DataFrame with cleaned data (outliers may be NaN).
        data_original: The original DataFrame before outlier detection.
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        contamination: The estimated proportion of outliers in the dataset.
            Default: 0.01.
        random_state: Random seed for reproducibility. Default: 1234.
        **kwargs: Additional keyword arguments passed to go.Figure.update_layout()
            (e.g., template, height, etc.).

    Returns:
        None. Displays Plotly figures.

    Raises:
        ValueError: If data or data_original is empty, or if specified columns
            don't exist.
        ImportError: If plotly is not installed.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from spotforecast2_safe.preprocessing.outlier import get_outliers
        from spotforecast2.plots.outlier_plots import visualize_outliers_plotly_scatter

        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=30, freq="h")
        normal_vals = rng.normal(loc=20.0, scale=2.0, size=28)
        outlier_vals_arr = [60.0, 65.0]  # two obvious outliers
        data_original = pd.DataFrame(
            {"temperature": np.concatenate([normal_vals, outlier_vals_arr])},
            index=dates,
        )
        data_cleaned = data_original.copy()

        # Verify that get_outliers detects the planted outliers before plotting
        detected = get_outliers(
            data_original, data_original=data_original, contamination=0.07
        )
        assert len(detected["temperature"]) >= 1, "Expected at least one outlier"

        # Renders an interactive Plotly time series with outliers marked in red
        visualize_outliers_plotly_scatter(
            data_cleaned,
            data_original,
            columns=["temperature"],
            contamination=0.07,
        )
        print(f"Detected {len(detected['temperature'])} outlier(s) in 'temperature'")
        ```
    """
    if go is None:
        raise ImportError(
            "plotly is required for this function. " "Install with: pip install plotly"
        )

    if data.empty or data_original.empty:
        raise ValueError("Input data is empty")

    columns_to_plot = columns if columns is not None else data.columns

    # Validate columns exist
    missing_cols = set(columns_to_plot) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    # Detect outliers
    outliers = get_outliers(
        data_original,
        data_original=data_original,
        contamination=contamination,
        random_state=random_state,
    )

    for col in columns_to_plot:
        fig = go.Figure()

        # Add regular data as line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                mode="lines",
                name="Regular Data",
                line=dict(color="lightgrey"),
            )
        )

        # Add outliers as scatter points
        outlier_vals = outliers[col]
        if not outlier_vals.empty:
            fig.add_trace(
                go.Scatter(
                    x=outlier_vals.index,
                    y=outlier_vals,
                    mode="markers",
                    name="Outliers",
                    marker=dict(color="red", size=8, symbol="x"),
                )
            )

        # Calculate percentage
        pct_outliers = (len(outlier_vals) / len(data_original)) * 100

        # Update layout with custom kwargs
        layout_kwargs = {
            "title": f"{col} Time Series with Outliers ({pct_outliers:.2f}%)",
            "xaxis_title": "Time",
            "yaxis_title": "Value",
            "template": "plotly_white",
            "legend": dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        }
        layout_kwargs.update(kwargs)
        fig.update_layout(**layout_kwargs)
        fig.show()
