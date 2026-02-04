"""Interactive time series visualization using Plotly."""

from typing import Dict, List, Optional, Any

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def visualize_ts_plotly(
    dataframes: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    title_suffix: str = "",
    figsize: tuple[int, int] = (1000, 500),
    template: str = "plotly_white",
    colors: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """Visualize multiple time series datasets interactively with Plotly.

    Creates interactive Plotly scatter plots for specified columns across multiple
    datasets (e.g., train, validation, test splits). Each dataset is displayed as
    a separate line with a unique color and name in the legend.

    Args:
        dataframes: Dictionary mapping dataset names to pandas DataFrames with datetime
            index. Example: {'Train': df_train, 'Validation': df_val, 'Test': df_test}
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        title_suffix: Suffix to append to the column name in the title. Useful for
            adding units or descriptions. Default: "".
        figsize: Figure size as (width, height) in pixels. Default: (1000, 500).
        template: Plotly template name for styling. Options include 'plotly_white',
            'plotly_dark', 'plotly', 'ggplot2', etc. Default: 'plotly_white'.
        colors: Dictionary mapping dataset names to colors. If None, uses Plotly
            default colors. Example: {'Train': 'blue', 'Validation': 'orange'}.
            Default: None.
        **kwargs: Additional keyword arguments passed to go.Scatter() (e.g.,
            mode='lines+markers', line=dict(dash='dash')).

    Returns:
        None. Displays Plotly figures.

    Raises:
        ValueError: If dataframes dict is empty, contains no columns, or if
            specified columns don't exist in all dataframes.
        ImportError: If plotly is not installed.
        TypeError: If dataframes parameter is not a dictionary.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.time_series_visualization import visualize_ts_plotly
        >>>
        >>> # Create sample time series data
        >>> np.random.seed(42)
        >>> dates_train = pd.date_range('2024-01-01', periods=100, freq='h')
        >>> dates_val = pd.date_range('2024-05-11', periods=50, freq='h')
        >>> dates_test = pd.date_range('2024-07-01', periods=30, freq='h')
        >>>
        >>> data_train = pd.DataFrame({
        ...     'temperature': np.random.normal(20, 5, 100),
        ...     'humidity': np.random.normal(60, 10, 100)
        ... }, index=dates_train)
        >>>
        >>> data_val = pd.DataFrame({
        ...     'temperature': np.random.normal(22, 5, 50),
        ...     'humidity': np.random.normal(55, 10, 50)
        ... }, index=dates_val)
        >>>
        >>> data_test = pd.DataFrame({
        ...     'temperature': np.random.normal(25, 5, 30),
        ...     'humidity': np.random.normal(50, 10, 30)
        ... }, index=dates_test)
        >>>
        >>> # Visualize all datasets
        >>> dataframes = {
        ...     'Train': data_train,
        ...     'Validation': data_val,
        ...     'Test': data_test
        ... }
        >>> visualize_ts_plotly(dataframes)

        Single dataset example:

        >>> # Visualize single dataset
        >>> dataframes = {'Data': data_train}
        >>> visualize_ts_plotly(dataframes, columns=['temperature'])

        Custom styling:

        >>> visualize_ts_plotly(
        ...     dataframes,
        ...     columns=['temperature'],
        ...     template='plotly_dark',
        ...     colors={'Train': 'blue', 'Validation': 'green', 'Test': 'red'},
        ...     mode='lines+markers'
        ... )
    """
    if go is None:
        raise ImportError(
            "plotly is required for this function. " "Install with: pip install plotly"
        )

    if not isinstance(dataframes, dict):
        raise TypeError("dataframes parameter must be a dictionary")

    if not dataframes:
        raise ValueError("dataframes dictionary is empty")

    # Validate all dataframes have data
    for name, df in dataframes.items():
        if df.empty:
            raise ValueError(f"DataFrame '{name}' is empty")
        if len(df.columns) == 0:
            raise ValueError(f"DataFrame '{name}' contains no columns")

    # Determine columns to plot
    all_columns = set()
    for df in dataframes.values():
        all_columns.update(df.columns)

    if not all_columns:
        raise ValueError("No columns found in any dataframe")

    columns_to_plot = columns if columns is not None else sorted(list(all_columns))

    # Validate columns exist in all dataframes
    for col in columns_to_plot:
        for name, df in dataframes.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe '{name}'")

    # Default colors if not provided
    if colors is None:
        # Use a set of distinct colors
        default_colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf",  # cyan
        ]
        colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(dataframes.keys())
        }

    # Create figures for each column
    for col in columns_to_plot:
        fig = go.Figure()

        # Add trace for each dataset
        for dataset_name, df in dataframes.items():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=dataset_name,
                    line=dict(color=colors[dataset_name]),
                    **kwargs,
                )
            )

        # Create title
        title = col
        if title_suffix:
            title = f"{col} {title_suffix}"

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=col,
            width=figsize[0],
            height=figsize[1],
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
        )

        fig.show()


def visualize_ts_comparison(
    dataframes: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    title_suffix: str = "",
    figsize: tuple[int, int] = (1000, 500),
    template: str = "plotly_white",
    colors: Optional[Dict[str, str]] = None,
    show_mean: bool = False,
    **kwargs: Any,
) -> None:
    """Visualize time series with optional statistical overlays.

    Similar to visualize_ts_plotly but adds options for statistical overlays
    like mean values across all datasets.

    Args:
        dataframes: Dictionary mapping dataset names to pandas DataFrames.
        columns: List of column names to visualize. If None, all columns are used.
            Default: None.
        title_suffix: Suffix to append to column names. Default: "".
        figsize: Figure size as (width, height) in pixels. Default: (1000, 500).
        template: Plotly template. Default: 'plotly_white'.
        colors: Dictionary mapping dataset names to colors. Default: None.
        show_mean: If True, overlay the mean of all datasets. Default: False.
        **kwargs: Additional keyword arguments for go.Scatter().

    Returns:
        None. Displays Plotly figures.

    Raises:
        ValueError: If dataframes is empty.
        ImportError: If plotly is not installed.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2.preprocessing.time_series_visualization import visualize_ts_comparison
        >>>
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> dates1 = pd.date_range('2024-01-01', periods=100, freq='h')
        >>> dates2 = pd.date_range('2024-05-11', periods=100, freq='h')
        >>>
        >>> df1 = pd.DataFrame({
        ...     'temperature': np.random.normal(20, 5, 100)
        ... }, index=dates1)
        >>>
        >>> df2 = pd.DataFrame({
        ...     'temperature': np.random.normal(22, 5, 100)
        ... }, index=dates2)
        >>>
        >>> # Compare with mean overlay
        >>> visualize_ts_comparison(
        ...     {'Dataset1': df1, 'Dataset2': df2},
        ...     show_mean=True
        ... )
    """
    if go is None:
        raise ImportError(
            "plotly is required for this function. " "Install with: pip install plotly"
        )

    if not dataframes:
        raise ValueError("dataframes dictionary is empty")

    # First visualize normally
    visualize_ts_plotly(
        dataframes,
        columns=columns,
        title_suffix=title_suffix,
        figsize=figsize,
        template=template,
        colors=colors,
        **kwargs,
    )

    # If show_mean, create additional mean plot
    if show_mean:
        # Determine columns to plot
        all_columns = set()
        for df in dataframes.values():
            all_columns.update(df.columns)

        columns_to_plot = columns if columns is not None else sorted(list(all_columns))

        for col in columns_to_plot:
            fig = go.Figure()

            # Add individual traces
            if colors is None:
                default_colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                ]
                colors_dict = {
                    name: default_colors[i % len(default_colors)]
                    for i, name in enumerate(dataframes.keys())
                }
            else:
                colors_dict = colors

            for dataset_name, df in dataframes.items():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode="lines",
                        name=dataset_name,
                        line=dict(color=colors_dict[dataset_name], width=1),
                        opacity=0.5,
                        **kwargs,
                    )
                )

            # Calculate and add mean
            # Align all dataframes by index and compute mean
            aligned_dfs = [
                dataframes[name][[col]].rename(columns={col: name})
                for name in dataframes.keys()
            ]
            combined = pd.concat(aligned_dfs, axis=1)
            mean_values = combined.mean(axis=1)

            fig.add_trace(
                go.Scatter(
                    x=mean_values.index,
                    y=mean_values,
                    mode="lines",
                    name="Mean",
                    line=dict(color="black", width=3, dash="dash"),
                )
            )

            title = f"{col} (with mean){title_suffix}"

            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=col,
                width=figsize[0],
                height=figsize[1],
                template=template,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                hovermode="x unified",
            )

            fig.show()
