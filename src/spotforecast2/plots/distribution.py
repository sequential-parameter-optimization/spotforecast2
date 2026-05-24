# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plotly histogram / KDE distribution plots for time series and DataFrames.

Renders as Plotly (not matplotlib) for consistency with the rest of
``spotforecast2.plots``.  Heuristic: a series with fewer than
``unique_threshold`` distinct values is shown as a histogram, otherwise
as a kernel-density estimate.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


def plot_distribution(
    series: pd.Series,
    *,
    unique_threshold: int = 100,
    template: str = "plotly_white",
    title: Optional[str] = None,
) -> go.Figure:
    """Plot the distribution of a single series.

    Args:
        series: Series of numeric values.
        unique_threshold: If the series has fewer distinct non-null
            values than this threshold, render as a histogram; otherwise
            render as a KDE.
        template: Plotly theme name.
        title: Optional figure title.  Defaults to the series ``name``.

    Returns:
        A `plotly.graph_objects.Figure`.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd

        from spotforecast2.plots.distribution import plot_distribution

        rng = np.random.default_rng(0)
        y = pd.Series(rng.standard_normal(500))
        fig = plot_distribution(y)
        print(fig.data[0].type)
        ```
    """
    values = series.dropna().to_numpy()
    fig = go.Figure()

    if values.size == 0:
        fig.update_layout(template=template, title=title or series.name)
        return fig

    if pd.Series(values).nunique() < unique_threshold:
        fig.add_trace(go.Histogram(x=values, nbinsx=30, name=series.name or "value"))
    else:
        kde = gaussian_kde(values)
        x = np.linspace(values.min(), values.max(), 200)
        fig.add_trace(
            go.Scatter(x=x, y=kde(x), mode="lines", name=series.name or "value")
        )

    fig.update_layout(template=template, title=title or series.name)
    return fig


def plot_distributions(
    df: pd.DataFrame,
    *,
    unique_threshold: int = 100,
    template: str = "plotly_white",
    n_cols: int = 2,
) -> go.Figure:
    """Plot per-column distributions of a DataFrame in a subplot grid.

    Args:
        df: DataFrame whose columns to plot.
        unique_threshold: Threshold passed to `plot_distribution()`.
        template: Plotly theme name.
        n_cols: Number of columns in the subplot grid.

    Returns:
        A `plotly.graph_objects.Figure` with one subplot per column.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd

        from spotforecast2.plots.distribution import plot_distributions

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "a": rng.standard_normal(200),
                "b": rng.integers(0, 5, 200),
            }
        )
        fig = plot_distributions(df)
        print(len(fig.data))
        ```
    """
    cols = list(df.columns)
    n_features = len(cols)
    n_rows = int(np.ceil(n_features / n_cols)) if n_features else 1

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols)
    for i, col in enumerate(cols):
        sub = plot_distribution(
            df[col], unique_threshold=unique_threshold, template=template
        )
        row = i // n_cols + 1
        column = i % n_cols + 1
        for trace in sub.data:
            fig.add_trace(trace, row=row, col=column)

    fig.update_layout(template=template, showlegend=False)
    return fig
