# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plotly visualization of a spectral periodogram.

Consumes the output of `compute_periodogram()` from
`spotforecast2_safe.stats.spectral`.
"""

from typing import Mapping, Optional, Union

import pandas as pd
import plotly.graph_objects as go
from spotforecast2_safe.stats.spectral import PeriodogramResult

# Default named period ticks, in samples (matches chag25a's hourly convention).
_DEFAULT_PERIOD_TICKS: Mapping[str, float] = {
    "4 years": 4 * 12 * 30 * 24,
    "2 years": 2 * 12 * 30 * 24,
    "year": 12 * 30 * 24,
    "6 months": 6 * 30 * 24,
    "3 months": 3 * 30 * 24,
    "month": 30 * 24,
    "2 weeks": 2 * 7 * 24,
    "week": 7 * 24,
    "day": 24,
    "12 hours": 12,
    "2 hours": 2,
}


def plot_periodogram(
    result: Union[PeriodogramResult, pd.Series],
    *,
    period_ticks: Optional[Mapping[str, float]] = None,
    template: str = "plotly_white",
    title: str = "Periodogram",
) -> go.Figure:
    """Render a log-x periodogram with named period ticks.

    Args:
        result: Output of `compute_periodogram()` (preferred), or a
            ``pd.Series`` of power indexed by frequency.
        period_ticks: Mapping from human-readable label to period length
            (in samples).  Each tick is placed at ``1 / period`` on the
            frequency axis.  Defaults to a sensible set for hourly data.
        template: Plotly theme name.
        title: Figure title.

    Returns:
        A `plotly.graph_objects.Figure`.

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd

        from spotforecast2_safe.stats.spectral import compute_periodogram
        from spotforecast2.plots.spectral import plot_periodogram

        t = np.arange(512)
        y = pd.Series(np.sin(2 * np.pi * t / 24))
        fig = plot_periodogram(compute_periodogram(y))
        print(fig.layout.xaxis.type)
        ```
    """
    if isinstance(result, PeriodogramResult):
        spectrum = result.spectrum
        freqs = spectrum.index.to_numpy()
        power = spectrum["power"].to_numpy()
    else:
        freqs = result.index.to_numpy()
        power = result.to_numpy()

    ticks = period_ticks if period_ticks is not None else _DEFAULT_PERIOD_TICKS

    fig = go.Figure()
    # Skip the DC bin on a log axis.
    mask = freqs > 0
    fig.add_trace(
        go.Scatter(
            x=freqs[mask],
            y=power[mask],
            mode="lines",
            line=dict(shape="hv"),
            name="power",
        )
    )

    tick_vals = [1.0 / period for period in ticks.values()]
    tick_text = [f"Every {label}" for label in ticks.keys()]

    fig.update_layout(
        template=template,
        title=title,
        xaxis=dict(
            type="log",
            title="Frequency (cycles per sample)",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-90,
        ),
        yaxis=dict(title="Variance"),
    )
    return fig


__all__ = ["plot_periodogram"]
