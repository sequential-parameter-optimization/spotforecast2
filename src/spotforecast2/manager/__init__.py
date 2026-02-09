# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Manager module for plotting and visualization.

This module provides utilities for generating interactive prediction plots
and visualizations for time series forecasting models.
"""

from spotforecast2.manager.plotter import (
    make_plot,
    plot_actual_vs_predicted,
    PredictionFigure,
)
from spotforecast2.manager.configurator import ConfigEntsoe

__all__ = [
    "ConfigEntsoe",
    "make_plot",
    "plot_actual_vs_predicted",
    "PredictionFigure",
]
