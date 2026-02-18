# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Manager module for plotting, visualization, and model management.

This module provides utilities for generating interactive prediction plots
and full-featured forecasting model classes with Bayesian tuning and SHAP.
"""

from spotforecast2.manager.plotter import (
    make_plot,
    plot_actual_vs_predicted,
    PredictionFigure,
)
from spotforecast2.manager.models import (
    ForecasterRecursiveModelFull,
    ForecasterRecursiveLGBMFull,
    ForecasterRecursiveXGBFull,
)
from spotforecast2_safe.manager.configurator import ConfigEntsoe

__all__ = [
    "ConfigEntsoe",
    "ForecasterRecursiveLGBMFull",
    "ForecasterRecursiveModelFull",
    "ForecasterRecursiveXGBFull",
    "make_plot",
    "plot_actual_vs_predicted",
    "PredictionFigure",
]
