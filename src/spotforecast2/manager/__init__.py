# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Manager module for plotting, visualization, and model management.

This module provides utilities for generating interactive prediction plots
and full-featured forecasting model classes with Bayesian tuning and SHAP.
"""

from spotforecast2.manager.multitask import (
    BaseTask,
    CleanTask,
    LazyTask,
    MultiTask,
    OptunaTask,
    PredictTask,
    SpotOptimTask,
    agg_predictor,
    run,
)
from spotforecast2.manager.plotter import (
    make_plot,
    plot_actual_vs_predicted,
    plot_with_outliers,
    PredictionFigure,
)
from spotforecast2.manager.models import (
    ForecasterRecursiveModelFull,
    ForecasterRecursiveLGBMFull,
    ForecasterRecursiveXGBFull,
)

# spotforecast2.manager.models is now a package; the above import works
# transparently via spotforecast2/manager/models/__init__.py
from spotforecast2_safe.manager.configurator import ConfigEntsoe

__all__ = [
    "BaseTask",
    "CleanTask",
    "ConfigEntsoe",
    "ForecasterRecursiveLGBMFull",
    "ForecasterRecursiveModelFull",
    "ForecasterRecursiveXGBFull",
    "LazyTask",
    "MultiTask",
    "OptunaTask",
    "PredictTask",
    "SpotOptimTask",
    "agg_predictor",
    "run",
    "make_plot",
    "plot_actual_vs_predicted",
    "plot_with_outliers",
    "PredictionFigure",
]
