# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""Metrics for evaluating forecasting models.

This module provides various metric functions for evaluating forecasting performance,
including custom metrics like MASE, RMSSE, and probabilistic metrics like CRPS.
These metrics are imported from the spotforecast2_safe package.
"""

from spotforecast2_safe.forecaster.metrics import (
    _get_metric,
    add_y_train_argument,
    mean_absolute_scaled_error,
    root_mean_squared_scaled_error,
    crps_from_predictions,
    crps_from_quantiles,
    calculate_coverage,
    create_mean_pinball_loss,
    symmetric_mean_absolute_percentage_error,
)

__all__ = [
    "_get_metric",
    "add_y_train_argument",
    "mean_absolute_scaled_error",
    "root_mean_squared_scaled_error",
    "crps_from_predictions",
    "crps_from_quantiles",
    "calculate_coverage",
    "create_mean_pinball_loss",
    "symmetric_mean_absolute_percentage_error",
]
