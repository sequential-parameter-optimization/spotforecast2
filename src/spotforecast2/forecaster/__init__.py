# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Forecaster module for spotforecast2.

This module provides forecasting classes and utilities for time series prediction.
The primary forecasters are provided by the spotforecast2_safe package for safety-critical
system operations. This module includes additional utilities.
"""

from . import metrics
from . import utils
from . import recursive

__all__ = [
    "metrics",
    "utils",
    "recursive",
]
