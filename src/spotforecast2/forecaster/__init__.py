"""
Forecaster module for spotforecast2.

This module provides forecasting classes and utilities for time series prediction.
The primary forecasters are provided by the spotforecast2_safe package for safety-critical
system operations. This module includes additional utilities.
"""

from . import metrics
from . import utils

__all__ = [
    "metrics",
    "utils",
]
