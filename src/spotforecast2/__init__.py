# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Spotforecast2: Time Series Forecasting Package

Main package for time series forecasting with safety-critical system support.
The primary forecasters are imported from spotforecast2_safe for safety-critical operations.
Additional utilities and convenience classes are provided here.
"""

from . import warnings  # noqa: F401  imported for warnings-style side effect
from . import model_selection
from . import models
from . import multitask
from . import plots
from . import stats
from . import trainer

__all__ = [
    "model_selection",
    "models",
    "multitask",
    "plots",
    "stats",
    "trainer",
    "warnings",
]

