# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Spotforecast2: Time Series Forecasting Package

Main package for time series forecasting with safety-critical system support.
The primary forecasters are imported from spotforecast2_safe for safety-critical operations.
Additional utilities and convenience classes are provided here.
"""

__version__ = "0.1.0"

from . import forecaster
from . import manager
from . import model_selection
from . import preprocessing
from . import stats
from . import utils
from . import weather
from spotforecast2_safe.manager.configurator import ConfigEntsoe

Config = ConfigEntsoe

__all__ = [
    "Config",
    "ConfigEntsoe",
    "forecaster",
    "manager",
    "model_selection",
    "preprocessing",
    "stats",
    "utils",
    "weather",
]


def hello() -> str:
    return "Hello from spotforecast2!"
