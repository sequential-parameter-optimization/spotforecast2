"""
Spotforecast2: Time Series Forecasting Package

Main package for time series forecasting with safety-critical system support.
The primary forecasters are imported from spotforecast2_safe for safety-critical operations.
Additional utilities and convenience classes are provided here.
"""

__version__ = "0.1.0"

from . import forecaster
from . import model_selection
from . import preprocessing
from . import processing
from . import stats
from . import utils
from . import weather

__all__ = [
    "forecaster",
    "model_selection",
    "preprocessing",
    "processing",
    "stats",
    "utils",
    "weather",
]


def hello() -> str:
    return "Hello from spotforecast2!"
