# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Statistical functions for time series analysis.

Pure compute primitives live in :mod:`spotforecast2_safe.stats` and are
re-exported here for backward-compatible imports.  Visualization
wrappers stay in :mod:`spotforecast2.plots`.
"""

from spotforecast2_safe.stats import (
    augmented_dickey_fuller,
    compute_periodogram,
)

from .autocorrelation import calculate_lag_autocorrelation

__all__ = [
    "augmented_dickey_fuller",
    "calculate_lag_autocorrelation",
    "compute_periodogram",
]
