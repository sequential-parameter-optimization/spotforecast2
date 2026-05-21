# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Forecaster module for spotforecast2.

This module exposes the local forecaster utilities and the `recursive`
subpackage. Symbols from `spotforecast2_safe` (e.g. `metrics`,
`ForecasterRecursive`) are not re-exported here; import them directly from
their fully qualified `spotforecast2_safe.*` paths.
"""

from . import utils
from . import recursive

__all__ = [
    "utils",
    "recursive",
]
