# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Warning-style configuration for spotforecast2.

Importing this subpackage installs the skforecast-style warning handler
via the side effect at the bottom of `exceptions.py`. Warning and error
classes themselves live in `spotforecast2_safe.exceptions`.
"""

from . import exceptions  # noqa: F401  imported for warnings-style side effect
