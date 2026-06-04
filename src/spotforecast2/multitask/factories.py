# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Forecaster factories for the multitask pipeline.

Re-exports ``default_lgbm_forecaster_factory`` from the safe package so that
importers of ``spotforecast2.multitask.factories`` remain unaffected after the
pipeline collapse.
"""

from spotforecast2_safe.multitask.factories import (  # noqa: F401  (re-exported)
    default_lgbm_forecaster_factory,
)

__all__ = ["default_lgbm_forecaster_factory"]
