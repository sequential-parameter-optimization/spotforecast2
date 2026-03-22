# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Multi-target forecasting pipeline orchestrator.

This package provides a class hierarchy for multi-target time-series
forecasting pipelines:

- :class:`BaseTask` — shared data-preparation and helper logic.
- :class:`LazyTask` — Task 1: lazy fitting with default parameters.
- :class:`TrainTask` — Task 2: explicit pre-training and serialisation.
- :class:`OptunaTask` — Task 3: Optuna Bayesian hyperparameter tuning.
- :class:`SpotOptimTask` — Task 4: SpotOptim surrogate-model tuning.
- :class:`MultiTask` — backward-compatible dispatcher that selects one
  of the four tasks via a ``TASK`` parameter.

Public API
----------
All classes are importable directly from ``spotforecast2.manager.multitask``.
"""

from spotforecast2.manager.multitask.base import BaseTask, agg_predictor
from spotforecast2.manager.multitask.lazy import LazyTask
from spotforecast2.manager.multitask.train import TrainTask
from spotforecast2.manager.multitask.optuna import OptunaTask
from spotforecast2.manager.multitask.spotoptim import SpotOptimTask
from spotforecast2.manager.multitask.multi import MultiTask

__all__ = [
    "BaseTask",
    "LazyTask",
    "TrainTask",
    "OptunaTask",
    "SpotOptimTask",
    "MultiTask",
    "agg_predictor",
]
