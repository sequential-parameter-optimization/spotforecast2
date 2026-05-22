# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Default hyperparameter search spaces and per-task constants.

A neutral module imported by both `spotforecast2.multitask.strategies`
and the user-facing shim modules `spotforecast2.multitask.optuna` /
`spotforecast2.multitask.spotoptim`.  Keeping the defaults here means
no shim has to import the strategy module back, and no strategy class has
to import a shim — eliminating the cyclic-import warning CodeQL raised
against the lazy-import workaround.

ADR-002 Step 3 additionally relocated ``LAGS_CONSIDER`` and
``WINDOW_FEATURES`` (the legacy ENTSO-E trainer constants) here so that
`spotforecast2.trainer.trainer_full` can be removed in Step 5 without
breaking anything that still imports those names.
"""

from typing import Any, Dict, List

from spotforecast2_safe.preprocessing import RollingFeatures


#: Candidate lag values used by the default Optuna / SpotOptim search
#: spaces and historically by ``spotforecast2.trainer.trainer_full``.
LAGS_CONSIDER: List[int] = list(range(1, 24))

#: Default rolling-window features matching the original chag25a
#: configuration.  Each entry is a separate ``RollingFeatures`` instance to
#: avoid duplicate-name collisions in spotforecast2-safe's
#: ``initialize_window_features``.
WINDOW_FEATURES = [
    RollingFeatures(stats="mean", window_sizes=24),
    RollingFeatures(stats="mean", window_sizes=24 * 7),
    RollingFeatures(stats="mean", window_sizes=24 * 30),
    RollingFeatures(stats="min", window_sizes=24),
    RollingFeatures(stats="max", window_sizes=24),
]


def _default_optuna_search_space(trial: Any) -> Dict[str, Any]:
    """Built-in Optuna search space for LightGBM."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100),
        "lags": trial.suggest_categorical(
            "lags",
            [24, 48, [1, 2, 24], [1, 2, 24, 48], [1, 2, 23, 24, 47, 48]],
        ),
    }


def _default_spotoptim_search_space() -> Dict[str, Any]:
    """Built-in SpotOptim search space for LightGBM."""
    return {
        "num_leaves": (8, 256),
        "max_depth": (3, 16),
        "learning_rate": (0.0001, 0.1, "log10"),
        "n_estimators": (10, 1000, "log10"),
        "bagging_fraction": (0.5, 1.0),
        "feature_fraction": (0.5, 1.0),
        "reg_alpha": (0.01, 100.0),
        "reg_lambda": (0.01, 100.0),
        "lags": [
            "[1, 2, 3, 11, 12, 22, 23, 24, 47, 48, 167, 168]",
            "48",
            "24",
            "[1, 2, 24, 48]",
            "[1, 2, 23, 24, 47, 48]",
            "[1, 2, 11, 12, 23, 24, 167, 168]",
        ],
    }
