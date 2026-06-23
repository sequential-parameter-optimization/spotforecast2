# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Default hyperparameter search spaces and per-task constants.

A neutral module imported by ``spotforecast2.multitask.strategies``, the
user-facing shim modules ``spotforecast2.multitask.optuna`` /
``spotforecast2.multitask.spotoptim``, and ``spotforecast2.models``'s
``ForecasterRecursiveModelFull.tune``.  Keeping the defaults here means
no shim has to import a strategy back, and no strategy class has to
import a shim — eliminating the cyclic-import warning CodeQL raised
against the lazy-import workaround.

This module is the single canonical home for:

- ``LAGS_CONSIDER`` / ``WINDOW_FEATURES`` (constants).
- ``_default_optuna_search_space`` / ``_default_spotoptim_search_space``
  (used by ``OptunaStrategy`` / ``SpotOptimStrategy`` in the multitask
  pipeline).
- ``search_space_lgbm`` / ``search_space_xgb`` / ``SEARCH_SPACES``
  (used by ``ForecasterRecursiveModelFull.tune``).
"""

from typing import Any, Dict, List

from spotforecast2_safe.preprocessing import RollingFeatures

#: Candidate lag values used by ``search_space_lgbm`` / ``search_space_xgb``
#: (consumed by ``ForecasterRecursiveModelFull.tune``).
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
    """Built-in Optuna search space for LightGBM.

    Estimator hyperparameters use the ``estimator__`` prefix so that
    ``ForecasterRecursive.set_params(**sample)`` forwards them to the wrapped
    LGBMRegressor instead of silently setting attributes on the wrapper.
    ``lags`` is consumed separately via ``set_lags`` and therefore stays bare.
    """
    return {
        "estimator__num_leaves": trial.suggest_int("estimator__num_leaves", 8, 256),
        "estimator__max_depth": trial.suggest_int("estimator__max_depth", 3, 16),
        "estimator__learning_rate": trial.suggest_float(
            "estimator__learning_rate", 0.001, 0.2, log=True
        ),
        "estimator__n_estimators": trial.suggest_int(
            "estimator__n_estimators", 50, 1000, log=True
        ),
        "estimator__bagging_fraction": trial.suggest_float(
            "estimator__bagging_fraction", 0.5, 1
        ),
        "estimator__feature_fraction": trial.suggest_float(
            "estimator__feature_fraction", 0.5, 1
        ),
        "estimator__reg_alpha": trial.suggest_float("estimator__reg_alpha", 0.01, 100),
        "estimator__reg_lambda": trial.suggest_float(
            "estimator__reg_lambda", 0.01, 100
        ),
        "lags": trial.suggest_categorical(
            "lags",
            [24, 48, [1, 2, 24], [1, 2, 24, 48], [1, 2, 23, 24, 47, 48]],
        ),
    }


def _default_spotoptim_search_space() -> Dict[str, Any]:
    """Built-in SpotOptim search space for LightGBM.

    Estimator hyperparameters carry the ``estimator__`` prefix; see the docstring
    of `_default_optuna_search_space` for the rationale.
    """
    return {
        "estimator__num_leaves": (8, 256),
        "estimator__max_depth": (3, 16),
        "estimator__learning_rate": (0.0001, 0.1, "log10"),
        "estimator__n_estimators": (10, 1000, "log10"),
        "estimator__bagging_fraction": (0.5, 1.0),
        "estimator__feature_fraction": (0.5, 1.0),
        "estimator__reg_alpha": (0.01, 100.0),
        "estimator__reg_lambda": (0.01, 100.0),
        "lags": [
            "[1, 2, 3, 11, 12, 22, 23, 24, 47, 48, 167, 168]",
            "48",
            "24",
            "[1, 2, 24, 48]",
            "[1, 2, 23, 24, 47, 48]",
            "[1, 2, 11, 12, 23, 24, 167, 168]",
        ],
    }


def search_space_lgbm(trial: Any) -> Dict[str, Any]:
    """Optuna search space for LightGBM hyperparameters.

    Consumed by ``ForecasterRecursiveModelFull.tune`` via the
    ``SEARCH_SPACES`` registry below. Estimator keys use the ``estimator__``
    prefix; see `_default_optuna_search_space` for the rationale.

    Args:
        trial: An ``optuna.trial.Trial`` instance.

    Returns:
        Mapping of hyperparameter name to suggested value for the current
        trial.

    Examples:
        ```{python}
        import optuna
        from spotforecast2.multitask.search_spaces import search_space_lgbm

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        trial = study.ask()
        params = search_space_lgbm(trial)
        print("Keys:", list(params.keys()))
        assert "estimator__num_leaves" in params
        assert "lags" in params
        assert isinstance(params["estimator__learning_rate"], float)
        ```
    """
    return {
        "estimator__num_leaves": trial.suggest_int("estimator__num_leaves", 8, 256),
        "estimator__max_depth": trial.suggest_int("estimator__max_depth", 3, 16),
        "estimator__learning_rate": trial.suggest_float(
            "estimator__learning_rate", 0.001, 0.2, log=True
        ),
        "estimator__n_estimators": trial.suggest_int(
            "estimator__n_estimators", 50, 1000, log=True
        ),
        "estimator__bagging_fraction": trial.suggest_float(
            "estimator__bagging_fraction", 0.5, 1
        ),
        "estimator__feature_fraction": trial.suggest_float(
            "estimator__feature_fraction", 0.5, 1
        ),
        "estimator__reg_alpha": trial.suggest_float("estimator__reg_alpha", 0.01, 100),
        "estimator__reg_lambda": trial.suggest_float(
            "estimator__reg_lambda", 0.01, 100
        ),
        "lags": trial.suggest_categorical("lags", LAGS_CONSIDER),
    }


def search_space_xgb(trial: Any) -> Dict[str, Any]:
    """Optuna search space for XGBoost hyperparameters.

    Consumed by ``ForecasterRecursiveModelFull.tune`` via the
    ``SEARCH_SPACES`` registry below. Estimator keys use the ``estimator__``
    prefix; see `_default_optuna_search_space` for the rationale.

    Args:
        trial: An ``optuna.trial.Trial`` instance.

    Returns:
        Mapping of hyperparameter name to suggested value for the current
        trial.

    Examples:
        ```{python}
        import optuna
        from spotforecast2.multitask.search_spaces import search_space_xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        trial = study.ask()
        params = search_space_xgb(trial)
        print("Keys:", list(params.keys()))
        assert "estimator__max_depth" in params
        assert "lags" in params
        assert isinstance(params["estimator__learning_rate"], float)
        ```
    """
    return {
        "estimator__max_depth": trial.suggest_int("estimator__max_depth", 2, 10),
        "estimator__learning_rate": trial.suggest_float(
            "estimator__learning_rate", 0.001, 0.2, log=True
        ),
        "estimator__subsample": trial.suggest_float("estimator__subsample", 0.6, 1),
        "estimator__colsample_bytree": trial.suggest_float(
            "estimator__colsample_bytree", 0.6, 1
        ),
        "estimator__min_child_weight": trial.suggest_int(
            "estimator__min_child_weight", 1, 8
        ),
        "estimator__n_estimators": trial.suggest_int(
            "estimator__n_estimators", 50, 600, step=50
        ),
        "estimator__alpha": trial.suggest_float("estimator__alpha", 0.0, 0.5),
        "estimator__lambda": trial.suggest_float("estimator__lambda", 0.0, 0.5),
        "lags": trial.suggest_categorical("lags", LAGS_CONSIDER),
    }


def search_space_catboost(trial: Any) -> Dict[str, Any]:
    """Optuna search space for CatBoost hyperparameters.

    Consumed by ``ForecasterRecursiveModelFull.tune`` via the
    ``SEARCH_SPACES`` registry below. Estimator keys use the ``estimator__``
    prefix; see `_default_optuna_search_space` for the rationale.

    The space is deliberately ``bootstrap_type``-agnostic: it omits
    ``subsample`` / ``colsample`` (valid only for specific CatBoost bootstrap
    types) to avoid invalid-combination errors, and instead tunes CatBoost's
    native ``depth`` / ``l2_leaf_reg`` / ``random_strength`` /
    ``bagging_temperature`` levers.

    Args:
        trial: An ``optuna.trial.Trial`` instance.

    Returns:
        Mapping of hyperparameter name to suggested value for the current
        trial.

    Examples:
        ```{python}
        import optuna
        from spotforecast2.multitask.search_spaces import search_space_catboost

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        trial = study.ask()
        params = search_space_catboost(trial)
        print("Keys:", list(params.keys()))
        assert "estimator__depth" in params
        assert "lags" in params
        assert isinstance(params["estimator__learning_rate"], float)
        ```
    """
    return {
        "estimator__depth": trial.suggest_int("estimator__depth", 4, 10),
        "estimator__learning_rate": trial.suggest_float(
            "estimator__learning_rate", 0.001, 0.2, log=True
        ),
        "estimator__l2_leaf_reg": trial.suggest_float(
            "estimator__l2_leaf_reg", 1.0, 10.0, log=True
        ),
        "estimator__n_estimators": trial.suggest_int(
            "estimator__n_estimators", 50, 600, step=50
        ),
        "estimator__random_strength": trial.suggest_float(
            "estimator__random_strength", 0.0, 2.0
        ),
        "estimator__bagging_temperature": trial.suggest_float(
            "estimator__bagging_temperature", 0.0, 1.0
        ),
        "lags": trial.suggest_categorical("lags", LAGS_CONSIDER),
    }


#: Registry mapping model name to its Optuna search-space function.
#: Consumed by ``ForecasterRecursiveModelFull.tune`` to pick the right
#: search space based on ``self.name``.
SEARCH_SPACES: Dict[str, Any] = {
    "lgbm": search_space_lgbm,
    "xgb": search_space_xgb,
    "catboost": search_space_catboost,
}
