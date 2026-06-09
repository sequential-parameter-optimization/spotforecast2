# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Training strategies for the multitask pipeline.

Re-exports ``TrainingStrategy``, ``LazyStrategy``, and ``DefaultsStrategy``
from the safe package (tuning-free strategies).  Defines ``OptunaStrategy``
and ``SpotOptimStrategy`` here because they depend on ``spotforecast2``-only
packages (``optuna``, ``spotoptim``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import pandas as pd

# Re-export tuning-free strategies from the safe package so importers of
# ``spotforecast2.multitask.strategies`` continue to find all four names.
from spotforecast2_safe.multitask.strategies import (  # noqa: F401  (re-exported)
    DefaultsStrategy,
    LazyStrategy,
    TrainingStrategy,
)

from spotforecast2.multitask.search_spaces import (
    _default_optuna_search_space,
    _default_spotoptim_search_space,
)

__all__ = [
    "TrainingStrategy",
    "LazyStrategy",
    "DefaultsStrategy",
    "OptunaStrategy",
    "SpotOptimStrategy",
]


class OptunaStrategy:
    """Approach 3 — Optuna Bayesian tuning, then apply best params.

    Examples:
        ```{python}
        from spotforecast2.multitask.strategies import OptunaStrategy

        strategy = OptunaStrategy()
        print(f"name: {strategy.name}")
        assert strategy.name == "optuna"
        assert strategy.search_space is None

        # Custom search space can be injected at construction time.
        def my_space(trial):
            return {"estimator__n_estimators": trial.suggest_int("estimator__n_estimators", 10, 50)}

        custom = OptunaStrategy(search_space=my_space)
        assert custom.search_space is my_space
        print(f"custom search_space: {custom.search_space.__name__}")
        ```
    """

    name = "optuna"

    def __init__(self, search_space: Optional[Callable] = None) -> None:
        self.search_space = search_space

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Run Optuna search and return a forecaster initialised with the best params.

        Args:
            task: A `BaseTask` (or compatible) instance that supplies ``cv_ts``,
                ``config``, ``logger``, ``save_tuning_results``, and
                ``create_forecaster``.
            target: Target column name; forwarded to ``task.create_forecaster``
                and ``task.save_tuning_results``.
            forecaster: An unfitted forecaster instance used as the search
                template.
            y_train: Training time series for the current target.
            exog_train: Optional exogenous features aligned with ``y_train``.

        Returns:
            A fresh, unfitted forecaster with ``set_params`` and ``set_lags``
            applied from the best trial found by Optuna.

        Examples:
            ```{python}
            import types, logging, warnings
            import numpy as np
            import pandas as pd
            from lightgbm import LGBMRegressor
            from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            from spotforecast2_safe.splitter import TimeSeriesFold
            from spotforecast2.multitask.strategies import OptunaStrategy

            rng = np.random.default_rng(0)
            n = 300
            idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
            y_train = pd.Series(rng.normal(0, 1, n), index=idx, name="A")

            forecaster = ForecasterRecursive(
                estimator=LGBMRegressor(n_estimators=10, verbose=-1), lags=3
            )
            cv = TimeSeriesFold(
                steps=24, initial_train_size=200, refit=False,
                gap=0, allow_incomplete_fold=True,
            )

            def tiny_space(trial):
                return {
                    "estimator__n_estimators": trial.suggest_int(
                        "estimator__n_estimators", 10, 30
                    ),
                    "lags": trial.suggest_categorical("lags", [3, 5, 6]),
                }

            cfg = types.SimpleNamespace(
                n_trials_optuna=1, random_state=0, verbose=False
            )
            task = types.SimpleNamespace(
                config=cfg,
                logger=logging.getLogger("example"),
                cv_ts=lambda y: cv,
                create_forecaster=lambda target=None: ForecasterRecursive(
                    estimator=LGBMRegressor(n_estimators=10, verbose=-1), lags=3
                ),
                save_tuning_results=lambda **kw: None,
                _show_progress=False,
            )

            strategy = OptunaStrategy(search_space=tiny_space)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tuned = strategy.prepare_forecaster(task, "A", forecaster, y_train)

            print(f"tuned type: {type(tuned).__name__}")
            assert isinstance(tuned, ForecasterRecursive)
            ```
        """
        from spotforecast2.model_selection import bayesian_search_forecaster

        search_space = self.search_space or _default_optuna_search_space
        cv = task.cv_ts(y_train)
        tuning_results, _ = bayesian_search_forecaster(
            forecaster=forecaster,
            y=y_train,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            exog=exog_train,
            n_trials=task.config.n_trials_optuna,
            random_state=task.config.random_state,
            return_best=True,
            verbose=task.config.verbose,
            show_progress=getattr(task, "_show_progress", False),
        )
        best_params = tuning_results.iloc[0].params
        best_lags = tuning_results.iloc[0].lags
        task.logger.info("  Best params: %s", best_params)
        task.logger.info("  Best lags: %s", best_lags)
        task.save_tuning_results(
            target=target,
            task_name="optuna",
            best_params=best_params,
            best_lags=best_lags,
        )
        tuned = task.create_forecaster(target=target)
        tuned.set_params(**best_params)
        if hasattr(tuned, "set_lags"):
            tuned.set_lags(best_lags)
        return tuned


class SpotOptimStrategy:
    """Approach 4 — SpotOptim surrogate-model tuning, then apply best params.

    Examples:
        ```{python}
        from spotforecast2.multitask.strategies import SpotOptimStrategy

        strategy = SpotOptimStrategy()
        print(f"name: {strategy.name}")
        assert strategy.name == "spotoptim"
        assert strategy.search_space is None

        # A custom search space narrows the grid explored by the surrogate model.
        custom_space = {
            "estimator__n_estimators": (10, 50),
            "lags": ["3", "5", "6"],
        }
        custom = SpotOptimStrategy(search_space=custom_space)
        assert custom.search_space is custom_space
        print(f"custom search_space keys: {list(custom.search_space.keys())}")
        ```
    """

    name = "spotoptim"

    def __init__(self, search_space: Optional[Dict[str, Any]] = None) -> None:
        self.search_space = search_space

    def prepare_forecaster(
        self,
        task: Any,
        target: str,
        forecaster: Any,
        y_train: pd.Series,
        exog_train: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Run SpotOptim surrogate search and return a forecaster with best params.

        Live tuning visualization: when the config carries the optional
        attributes ``tensorboard_log`` (bool), ``tensorboard_path`` (str)
        or ``tensorboard_clean`` (bool), they are forwarded to the
        SpotOptim constructor and the ongoing search can be watched with
        ``tensorboard --logdir <path>`` (defaults to ``runs/``).  Set them
        as plain attributes, e.g. ``cfg.tensorboard_log = True`` — no
        config-class change is required.  When logging is enabled and
        ``tensorboard_clean`` is unset, it defaults to ``True`` so every
        run starts with a fresh dashboard (the configured log directory
        is wiped at search start); set ``cfg.tensorboard_clean = False``
        to keep accumulating event files instead — do that, or give each
        task its own ``tensorboard_path``, when several tasks share one
        log directory.

        Args:
            task: A `BaseTask` (or compatible) instance that supplies ``cv_ts``,
                ``config``, ``logger``, ``save_tuning_results``, and
                ``create_forecaster``.  The config must expose
                ``n_trials_spotoptim``, ``n_initial_spotoptim``,
                ``random_state``, ``warm_start_lags``, and optionally
                ``lags_consider`` and the TensorBoard knobs described
                above.
            target: Target column name; forwarded to ``task.create_forecaster``
                and ``task.save_tuning_results``.
            forecaster: An unfitted forecaster instance used as the search
                template.
            y_train: Training time series for the current target.
            exog_train: Optional exogenous features aligned with ``y_train``.

        Returns:
            A fresh, unfitted forecaster with ``set_params`` and ``set_lags``
            applied from the best configuration found by the SpotOptim surrogate.

        Examples:
            ```{python}
            import types, logging, warnings
            import numpy as np
            import pandas as pd
            from lightgbm import LGBMRegressor
            from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            from spotforecast2_safe.splitter import TimeSeriesFold
            from spotforecast2.multitask.strategies import SpotOptimStrategy

            rng = np.random.default_rng(0)
            n = 300
            idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
            y_train = pd.Series(rng.normal(0, 1, n), index=idx, name="A")

            forecaster = ForecasterRecursive(
                estimator=LGBMRegressor(n_estimators=10, verbose=-1), lags=3
            )
            cv = TimeSeriesFold(
                steps=24, initial_train_size=200, refit=False,
                gap=0, allow_incomplete_fold=True,
            )

            tiny_space = {
                "estimator__n_estimators": (10, 30),
                "lags": ["3", "5", "6"],
            }

            cfg = types.SimpleNamespace(
                n_trials_spotoptim=5,
                n_initial_spotoptim=3,
                random_state=0,
                warm_start_lags=False,
            )
            task = types.SimpleNamespace(
                config=cfg,
                logger=logging.getLogger("example"),
                cv_ts=lambda y: cv,
                create_forecaster=lambda target=None: ForecasterRecursive(
                    estimator=LGBMRegressor(n_estimators=10, verbose=-1), lags=3
                ),
                save_tuning_results=lambda **kw: None,
                _show_progress=False,
            )

            strategy = SpotOptimStrategy(search_space=tiny_space)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tuned = strategy.prepare_forecaster(task, "A", forecaster, y_train)

            print(f"tuned type: {type(tuned).__name__}")
            assert isinstance(tuned, ForecasterRecursive)
            ```
        """
        from spotforecast2.model_selection import (
            build_warm_start_x0,
            spotoptim_search_forecaster,
        )

        search_space = self.search_space or _default_spotoptim_search_space()
        cv = task.cv_ts(y_train)

        # Warm start: inject ``lags_consider`` as a candidate lag set and seed
        # the optimizer's first evaluation with it.  Only dict search spaces
        # with a ``"lags"`` list are eligible; anything else falls through to a
        # normal cold-start run.
        kwargs_spotoptim: Dict[str, Any] = {}
        lags_seed = getattr(task.config, "lags_consider", None)
        if (
            getattr(task.config, "warm_start_lags", False)
            and lags_seed
            and isinstance(search_space, dict)
            and isinstance(search_space.get("lags"), list)
        ):
            seed_str = str(list(lags_seed))
            search_space = deepcopy(search_space)
            if seed_str not in search_space["lags"]:
                search_space["lags"] = [seed_str, *search_space["lags"]]
            x0 = build_warm_start_x0(search_space, forecaster, lags_seed)
            if x0 is not None:
                kwargs_spotoptim["x0"] = x0
                task.logger.info("  Warm-start lags seeded: %s", seed_str)

        # Tuning visualization: forward TensorBoard knobs to SpotOptim. Users
        # set these as plain attributes on the config (the safe-package config
        # classes carry no __slots__, so no spotforecast2_safe change is
        # needed). Unset attributes are skipped so SpotOptim's own defaults
        # (tensorboard_log=False) stay in charge.
        tb_kwargs = {
            key: getattr(task.config, key)
            for key in ("tensorboard_log", "tensorboard_path", "tensorboard_clean")
            if getattr(task.config, key, None) is not None
        }
        if tb_kwargs.get("tensorboard_log") and "tensorboard_clean" not in tb_kwargs:
            # Fresh dashboard per run: wipe the configured log directory before
            # the search starts (spotoptim >= 0.12.9 cleans tensorboard_path
            # itself). Opt out with ``cfg.tensorboard_clean = False``.
            tb_kwargs["tensorboard_clean"] = True
        if tb_kwargs:
            kwargs_spotoptim.update(tb_kwargs)
            task.logger.info("  SpotOptim TensorBoard: %s", tb_kwargs)

        tuning_results, _ = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_train,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            exog=exog_train,
            return_best=True,
            random_state=task.config.random_state,
            verbose=False,
            n_trials=task.config.n_trials_spotoptim,
            n_initial=task.config.n_initial_spotoptim,
            show_progress=getattr(task, "_show_progress", False),
            kwargs_spotoptim=kwargs_spotoptim or None,
        )
        best_params = tuning_results.iloc[0].params
        best_lags = tuning_results.iloc[0].lags
        task.logger.info("  Best params: %s", best_params)
        task.logger.info("  Best lags: %s", best_lags)
        task.save_tuning_results(
            target=target,
            task_name="spotoptim",
            best_params=best_params,
            best_lags=best_lags,
        )
        tuned = task.create_forecaster(target=target)
        tuned.set_params(**best_params)
        if hasattr(tuned, "set_lags"):
            tuned.set_lags(best_lags)
        return tuned
