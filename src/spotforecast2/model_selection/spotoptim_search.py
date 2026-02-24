# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hyperparameter search functions for forecasters using SpotOptim.

This module provides an alternative to Bayesian (Optuna-based) search
by leveraging the SpotOptim surrogate-model-based optimizer.  It
follows the same interface as
:func:`spotforecast2.model_selection.bayesian_search_forecaster`, so the
two can be used interchangeably.
"""

from __future__ import annotations

import ast
import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

try:
    from spotoptim import SpotOptim
    from spotoptim.hyperparameters import ParameterSet
except ImportError:
    warnings.warn(
        "spotoptim is not installed. spotoptim_search_forecaster will not work. "
        "Install it using `pip install spotoptim`.",
        ImportWarning,
    )

from spotforecast2.exceptions import IgnoredArgumentWarning
from spotforecast2.forecaster.metrics import _get_metric, add_y_train_argument
from spotforecast2.forecaster.utils import date_to_index_position, initialize_lags
from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold
from spotforecast2.model_selection.utils_common import (
    check_backtesting_input,
    check_one_step_ahead_input,
    select_n_jobs_backtesting,
)
from spotforecast2.model_selection.utils_metrics import (
    _calculate_metrics_one_step_ahead,
)
from spotforecast2_safe.exceptions import set_skforecast_warnings
from spotforecast2_safe.model_selection import OneStepAheadFold, _backtesting_forecaster

logger = logging.getLogger(__name__)


def parse_lags_from_strings(lags_str: str | int | list) -> int | list:
    """Parse a lags representation back to a Python object.

    Handles three input scenarios:
    1. Already an integer or list: returned as is.
    2. Single integer as string: ``"24"`` → ``24``
    3. List representation:      ``"[1, 2, 3]"`` → ``[1, 2, 3]``

    Args:
        lags_str: Lag specification (string, int, or list).

    Returns:
        Either an integer or a list of integers representing lags.

    Examples:
        Basic parsing:

        >>> from spotforecast2.model_selection.spotoptim_search import (
        ...     parse_lags_from_strings,
        ... )
        >>> parse_lags_from_strings(24)
        24
        >>> parse_lags_from_strings("[1, 2, 3]")
        [1, 2, 3]

        Visualizing the safety threshold (Example of dynamic documentation):

        ```{python}
        import matplotlib.pyplot as plt
        import numpy as np

        def check_safety_threshold(val, threshold):
            return 1 if val >= threshold else 0

        threshold = 0.95
        x = np.linspace(0.8, 1.0, 50)
        y = [check_safety_threshold(val, threshold) for val in x]

        plt.step(x, y, where='post')
        plt.axvline(threshold, color='red', linestyle='--')
        plt.title("Safety Status Transition")
        # plt.show()  # Commented for non-interactive environments
        ```
    """
    if isinstance(lags_str, (int, list)):
        return lags_str

    lags_str = lags_str.strip()
    if lags_str.startswith("["):
        return ast.literal_eval(lags_str)
    else:
        try:
            return int(lags_str)
        except ValueError:
            return lags_str


def spotoptim_search_forecaster(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    search_space: ParameterSet | Dict[str, Any],
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    n_trials: int = 10,
    n_initial: int = 5,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: int | str = "auto",
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
    kwargs_spotoptim: dict | None = None,
) -> tuple[pd.DataFrame, object]:
    """Hyperparameter optimisation for a Forecaster using SpotOptim.

    Drop-in alternative to
    :func:`~spotforecast2.model_selection.bayesian_search_forecaster`
    that uses the SpotOptim surrogate-model-based optimizer instead of
    Optuna's TPE sampler.

    Args:
        forecaster: Forecaster model (e.g. ``ForecasterRecursive``).
        y: Training time series.  Must have a datetime or numeric index.
        cv: Cross-validation strategy — ``TimeSeriesFold`` or
            ``OneStepAheadFold``.
        search_space: Hyperparameter search space.  Either a
            :class:`~spotoptim.hyperparameters.ParameterSet` or a plain
            ``dict`` (see examples below).
        metric: Metric name, callable, or list thereof.
        exog: Optional exogenous variable(s).
        n_trials: Total evaluations (initial + sequential).
        n_initial: Random initial points before surrogate kicks in.
        random_state: RNG seed.
        return_best: Re-fit forecaster with best params after search.
        n_jobs: Parallel jobs for backtesting (``"auto"`` or int).
        verbose: Print optimisation progress.
        show_progress: Show progress bar during backtesting/validation.
        suppress_warnings: Suppress spotforecast warnings.
        output_file: Save results as TSV to this path.
        kwargs_spotoptim: Extra kwargs passed to ``SpotOptim()``.

    Returns:
        tuple: ``(results, optimizer)`` where *results* is a sorted
        ``DataFrame`` and *optimizer* is the ``SpotOptim`` instance.

    Raises:
        ValueError: If ``exog`` length ≠ ``y`` length and
            ``return_best`` is True.
        TypeError: If ``cv`` is not ``TimeSeriesFold`` or
            ``OneStepAheadFold``.

    Examples:
        **1 — Dict-based search space (no ParameterSet needed):**

        ```{python}
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Ridge
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2.model_selection import (
            TimeSeriesFold,
            spotoptim_search_forecaster,
        )

        np.random.seed(42)
        y = pd.Series(
            np.random.randn(200).cumsum(),
            index=pd.date_range("2022-01-01", periods=200, freq="h"),
            name="load",
        )

        forecaster = ForecasterRecursive(estimator=Ridge(), lags=5)
        cv = TimeSeriesFold(
            steps=5,
            initial_train_size=150,
            refit=False,
        )

        search_space = {"alpha": (0.01, 10.0)}

        results, optimizer = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            search_space=search_space,
            metric="mean_absolute_error",
            n_trials=5,
            n_initial=3,
            random_state=42,
            return_best=False,
            verbose=False,
            show_progress=False,
        )

        print(f"Is DataFrame: {isinstance(results, pd.DataFrame)}")
        print(f"Contains 'alpha': {'alpha' in results.columns}")
        ```

        **2 — ParameterSet-based search space:**

        ```{python}
        from spotoptim.hyperparameters import ParameterSet

        ps = ParameterSet()
        _ = ps.add_float("alpha", low=0.01, high=10.0)

        results2, _ = spotoptim_search_forecaster(
            forecaster=ForecasterRecursive(estimator=Ridge(), lags=5),
            y=y,
            cv=cv,
            search_space=ps,
            metric="mean_absolute_error",
            n_trials=5,
            n_initial=3,
            return_best=False,
            verbose=False,
            show_progress=False,
        )

        print(f"Number of configurations evaluated: {len(results2)}")
        ```
    """

    if return_best and exog is not None and (len(exog) != len(y)):
        raise ValueError(
            f"`exog` must have same number of samples as `y`. "
            f"length `exog`: ({len(exog)}), length `y`: ({len(y)})"
        )

    results, optimizer = _spotoptim_search(
        forecaster=forecaster,
        y=y,
        cv=cv,
        exog=exog,
        search_space=search_space,
        metric=metric,
        n_trials=n_trials,
        n_initial=n_initial,
        random_state=random_state,
        return_best=return_best,
        n_jobs=n_jobs,
        verbose=verbose,
        show_progress=show_progress,
        suppress_warnings=suppress_warnings,
        output_file=output_file,
        kwargs_spotoptim=kwargs_spotoptim,
    )

    return results, optimizer


def spotoptim_objective(
    X: np.ndarray,
    forecaster_search: object,
    cv_name: str,
    cv: TimeSeriesFold | OneStepAheadFold,
    metric: list[Callable],
    y: pd.Series,
    exog: pd.Series | pd.DataFrame | None,
    n_jobs: int,
    verbose: bool,
    show_progress: bool,
    suppress_warnings: bool,
    var_name: list,
    var_type: list,
    bounds: list,
    all_metric_values: list[list[float]],
    all_lags: list,
    all_params: list[dict],
) -> np.ndarray:
    """SpotOptim objective function to evaluate hyperparameter sets.

    Evaluates a given array of hyperparameter configurations `X` and returns an array
    of the primary metric errors.

    Args:
        X: 2D array of hyperparameters from SpotOptim.
        forecaster_search: The forecaster to evaluate.
        cv_name: Type of cross-validation ("TimeSeriesFold" or "OneStepAheadFold").
        cv: Cross-validation configuration.
        metric: List of metrics to compute.
        y: Target time series.
        exog: Exogenous variables.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity level flag.
        show_progress: Show progress bar flag.
        suppress_warnings: Suppress warnings flag.
        var_name: Parameter names.
        var_type: Parameter types.
        bounds: Parameter bounds.
        all_metric_values: List to record all metric results.
        all_lags: List to record all evaluated lag configurations.
        all_params: List to record all evaluated parameters.

    Returns:
        np.ndarray: 1D array of results for the primary metric.

    Examples:
        Generating textual output of parameter evaluation:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.model_selection import TimeSeriesFold
        from spotforecast2.model_selection.spotoptim_search import spotoptim_objective

        # Mock forecaster for documentation
        class MockForecaster:
            def set_params(self, **kwargs): pass
            def set_lags(self, lags): pass

        # Provide dummy data and configuration
        X = np.array([[0.05], [0.1]])
        cv = TimeSeriesFold(initial_train_size=10, steps=2)
        metric = [lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))]

        # Track results
        metric_vals, lags, params = [], [], []

        # When evaluated for real, the mock objects would produce metrics.
        # Here we just show the call structure.
        print("Ready to evaluate hyperparameters.")
        ```
    """
    results_arr = []
    for params_array in X:
        params_dict = array_to_params(params_array, var_name, var_type, bounds)
        sample_params = {k: v for k, v in params_dict.items() if k != "lags"}

        f_search = deepcopy(forecaster_search)
        f_search.set_params(**sample_params)

        if "lags" in params_dict:
            lags_value = parse_lags_from_strings(params_dict["lags"])
            f_search.set_lags(lags_value)

        if cv_name == "TimeSeriesFold":
            metrics_df, _ = _backtesting_forecaster(
                forecaster=f_search,
                y=y,
                cv=cv,
                exog=exog,
                metric=metric,
                n_jobs=n_jobs,
                verbose=verbose,
                show_progress=show_progress,
                suppress_warnings=suppress_warnings,
            )
            metrics_list = metrics_df.iloc[0, :].to_list()
        else:
            X_train, y_train, X_test, y_test = (
                f_search._train_test_split_one_step_ahead(
                    y=y, initial_train_size=cv.initial_train_size, exog=exog
                )
            )
            metrics_list = _calculate_metrics_one_step_ahead(
                forecaster=f_search,
                metrics=metric,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        all_metric_values.append(metrics_list)
        lags_val = params_dict.get(
            "lags", f_search.lags if hasattr(f_search, "lags") else None
        )
        if isinstance(lags_val, str):
            lags_val = parse_lags_from_strings(lags_val)
        all_lags.append(lags_val)
        all_params.append(sample_params)
        results_arr.append(metrics_list[0])

    return np.array(results_arr)


def _spotoptim_search(
    forecaster: object,
    y: pd.Series,
    cv: TimeSeriesFold | OneStepAheadFold,
    search_space: ParameterSet | Dict[str, Any],
    metric: str | Callable | list[str | Callable],
    exog: pd.Series | pd.DataFrame | None = None,
    n_trials: int = 10,
    n_initial: int = 5,
    random_state: int = 123,
    return_best: bool = True,
    n_jobs: int | str = "auto",
    verbose: bool = False,
    show_progress: bool = True,
    suppress_warnings: bool = False,
    output_file: str | None = None,
    kwargs_spotoptim: dict | None = None,
) -> tuple[pd.DataFrame, object]:
    """Internal implementation of the SpotOptim search.

    This function is not intended to be called directly. Use
    :func:`spotoptim_search_forecaster` instead.

    Returns:
        tuple: ``(results_df, optimizer)``
    """

    set_skforecast_warnings(suppress_warnings, action="ignore")

    kwargs_spotoptim_ = kwargs_spotoptim.copy() if kwargs_spotoptim is not None else {}

    forecaster_search = deepcopy(forecaster)
    forecaster_name = type(forecaster_search).__name__
    is_regression = (
        forecaster_search.__spotforecast_tags__["forecaster_task"] == "regression"
    )
    cv_name = type(cv).__name__

    if cv_name not in ["TimeSeriesFold", "OneStepAheadFold"]:
        raise TypeError(
            f"`cv` must be an instance of `TimeSeriesFold` or `OneStepAheadFold`. "
            f"Got {type(cv)}."
        )

    if cv_name == "OneStepAheadFold":
        check_one_step_ahead_input(
            forecaster=forecaster_search,
            cv=cv,
            metric=metric,
            y=y,
            exog=exog,
            show_progress=show_progress,
            suppress_warnings=False,
        )

        cv = deepcopy(cv)
        initial_train_size = date_to_index_position(
            index=cv._extract_index(y),
            date_input=cv.initial_train_size,
            method="validation",
            date_literal="initial_train_size",
        )
        cv.set_params(
            {
                "initial_train_size": initial_train_size,
                "window_size": forecaster_search.window_size,
                "differentiation": forecaster_search.differentiation_max,
                "verbose": verbose,
            }
        )
    else:
        check_backtesting_input(
            forecaster=forecaster_search,
            cv=cv,
            y=y,
            metric=metric,
            exog=exog,
            n_jobs=n_jobs,
            show_progress=show_progress,
            suppress_warnings=suppress_warnings,
        )

    if not isinstance(metric, list):
        metric = [metric]
    metric = [
        _get_metric(metric=m) if isinstance(m, str) else add_y_train_argument(m)
        for m in metric
    ]
    metric_dict = {(m if isinstance(m, str) else m.__name__): [] for m in metric}

    if len(metric_dict) != len(metric):
        raise ValueError("When `metric` is a `list`, each metric name must be unique.")

    if n_jobs == "auto":
        refit = cv.refit if isinstance(cv, TimeSeriesFold) else None
        n_jobs = select_n_jobs_backtesting(forecaster=forecaster_search, refit=refit)
    elif isinstance(cv, TimeSeriesFold) and cv.refit != 1 and n_jobs != 1:
        warnings.warn(
            "If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
            "is set to 1 to avoid unexpected results during parallelization.",
            IgnoredArgumentWarning,
        )
        n_jobs = 1

    # --- Convert search space to SpotOptim arrays -------------------------
    bounds, var_type, var_name, var_trans = convert_search_space(search_space)

    all_metric_values: list[list[float]] = []
    all_lags: list = []
    all_params: list[dict] = []

    # --- Objective function -----------------------------------------------
    def _objective_wrapper(X: np.ndarray) -> np.ndarray:
        return spotoptim_objective(
            X=X,
            forecaster_search=forecaster_search,
            cv_name=cv_name,
            cv=cv,
            metric=metric,
            y=y,
            exog=exog,
            n_jobs=n_jobs,
            verbose=verbose,
            show_progress=show_progress,
            suppress_warnings=suppress_warnings,
            var_name=var_name,
            var_type=var_type,
            bounds=bounds,
            all_metric_values=all_metric_values,
            all_lags=all_lags,
            all_params=all_params,
        )

    # --- Run SpotOptim ----------------------------------------------------
    optimizer = SpotOptim(
        fun=_objective_wrapper,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        var_trans=var_trans,
        max_iter=n_trials,
        n_initial=n_initial,
        seed=random_state,
        verbose=verbose,
        **kwargs_spotoptim_,
    )

    optimizer.optimize()

    # --- Build results DataFrame ------------------------------------------
    lags_list = [
        initialize_lags(forecaster_name=forecaster_name, lags=lag)[0]
        for lag in all_lags
    ]

    for metrics_vals in all_metric_values:
        for m, m_val in zip(metric, metrics_vals):
            m_name = m if isinstance(m, str) else m.__name__
            metric_dict[m_name].append(m_val)

    results = pd.DataFrame({"lags": lags_list, "params": all_params, **metric_dict})
    results = results.sort_values(
        by=list(metric_dict.keys())[0], ascending=True if is_regression else False
    ).reset_index(drop=True)
    results = pd.concat([results, results["params"].apply(pd.Series)], axis=1)

    if output_file is not None:
        results.to_csv(output_file, sep="\t", index=False)

    if return_best:
        best_lags = results.loc[0, "lags"]
        best_params = results.loc[0, "params"]
        best_metric = results.loc[0, list(metric_dict.keys())[0]]

        forecaster.set_lags(best_lags)
        forecaster.set_params(**best_params)
        forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)

        print(
            f"`Forecaster` refitted using the best-found lags and parameters, "
            f"and the whole data set: \n"
            f"  Lags: {best_lags} \n"
            f"  Parameters: {best_params}\n"
            f"  {'Backtesting' if cv_name == 'TimeSeriesFold' else 'One-step-ahead'} "
            f"metric: {best_metric}"
        )

    set_skforecast_warnings(suppress_warnings, action="default")

    return results, optimizer


# ======================================================================
# Conversion helpers
# ======================================================================


def convert_search_space(
    search_space: ParameterSet | dict[str, Any],
) -> tuple[list[Any], list[str], list[str], list[Callable | None]]:
    """Convert search space into SpotOptim compatible format.

    Args:
        search_space: Search space as a SpotOptim ParameterSet or a dictionary.

    Returns:
        tuple containing:
        - bounds: List of parameter bounds or categories.
        - var_type: List of variable types ('float', 'int', or 'factor').
        - var_name: List of variable names.
        - var_trans: List of transformation functions (e.g., log10) or None.

    Examples:
        Basic usage:

        >>> from spotoptim.hyperparameters import ParameterSet
        >>> from spotforecast2.model_selection.spotoptim_search import (
        ...     convert_search_space,
        ... )
        >>> ps = ParameterSet()
        >>> _ = ps.add_float("alpha", 0.01, 10.0)
        >>> b, t, n, tr = convert_search_space(ps)
        >>> b
        [(0.01, 10.0)]
        >>> t
        ['float']

        Converting a complex dictionary search space:

        ```{python}
        from spotforecast2.model_selection.spotoptim_search import convert_search_space

        search_space = {
            "learning_rate": (0.001, 0.1, "log10"),
            "max_depth": (2, 10),
            "model_type": ["RandomForest", "XGBoost"]
        }

        bounds, vt, vn, vtr = convert_search_space(search_space)

        for name, typ, bound, trans in zip(vn, vt, bounds, vtr):
            print(f"{name} ({typ}): {bound} | transform: {trans}")
        ```
    """
    if isinstance(search_space, ParameterSet):
        return (
            search_space.bounds,
            search_space.var_type,
            search_space.var_name,
            search_space.var_trans,
        )

    if isinstance(search_space, dict):
        if all(
            k in search_space for k in ["bounds", "var_type", "var_name", "var_trans"]
        ):
            return (
                search_space["bounds"],
                search_space["var_type"],
                search_space["var_name"],
                search_space["var_trans"],
            )

        bounds, var_type, var_name, var_trans = [], [], [], []
        for name, value in search_space.items():
            var_name.append(name)

            # Formats:
            # 2-element tuple/list (low, high) -> var_trans = None
            # 3-element tuple/list (low, high, trans) -> var_trans = trans

            if (
                isinstance(value, tuple)
                and len(value) in (2, 3)
                and isinstance(value[0], (int, float))
                and isinstance(value[1], (int, float))
            ):
                if isinstance(value[0], int) and isinstance(value[1], int):
                    var_type.append("int")
                else:
                    var_type.append("float")

                bounds.append(value[:2])  # Keep only low/high for bounds

                if len(value) == 3:
                    var_trans.append(value[2])
                else:
                    var_trans.append(None)

            elif isinstance(value, list):
                # Categorical factors
                var_type.append("factor")
                bounds.append(value)
                var_trans.append(None)
            else:
                raise ValueError(f"Invalid search space for '{name}': {value}")

        return bounds, var_type, var_name, var_trans

    raise TypeError(
        f"search_space must be ParameterSet or dict, got {type(search_space)}"
    )


def array_to_params(
    params_array: np.ndarray,
    var_name: list,
    var_type: list,
    bounds: list,
) -> Dict[str, Any]:
    """Convert a SpotOptim parameter array back to a dict.

    Each element of *params_array* is mapped to the corresponding
    name / type / bounds entry, converting to the correct Python type.

    Args:
        params_array: 1-D array of raw parameter values from SpotOptim.
        var_name: Parameter names (same order as *params_array*).
        var_type: Parameter types (``"int"``, ``"float"``, ``"factor"``).
        bounds: Parameter bounds.

    Returns:
        Dictionary mapping parameter names to typed values.

    Examples:
        Basic usage:

        >>> import numpy as np
        >>> from spotforecast2.model_selection.spotoptim_search import (
        ...     array_to_params,
        ... )
        >>> array_to_params(
        ...     np.array([100.0, 0.05]),
        ...     var_name=["n_estimators", "lr"],
        ...     var_type=["int", "float"],
        ...     bounds=[(50, 200), (0.01, 0.3)],
        ... )
        {'n_estimators': 100, 'lr': 0.05}

        Generating textual output of parameter mapping:

        ```{python}
        import numpy as np
        from spotforecast2.model_selection.spotoptim_search import array_to_params

        params_array = np.array([0.05, 5.0, 2.0])
        var_name = ["alpha", "max_depth", "model"]
        var_type = ["float", "int", "factor"]
        bounds = [(0.01, 10.0), (2, 8), ["Ridge", "Lasso", "ElasticNet"]]

        params_dict = array_to_params(params_array, var_name, var_type, bounds)

        for k, v in params_dict.items():
            print(f"{k}: {v} (type: {type(v).__name__})")
        ```
    """
    params_dict: Dict[str, Any] = {}
    for i, (name, ptype, value) in enumerate(zip(var_name, var_type, params_array)):
        if ptype == "factor":
            str_value = str(value)
            if str_value in bounds[i]:
                params_dict[name] = str_value
            else:
                try:
                    idx = int(round(float(str_value)))
                    idx = max(0, min(idx, len(bounds[i]) - 1))
                    params_dict[name] = bounds[i][idx]
                except (ValueError, TypeError):
                    params_dict[name] = str_value
        elif ptype == "int":
            params_dict[name] = int(round(float(str(value))))
        else:
            params_dict[name] = float(str(value))
    return params_dict
