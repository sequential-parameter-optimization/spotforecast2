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


def _parse_lags_from_string(lags_str: str) -> int | list:
    """Parse a lags string representation back to a Python object.

    Handles two formats:

    * Single integer as string: ``"24"`` → ``24``
    * List representation:      ``"[1, 2, 3]"`` → ``[1, 2, 3]``

    Args:
        lags_str: String representation of lags.

    Returns:
        Either an integer or a list of integers representing lags.

    Examples:
        >>> from spotforecast2.model_selection.spotoptim_search import (
        ...     _parse_lags_from_string,
        ... )
        >>> _parse_lags_from_string("24")
        24
        >>> _parse_lags_from_string("[1, 2, 3]")
        [1, 2, 3]
    """
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
        show_progress: (Handled by *verbose*.)
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

        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.linear_model import Ridge
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from spotforecast2.model_selection import (
        ...     TimeSeriesFold,
        ...     spotoptim_search_forecaster,
        ... )
        >>> np.random.seed(42)
        >>> y = pd.Series(
        ...     np.random.randn(200).cumsum(),
        ...     index=pd.date_range("2022-01-01", periods=200, freq="h"),
        ...     name="load",
        ... )
        >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=5)
        >>> cv = TimeSeriesFold(
        ...     steps=5,
        ...     initial_train_size=150,
        ...     refit=False,
        ... )
        >>> search_space = {"alpha": (0.01, 10.0)}
        >>> results, optimizer = spotoptim_search_forecaster(
        ...     forecaster=forecaster,
        ...     y=y,
        ...     cv=cv,
        ...     search_space=search_space,
        ...     metric="mean_absolute_error",
        ...     n_trials=5,
        ...     n_initial=3,
        ...     random_state=42,
        ...     return_best=False,
        ...     verbose=False,
        ...     show_progress=False,
        ... )
        >>> isinstance(results, pd.DataFrame)
        True
        >>> "alpha" in results.columns
        True

        **2 — ParameterSet-based search space:**

        >>> from spotoptim.hyperparameters import ParameterSet
        >>> ps = ParameterSet()
        >>> _ = ps.add_float("alpha", low=0.01, high=10.0)
        >>> results2, _ = spotoptim_search_forecaster(
        ...     forecaster=ForecasterRecursive(estimator=Ridge(), lags=5),
        ...     y=y,
        ...     cv=cv,
        ...     search_space=ps,
        ...     metric="mean_absolute_error",
        ...     n_trials=5,
        ...     n_initial=3,
        ...     return_best=False,
        ...     verbose=False,
        ...     show_progress=False,
        ... )
        >>> len(results2) == 5
        True
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
        suppress_warnings=suppress_warnings,
        output_file=output_file,
        kwargs_spotoptim=kwargs_spotoptim,
    )

    return results, optimizer


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
            show_progress=False,
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
            show_progress=False,
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
    bounds, var_type, var_name, var_trans = _convert_search_space(search_space)

    all_metric_values: list[list[float]] = []
    all_lags: list = []
    all_params: list[dict] = []

    # --- Objective function -----------------------------------------------
    def _objective(X: np.ndarray) -> np.ndarray:
        results_arr = []
        for params_array in X:
            params_dict = _array_to_params(params_array, var_name, var_type, bounds)
            sample_params = {k: v for k, v in params_dict.items() if k != "lags"}

            f_search = deepcopy(forecaster_search)
            f_search.set_params(**sample_params)

            if "lags" in params_dict:
                lags_value = _parse_lags_from_string(params_dict["lags"])
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
                    show_progress=False,
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
                lags_val = _parse_lags_from_string(lags_val)
            all_lags.append(lags_val)
            all_params.append(sample_params)
            results_arr.append(metrics_list[0])

        return np.array(results_arr)

    # --- Run SpotOptim ----------------------------------------------------
    optimizer = SpotOptim(
        fun=_objective,
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


def _convert_search_space(
    search_space: ParameterSet | Dict[str, Any],
) -> tuple[list, list, list, list]:
    """Convert a search space to the SpotOptim internal format.

    Accepts either a :class:`~spotoptim.hyperparameters.ParameterSet`
    or a plain ``dict``.  Three dict flavours are supported:

    1. **Raw SpotOptim dict** — keys ``bounds``, ``var_type``,
       ``var_name``, ``var_trans``.
    2. **Simple tuples** — ``{"param_name": (low, high), ...}``
       (int or float are inferred).
    3. **Factor list** — ``{"param_name": ["a", "b", "c"]}``

    Args:
        search_space: The search space to convert.

    Returns:
        Tuple ``(bounds, var_type, var_name, var_trans)``.

    Raises:
        TypeError: If *search_space* is not a supported type.
        ValueError: If a dict value is not a valid bound description.

    Examples:
        >>> from spotforecast2.model_selection.spotoptim_search import (
        ...     _convert_search_space,
        ... )
        >>> bounds, vt, vn, vtrans = _convert_search_space(
        ...     {"alpha": (0.01, 10.0), "max_depth": (2, 8)}
        ... )
        >>> vn
        ['alpha', 'max_depth']
        >>> vt
        ['float', 'int']

        >>> from spotoptim.hyperparameters import ParameterSet
        >>> ps = ParameterSet()
        >>> _ = ps.add_float("lr", low=0.001, high=0.1)
        >>> b, t, n, tr = _convert_search_space(ps)
        >>> n
        ['lr']
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
                isinstance(value, (list, tuple))
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


def _array_to_params(
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
        >>> import numpy as np
        >>> from spotforecast2.model_selection.spotoptim_search import (
        ...     _array_to_params,
        ... )
        >>> _array_to_params(
        ...     np.array([100.0, 0.05]),
        ...     var_name=["n_estimators", "lr"],
        ...     var_type=["int", "float"],
        ...     bounds=[(50, 200), (0.01, 0.3)],
        ... )
        {'n_estimators': 100, 'lr': 0.05}
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
