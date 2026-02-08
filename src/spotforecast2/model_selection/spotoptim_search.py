# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hyperparameter search functions for forecasters using SpotOptim.
"""

from __future__ import annotations
from typing import Callable, Dict, Any
import warnings
from copy import deepcopy
import ast
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

from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold
from spotforecast2_safe.model_selection import OneStepAheadFold
from spotforecast2_safe.model_selection import _backtesting_forecaster
from spotforecast2.forecaster.metrics import add_y_train_argument, _get_metric
from spotforecast2.model_selection.utils_common import (
    check_one_step_ahead_input,
    check_backtesting_input,
    select_n_jobs_backtesting,
)
from spotforecast2.model_selection.utils_metrics import (
    _calculate_metrics_one_step_ahead,
)
from spotforecast2.forecaster.utils import (
    initialize_lags,
    date_to_index_position,
)
from spotforecast2.exceptions import IgnoredArgumentWarning
from spotforecast2_safe.exceptions import set_skforecast_warnings


def _parse_lags_from_string(lags_str: str) -> int | list:
    """
    Parse lags string representation back to Python object.

    Handles two formats:
    - Single integer as string: "24" -> 24
    - List representation: "[1, 2, 3]" -> [1, 2, 3]

    Args:
        lags_str: String representation of lags.

    Returns:
        Either an integer or a list of integers representing lags.
    """
    lags_str = lags_str.strip()
    if lags_str.startswith("["):
        # It's a list representation - parse it
        return ast.literal_eval(lags_str)
    else:
        # It's a single integer
        try:
            return int(lags_str)
        except ValueError:
            # Fallback for unexpected formats
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
    kwargs_spotoptim: dict = {},
) -> tuple[pd.DataFrame, object]:
    """
    Hyperparameter optimization for a Forecaster using SpotOptim.

    Performs hyperparameter search using the SpotOptim library for a
    Forecaster object. Validation is done using time series backtesting with
    the provided cross-validation strategy.

    Args:
        forecaster: Forecaster model. Can be ForecasterRecursive, ForecasterDirect,
            or any compatible forecaster class.
        y: Training time series values. Must be a pandas Series with a
            datetime or numeric index.
        cv: Cross-validation strategy with information needed to split the data
            into folds. Must be an instance of TimeSeriesFold or OneStepAheadFold.
        search_space: Hyperparameter search space. Can be either:
            - ParameterSet: A ParameterSet object from spotoptim.hyperparameters
              defining parameters with their types, bounds, and transformations.
            - Dict: A dictionary with keys 'bounds', 'var_type', 'var_name', 'var_trans'
              (SpotOptim format), or a mapping of parameter names to (low, high) tuples.
        metric: Metric(s) to quantify model goodness of fit.
        exog: Exogenous variable(s) included as predictors. Default is None.
        n_trials: Total number of evaluations (initial + sequential iterations).
            Default is 10.
        n_initial: Number of initial random points to sample before starting
            sequential optimization. Default is 5.
        random_state: Seed for sampling reproducibility. Default is 123.
        return_best: If True, refit the forecaster using the best parameters
            found on the whole dataset at the end. Default is True.
        n_jobs: Number of parallel jobs. If -1, uses all cores. If 'auto',
            automatically determines the number of jobs. Default is 'auto'.
        verbose: If True, print optimization progress. Default is False.
        show_progress: Whether to show progress (currently handled by verbose).
            Default is True.
        suppress_warnings: If True, suppress spotforecast warnings. Default is False.
        output_file: Filename or full path to save results as TSV. Default is None.
        kwargs_spotoptim: Additional keyword arguments passed to SpotOptim().
            Default is {}.

    Returns:
        tuple[pd.DataFrame, object]: A tuple containing:
            - results: DataFrame with columns 'lags', 'params', metric values,
              and individual parameter columns. Sorted by the first metric.
            - optimizer: SpotOptim object containing the optimization results.

    Raises:
        ValueError: If exog length doesn't match y length when return_best=True.
        TypeError: If cv is not an instance of TimeSeriesFold or OneStepAheadFold.
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
    kwargs_spotoptim: dict = {},
) -> tuple[pd.DataFrame, object]:
    """
    Internal implementation of SpotOptim search.
    """

    set_skforecast_warnings(suppress_warnings, action="ignore")

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

    # Convert search space to SpotOptim format
    bounds, var_type, var_name, var_trans = _convert_search_space(search_space)

    all_metric_values = []
    all_lags = []
    all_params = []

    def _objective(X: np.ndarray) -> np.ndarray:
        results = []
        for params_array in X:
            params_dict = _array_to_params(params_array, var_name, var_type, bounds)
            sample_params = {k: v for k, v in params_dict.items() if k != "lags"}

            # Use a fresh copy for each evaluation to be safe in multithreaded (though SpotOptim is usually seq)
            # or just reset state.
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
            results.append(metrics_list[0])

        return np.array(results)

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
        **kwargs_spotoptim,
    )

    optimizer.optimize()

    # Build results
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


def _convert_search_space(
    search_space: ParameterSet | Dict[str, Any],
) -> tuple[list, list, list, list]:
    """
    Convert search space to SpotOptim internal format.
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

        # Convert dictionary mapping to SpotOptim format
        bounds, var_type, var_name, var_trans = [], [], [], []
        for name, value in search_space.items():
            var_name.append(name)
            var_trans.append(None)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                if isinstance(value[0], int) and isinstance(value[1], int):
                    var_type.append("int")
                else:
                    var_type.append("float")
                bounds.append(value)
            elif isinstance(value, list):
                var_type.append("factor")
                bounds.append(value)
            else:
                raise ValueError(f"Invalid search space for '{name}': {value}")
        return bounds, var_type, var_name, var_trans

    raise TypeError(
        f"search_space must be ParameterSet or dict, got {type(search_space)}"
    )


def _array_to_params(
    params_array: np.ndarray, var_name: list, var_type: list, bounds: list
) -> Dict[str, Any]:
    """
    Convert SpotOptim array back to parameter dictionary.
    """
    params_dict = {}
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
