# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convenience runner for the MultiTask forecasting pipeline.

Provides a single ``run`` function that wraps the full pipeline
sequence (prepare_data, detect_outliers, impute, build_exogenous_features,
run) behind a one-call interface.
"""

from typing import Any, List, Optional, Tuple

import pandas as pd

from spotforecast2.manager.multitask.multi import MultiTask

_DEFAULT_BOUNDS: List[Tuple[float, float]] = [
    (-2500, 4500),
    (-10, 3000),
    (0, 230),
    (0, 550),
    (0, 1400),
    (0, 1400),
    (0, 10),
    (0, 4500),
    (0, 300),
    (0, 400),
    (0, 300),
]

_DEFAULT_AGG_WEIGHTS: List[float] = [
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
]

_PIPELINE_TASKS = frozenset({"lazy", "optuna", "spotoptim", "predict"})
_ALL_TASKS = _PIPELINE_TASKS | {"clean"}


def run(
    dataframe: pd.DataFrame,
    task: str = "lazy",
    cache_data: bool = False,
    cache_home: Optional[str] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    agg_weights: Optional[List[float]] = None,
    project_name: str = "test_project",
    n_trials_optuna: Optional[int] = 10,
    train_days: Optional[int] = 3*365,
    val_days: Optional[int] = 31,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run the MultiTask forecasting pipeline and return predictions.

    Wraps the standard pipeline sequence into a single call.  For the
    ``"clean"`` task only the cache directory is wiped and an empty
    DataFrame is returned.  For all other tasks the full sequence

        prepare_data → detect_outliers → impute →
        build_exogenous_features → run

    is executed and the aggregated future predictions are returned as a
    DataFrame.

    Args:
        dataframe: Input time-series data.  Must contain a datetime
            column matching the configured ``index_name`` and at least one
            numeric target column.
        task: Pipeline mode — one of ``"lazy"``, ``"optuna"``,
            ``"spotoptim"``, ``"predict"``, or ``"clean"``.
            Defaults to ``"lazy"``.
        cache_data: Whether to cache the preprocessed data.  Defaults to
            ``False``.
        cache_home: Optional path to the cache directory.  Defaults to
            ``None``, which uses the package default cache location that
            is defined via spotforecast2_safe's `get_cache_home()`.
        bounds: Per-column hard outlier bounds as a list of
            ``(lower, upper)`` tuples, one per target column.  ``None``
            uses the package defaults.
        agg_weights: Per-column weights for the final aggregation step as
            a list of floats, one per target column.  ``None`` uses the
            package defaults.
        project_name: Identifier used for cache-directory and
            model-file naming.  Defaults to ``"test_project"``.
        train_days: Optional number of days in the training window. Defaults to 3 years (1095 days). 
        val_days: Optional number of days in the validation window.  If
            ``None``, the default of 31 days is used.
        **kwargs: Additional keyword arguments forwarded verbatim to
            MultiTask (e.g. ``predict_size``, ``train_days``,
            ``val_days``, ``cache_home``).

    Returns:
        DataFrame whose index is the forecast horizon timestamps and
        whose single column ``"forecast"`` contains the aggregated
        predicted values.  For the ``"clean"`` task an empty DataFrame
        is returned.

    Raises:
        ValueError: If ``task`` is not one of the supported task names.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2.manager.multitask.runner import run
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

        data_home = get_package_data_home()
        df = fetch_data(filename=str(data_home / "demo10.csv"))

        forecast = run(df, task="lazy", project_name="demo10", predict_size=24)
        print(forecast.head())
        ```
    """
    if task not in _ALL_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {sorted(_ALL_TASKS)}")

    if task == "clean":
        mt = MultiTask(
            task="clean",
            data_frame_name=project_name,
            cache_data=cache_data,
            cache_home=cache_home,
            **kwargs,
        )
        mt.run()
        return pd.DataFrame()

    effective_bounds = bounds if bounds is not None else _DEFAULT_BOUNDS
    effective_agg_weights = (
        agg_weights if agg_weights is not None else _DEFAULT_AGG_WEIGHTS
    )

    mt = MultiTask(
        dataframe=dataframe,
        task=task,
        data_frame_name=project_name,
        agg_weights=effective_agg_weights,
        bounds=effective_bounds,
        cache_data=cache_data,
        cache_home=cache_home,
        n_trials_optuna=n_trials_optuna,
        train_days=train_days,
        val_days=val_days,
        **kwargs,
    )
    mt.prepare_data()
    mt.detect_outliers()
    mt.impute()
    mt.build_exogenous_features()
    result = mt.run(show=False)
    return result["future_pred"].to_frame("forecast")
