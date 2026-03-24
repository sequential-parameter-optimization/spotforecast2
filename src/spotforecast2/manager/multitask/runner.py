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
    bounds: Optional[List[Tuple[float, float]]] = None,
    data_frame_name: str = "demo10",
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
        bounds: Per-column hard outlier bounds as a list of
            ``(lower, upper)`` tuples, one per target column.  ``None``
            uses the package defaults.
        data_frame_name: Dataset identifier used for cache-directory and
            model-file naming.  Defaults to ``"demo10"``.
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

        forecast = run(df, task="lazy", data_frame_name="demo10", predict_size=24)
        print(forecast.head())
        ```
    """
    if task not in _ALL_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {sorted(_ALL_TASKS)}")

    if task == "clean":
        mt = MultiTask(
            task="clean",
            dataframe=dataframe,
            data_frame_name=data_frame_name,
            **kwargs,
        )
        mt.run()
        return pd.DataFrame()

    effective_bounds = bounds if bounds is not None else _DEFAULT_BOUNDS

    mt = MultiTask(
        task=task,
        dataframe=dataframe,
        data_frame_name=data_frame_name,
        agg_weights=_DEFAULT_AGG_WEIGHTS,
        bounds=effective_bounds,
        **kwargs,
    )
    mt.prepare_data()
    mt.detect_outliers()
    mt.impute()
    mt.build_exogenous_features()
    result = mt.run(show=False)
    return result["future_pred"].to_frame("forecast")
