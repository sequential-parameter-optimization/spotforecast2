# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convenience runner for the full MultiTask forecasting pipeline.

Provides the ``run`` entry point that wraps the standard pipeline sequence
(``prepare_data`` -> ``detect_outliers`` -> ``impute`` ->
``build_exogenous_features`` -> ``run``) behind a single call, returning the
aggregated forecast as a DataFrame.

This is the ``spotforecast2`` counterpart of
``spotforecast2_safe.multitask.runner.run``.  Rather than duplicating the
orchestration body, it binds the full-package :class:`MultiTask` (which adds
plotting and the auto-tuning tasks) and the wider task set to the reusable
``run_with`` seam that ``spotforecast2-safe`` exposes for exactly this purpose.
Consequently this runner additionally supports the auto-tuning tasks
``"optuna"`` and ``"spotoptim"`` that the safe package rejects, and honours
``plot_with_outliers=True`` (which raises ``NotImplementedError`` in the safe
package).

Available tasks: ``"lazy"``, ``"defaults"``, ``"optuna"``, ``"spotoptim"``,
``"predict"``, ``"clean"``.
"""

from typing import Any, FrozenSet, List, Optional

import pandas as pd
from spotforecast2_safe.multitask.runner import run_with

from spotforecast2.multitask.base import PipelineConfig
from spotforecast2.multitask.multi import MultiTask

# The full task surface available in ``spotforecast2``: the safe pipeline tasks
# plus the auto-tuning tasks the sibling package adds.
PIPELINE_TASKS: FrozenSet[str] = frozenset(
    {"lazy", "defaults", "optuna", "spotoptim", "predict"}
)
_ALL_TASKS: FrozenSet[str] = PIPELINE_TASKS | {"clean"}


def _unknown_task_message(task: str, sorted_tasks: List[str]) -> str:
    """Build the ``ValueError`` message for an unrecognised task name."""
    return f"Unknown task '{task}'. Choose from: {sorted_tasks}."


def run(
    config: Optional[PipelineConfig] = None,
    *,
    task: str = "lazy",
    dataframe: Optional[pd.DataFrame] = None,
    data_test: Optional[pd.DataFrame] = None,
    project_name: str = "test_project",
    cache_home: Optional[str] = None,
    plot_with_outliers: bool = False,
    show: bool = False,
    show_progress: bool = False,
    dry_run: bool = False,
    log_level: int = 40,
    **overrides: Any,
) -> pd.DataFrame:
    """Run the MultiTask forecasting pipeline and return predictions.

    Wraps the standard pipeline sequence into a single call.  For the
    ``"clean"`` task only the cache directory is wiped and an empty
    DataFrame is returned.  For all other tasks the full sequence

        prepare_data -> detect_outliers -> impute ->
        build_exogenous_features -> run

    is executed and the aggregated future predictions are returned as a
    DataFrame.

    Available tasks: ``"lazy"``, ``"defaults"``, ``"optuna"``,
    ``"spotoptim"``, ``"predict"``, ``"clean"``.  The auto-tuning tasks
    ``"optuna"`` and ``"spotoptim"`` are available here (unlike in the
    ``spotforecast2-safe`` runner, which rejects them).

    Args:
        config: A ``PipelineConfig``-conforming object (typically
            ``ConfigMulti``).  When ``None``, a fresh ``ConfigMulti()`` is
            constructed with default fields.  Outlier ``bounds`` and
            aggregation ``agg_weights`` are domain-specific calibrations and
            must be supplied explicitly on ``ConfigMulti``.
        task: Pipeline mode — one of ``"lazy"``, ``"defaults"``,
            ``"optuna"``, ``"spotoptim"``, ``"predict"``, or ``"clean"``.
            Defaults to ``"lazy"``.
        dataframe: Input time-series data.  Must contain a datetime column
            matching ``config.index_name`` and at least one numeric target
            column.  Optional for ``"clean"``, required otherwise.
        data_test: Ground-truth DataFrame covering the prediction horizon.
            When supplied, populates ``test_actual`` and ``metrics_future``
            in the prediction package.  Optional.
        project_name: Active-dataset identifier.  Sets
            ``config.data_frame_name``, which drives cache-subdirectory and
            model-file naming.
        cache_home: Cache directory override.  When ``None``, the package
            default from ``get_cache_home()`` is used.
        plot_with_outliers: Whether to render the optional
            outlier-visualisation step between ``detect_outliers`` and
            ``impute``.  Available in ``spotforecast2`` (the figure is shown);
            the same flag raises ``NotImplementedError`` in
            ``spotforecast2-safe``.
        show: Whether to invoke the prediction display hooks after the task
            runs.
        show_progress: Whether to print progress messages during pipeline
            execution.
        dry_run: Forwarded to ``MultiTask``; only meaningful for the
            ``"clean"`` task.
        log_level: Logging level.  Defaults to 40 (ERROR).
        **overrides: Forwarded to ``config.set_params(**overrides)`` — a
            convenience for one-line tweaks (e.g. ``predict_size=24``,
            ``n_trials_optuna=25``) without building a fresh config.  Unknown
            keys raise ``ValueError``.  Mutates the caller's config object.

    Returns:
        DataFrame whose index is the forecast horizon timestamps and whose
        single column ``"forecast"`` contains the aggregated predicted values.
        For the ``"clean"`` task an empty DataFrame is returned.

    Raises:
        ValueError: If ``task`` is not one of the supported task names, or if
            an unknown key is passed via ``**overrides``.

    Examples:
        Run the pipeline using cached or default model parameters
        (``"lazy"`` task) and read the aggregated forecast off the returned
        DataFrame:

        ```{python}
        import tempfile
        import warnings

        warnings.filterwarnings("ignore")

        from spotforecast2.multitask import run
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

        data_home = get_package_data_home()
        df = fetch_data(filename=str(data_home / "demo10.csv")).iloc[:500]

        config = ConfigMulti(
            predict_size=12,
            targets=["A"],
            lags_consider=[1, 2, 3],
            window_size=4,
            number_folds=2,
            use_exogenous_features=False,
            use_outlier_detection=False,
            auto_save_models=False,
            verbose=False,
        )

        forecast = run(
            config,
            task="lazy",
            dataframe=df,
            project_name="demo10_run",
            cache_home=tempfile.mkdtemp(),
        )
        print("columns:", list(forecast.columns))
        print("rows:", len(forecast))
        forecast.head()
        ```

        Remove all cached models and artefacts for a project (``"clean"``
        task).  Returns an empty DataFrame:

        ```{python}
        result = run(task="clean", project_name="demo10_run", cache_home=tempfile.mkdtemp())
        print("empty:", result.empty)
        ```
    """
    return run_with(
        multitask_cls=MultiTask,
        all_tasks=_ALL_TASKS,
        unknown_task_message=_unknown_task_message,
        config=config,
        task=task,
        dataframe=dataframe,
        data_test=data_test,
        project_name=project_name,
        cache_home=cache_home,
        plot_with_outliers=plot_with_outliers,
        show=show,
        show_progress=show_progress,
        dry_run=dry_run,
        log_level=log_level,
        **overrides,
    )
