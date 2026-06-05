# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convenience runner for the MultiTask forecasting pipeline.

Provides a single ``run`` function that wraps the full pipeline
sequence (prepare_data, detect_outliers, impute, build_exogenous_features,
run) behind a one-call interface.

The orchestration body lives in ``spotforecast2_safe.multitask.runner.run_with``
and is reused here via delegation.  This module extends the safe-side task set
with the auto-tuning tasks (``"optuna"``, ``"spotoptim"``) and binds the
``spotforecast2.multitask.multi.MultiTask`` subclass that supports them.
"""

from typing import Any, List, Optional

import pandas as pd

from spotforecast2_safe.multitask.runner import (  # noqa: F401
    SAFE_PIPELINE_TASKS,
    _DEMO10_AGG_WEIGHTS,
    _DEMO10_BOUNDS,
    make_demo10_config,
    run_with,
)

from spotforecast2.multitask.base import PipelineConfig
from spotforecast2.multitask.multi import MultiTask

__all__ = [
    "SAFE_PIPELINE_TASKS",
    "_DEMO10_AGG_WEIGHTS",
    "_DEMO10_BOUNDS",
    "make_demo10_config",
    "run_with",
    "run",
]

_PIPELINE_TASKS = SAFE_PIPELINE_TASKS | frozenset({"optuna", "spotoptim"})
_ALL_TASKS = _PIPELINE_TASKS | {"clean"}


def _unknown_task_message(task: str, sorted_tasks: List[str]) -> str:
    return f"Unknown task '{task}'. Choose from: {sorted_tasks}"


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

        prepare_data → detect_outliers → impute →
        build_exogenous_features → run

    is executed and the aggregated future predictions are returned as a
    DataFrame.

    Args:
        config: A ``PipelineConfig``-conforming object (typically
            ``ConfigMulti`` or ``ConfigEntsoe``).  When ``None``, a fresh
            ``ConfigMulti()`` is constructed with default fields.  Use
            ``make_demo10_config()`` to opt in to the 11-target demo10
            ``bounds`` / ``agg_weights`` presets.
        task: Pipeline mode — one of ``"lazy"``, ``"defaults"``,
            ``"optuna"``, ``"spotoptim"``, ``"predict"``, or ``"clean"``.
            Defaults to ``"lazy"``.
        dataframe: Input time-series data.  Must contain a datetime
            column matching ``config.index_name`` and at least one numeric
            target column.  Optional for ``"clean"``, required otherwise.
        data_test: Ground-truth DataFrame covering the prediction horizon.
            When supplied, populates ``test_actual`` and ``metrics_future``
            in the prediction package.  Takes precedence over
            ``config.test_data_loader``.  Optional; ``None`` leaves the
            test-actuals series empty (today's behaviour).
        project_name: Active-dataset identifier.  Sets
            ``config.data_frame_name``, which drives cache-subdirectory
            and model-file naming.
        cache_home: Cache directory override.  When ``None``, the package
            default from ``get_cache_home()`` is used; the value is then
            written onto ``config.cache_home``.
        plot_with_outliers: Whether to render the optional
            outlier-visualisation step.
        show: Whether to display prediction figures after the task runs.
        show_progress: Whether to print progress messages during pipeline
            execution.
        dry_run: Forwarded to ``MultiTask``; only meaningful for the
            ``"clean"`` task.
        log_level: Logging level.  Defaults to 40 (ERROR).
        **overrides: Forwarded to ``config.set_params(**overrides)``.
            Mutates the caller's config object.

    Returns:
        DataFrame whose index is the forecast horizon timestamps and
        whose single column ``"forecast"`` contains the aggregated
        predicted values.  For the ``"clean"`` task an empty DataFrame is
        returned.

    Raises:
        ValueError: If ``task`` is not one of the supported task names.

    Examples:
        Run the pipeline using cached or default model parameters
        (``"lazy"`` task):

        ```{python}
        from spotforecast2.multitask.runner import run
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
        import pandas as pd
        import warnings
        warnings.filterwarnings("ignore")

        data_home = get_package_data_home()
        df = fetch_data(filename=str(data_home / "demo02.csv"))

        cfg = ConfigMulti(
            train_size=pd.Timedelta(days=365),
            predict_size=24,
            imputation_method="weighted",
            use_exogenous_features=False,
        )
        forecast = run(cfg, task="lazy", dataframe=df, project_name="demo02")
        print(forecast)
        ```

        Tune hyperparameters via Optuna Bayesian search (``"optuna"`` task):

        ```{python}
        from spotforecast2.multitask.runner import run
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
        import pandas as pd
        import warnings
        warnings.filterwarnings("ignore")

        data_home = get_package_data_home()
        df = fetch_data(filename=str(data_home / "demo02.csv"))

        cfg = ConfigMulti(
            n_trials_optuna=5,
            predict_size=24,
            train_size=pd.Timedelta(days=365),
            delta_val=pd.Timedelta(days=7 * 10),
            imputation_method="weighted",
            use_exogenous_features=False,
        )
        forecast = run(cfg, task="optuna", dataframe=df, project_name="demo02")
        print(forecast)
        ```

        Remove all cached models and artefacts for a project
        (``"clean"`` task).  Returns an empty DataFrame:

        ```{python}
        from spotforecast2.multitask.runner import run

        result = run(task="clean", project_name="demo02")
        print(result.empty)
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
