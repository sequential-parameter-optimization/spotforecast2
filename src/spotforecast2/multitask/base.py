# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Base class for multi-target forecasting pipeline tasks.

Collapses the duplicated pipeline code onto ``spotforecast2_safe.multitask.base``
by inheriting from it.  This module defines ``PlottingMixin``, which overrides
the three visualisation hooks with live Plotly calls, and assembles the public
``BaseTask`` as ``class BaseTask(PlottingMixin, SafeBaseTask)``.

``PipelineConfig`` and ``agg_predictor`` are re-exported from the safe base so
that existing importers of ``spotforecast2.multitask.base`` remain unaffected.

Task subclasses — ``LazyTask``, ``OptunaTask``, ``SpotOptimTask``,
``PredictTask``, ``CleanTask`` — each implement ``run()`` with the logic
appropriate for their task mode (lazy, optuna, spotoptim, predict, clean).
"""

from pathlib import Path
from typing import Any, Dict, Optional

from spotforecast2_safe.multitask.base import (  # noqa: F401  (re-exported)
    BaseTask as SafeBaseTask,
)
from spotforecast2_safe.multitask.base import (
    PipelineConfig,
    agg_predictor,
)

from spotforecast2.plots.plotter import make_plot
from spotforecast2.plots.plotter import plot_with_outliers as _plot_with_outliers

__all__ = [
    "SafeBaseTask",
    "PipelineConfig",
    "agg_predictor",
    "PlottingMixin",
    "BaseTask",
]


class PlottingMixin:
    """Mixin that wires the three visualisation hooks to live Plotly figures.

    Intended to be mixed in before ``SafeBaseTask`` in the MRO so that its
    method definitions take precedence over the no-op stubs in the safe base.

    Methods:
        plot_with_outliers: Display original vs. cleaned data with outlier markers.
        _show_prediction_figure: Show an interactive per-target prediction figure.
        _show_prediction_figure_agg: Show an interactive aggregated prediction figure.

    Examples:
        ```{python}
        import tempfile
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        from spotforecast2.multitask import LazyTask
        from spotforecast2.multitask.base import PlottingMixin

        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
        df = pd.DataFrame({"load": rng.normal(100, 10, len(idx))}, index=idx)
        df.index.name = "DateTime"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(
                predict_size=6,
                use_exogenous_features=False,
                use_outlier_detection=False,
                auto_save_models=False,
                cache_home=tmp,
            )
            task = LazyTask(cfg, dataframe=df)

        # LazyTask inherits from BaseTask(PlottingMixin, SafeBaseTask), so
        # PlottingMixin.plot_with_outliers overrides the safe-base no-op stub.
        print("PlottingMixin in MRO:", PlottingMixin in type(task).__mro__)
        print(
            "plot_with_outliers wired to PlottingMixin:",
            type(task).plot_with_outliers is PlottingMixin.plot_with_outliers,
        )
        assert PlottingMixin in type(task).__mro__
        assert type(task).plot_with_outliers is PlottingMixin.plot_with_outliers
        ```
    """

    def plot_with_outliers(self) -> None:
        """Visualise original vs. cleaned data with outlier markers.

        Raises:
            RuntimeError: If ``detect_outliers`` has not been called.

        Examples:
            ```{python}
            import tempfile
            import numpy as np
            import pandas as pd
            from spotforecast2_safe.configurator.config_multi import ConfigMulti
            from spotforecast2.multitask import LazyTask

            rng = np.random.default_rng(0)
            idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
            df = pd.DataFrame({"load": rng.normal(100, 10, len(idx))}, index=idx)
            df.index.name = "DateTime"

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(
                    predict_size=6,
                    use_exogenous_features=False,
                    use_outlier_detection=False,
                    bounds=[(50, 150)],
                    auto_save_models=False,
                    cache_home=tmp,
                )
                task = LazyTask(cfg, dataframe=df)
                task.prepare_data().detect_outliers()
                task.plot_with_outliers()
            ```
        """
        if self.df_pipeline_original is None:  # type: ignore[attr-defined]
            raise RuntimeError("Call detect_outliers() before plot_with_outliers().")
        _plot_with_outliers(
            df_pipeline=self.df_pipeline,  # type: ignore[attr-defined]
            df_pipeline_original=self.df_pipeline_original,  # type: ignore[attr-defined]
            config=self.config,  # type: ignore[attr-defined]
            targets=self.run_state.targets,  # type: ignore[attr-defined]
        )

    def _show_prediction_figure(
        self,
        pred_pkg: Dict[str, Any],
        target: str,
        task_name: str,
    ) -> None:
        """Display an interactive prediction figure for one target.

        Args:
            pred_pkg: Prediction package for this target.
            target: Target column name.
            task_name: Human-readable task label for the figure title.
        """
        fig = make_plot(
            pred_pkg,
            title=f"Prediction for Target '{target}' ({task_name})",
            save=False,
        )
        fig.show()

    def _show_prediction_figure_agg(
        self,
        agg_pkg: Dict[str, Any],
        task_name: str,
    ) -> None:
        """Display an interactive aggregated prediction figure.

        Args:
            agg_pkg: Aggregated prediction package.
            task_name: Human-readable task label for the figure title.
        """
        fig = make_plot(
            agg_pkg,
            title=(
                f"Aggregated Forecast: Weighted Combination of "
                f"Targets {self.run_state.targets} ({task_name})"  # type: ignore[attr-defined]
            ),
            save=False,
        )
        fig.show()


class BaseTask(PlottingMixin, SafeBaseTask):
    """Shared base for all multi-target forecasting pipeline tasks.

    Inherits the complete data-preparation pipeline (steps 1-7) and all
    helper methods from ``spotforecast2_safe.multitask.base.BaseTask``.
    ``PlottingMixin`` overrides the three visualisation hooks with live
    Plotly figures.

    The public API — constructor signature, attributes, method names, and
    the ``PipelineConfig`` protocol — is identical to the safe-package base.
    See ``spotforecast2_safe.multitask.base.BaseTask`` for full documentation.

    Task subclasses implement ``run()`` for their specific mode:
    ``LazyTask`` (lazy), ``OptunaTask`` (optuna), ``SpotOptimTask``
    (spotoptim), ``PredictTask`` (predict), ``CleanTask`` (clean).

    Visualisation additions over the safe base:
        plot_with_outliers: Renders original vs. cleaned data with outlier
            markers via `spotforecast2.plots.plotter.plot_with_outliers`.
        _show_prediction_figure: Calls `make_plot` and shows the figure
            interactively.
        _show_prediction_figure_agg: Same for the aggregated prediction.

    Examples:
        ```{python}
        import tempfile
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.configurator.config_multi import ConfigMulti
        from spotforecast2.multitask.base import BaseTask, PlottingMixin

        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
        df = pd.DataFrame({"load": rng.normal(100, 10, len(idx))}, index=idx)
        df.index.name = "DateTime"

        with tempfile.TemporaryDirectory() as tmp:
            cfg = ConfigMulti(
                predict_size=6,
                use_exogenous_features=False,
                use_outlier_detection=False,
                auto_save_models=False,
                cache_home=tmp,
            )
            task = BaseTask(cfg, dataframe=df)
            # Data-preparation pipeline (steps 1-3)
            task.prepare_data().detect_outliers().impute()

        print("Pipeline shape:", task.df_pipeline.shape)
        print("Targets:", task.config.targets)
        # PlottingMixin is in the MRO — visualisation hooks are live Plotly calls.
        print("PlottingMixin in MRO:", PlottingMixin in type(task).__mro__)
        assert task.df_pipeline.shape[1] == 1
        assert PlottingMixin in type(task).__mro__
        ```
    """

    # ``_show_prediction_figure`` and ``_show_prediction_figure_agg`` are
    # called from ``SafeBaseTask._run_strategy`` and ``_aggregate_and_show``
    # respectively; they are resolved from ``PlottingMixin`` first in the MRO.

    # Override _aggregate_and_show to restore show=True default and
    # to call _show_prediction_figure_agg rather than the stub.
    def _aggregate_and_show(
        self,
        results: Dict[str, Any],
        task_name: str,
        show: bool = True,
    ) -> Dict[str, Any]:
        """Aggregate results and optionally display the combined figure.

        For multi-target configurations, aggregates via ``agg_predictor``
        using either the configured ``agg_weights`` or equal weights as a
        fallback.  For single-target configurations, aggregation is a no-op:
        the per-target prediction package is returned directly.  The result
        is stored in ``agg_results`` keyed by ``task_name``.

        Args:
            results: Per-target prediction packages.
            task_name: Human-readable label used in figure titles and
                ``agg_results`` keys.
            show: If ``True``, display the aggregated figure (multi-target
                only).

        Returns:
            Aggregated prediction package dict.
        """
        if len(self.run_state.targets) == 1:
            target = self.run_state.targets[0]
            agg_pkg = results[target]
            self.agg_results[task_name] = agg_pkg
            return agg_pkg

        if self.config.agg_weights is not None:
            active_weights = self.config.agg_weights[: len(self.run_state.targets)]
        else:
            n = len(self.run_state.targets)
            active_weights = [1.0 / n] * n
            self.logger.info(
                "No agg_weights configured — using equal weights (1/%d each).", n
            )

        agg_pkg = self.agg_predictor(
            results=results,
            targets=self.run_state.targets,
            weights=active_weights,
        )
        self.agg_results[task_name] = agg_pkg
        if show:
            self._show_prediction_figure_agg(agg_pkg, task_name)
        return agg_pkg

    def run(  # noqa: PLR0913
        self,
        show: bool = True,
        task: Optional[str] = None,
        task_name: Optional[str] = None,
        use_tuned_params: bool = True,
        max_age_days: Optional[float] = None,
        search_space: Optional[Any] = None,
        dry_run: bool = False,
        cache_home: Optional[Path] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the task-specific training / tuning pipeline.

        Subclasses must override this method.

        Args:
            show: If ``True``, display prediction figures.
            task: Task mode override (used by ``MultiTask``).
            task_name: Restrict model loading to a specific source task
                (used by ``PredictTask``).
            use_tuned_params: Load cached tuning results when available
                (used by ``LazyTask``).
            max_age_days: Maximum age in days for cached results
                (used by ``LazyTask`` and ``PredictTask``).
            search_space: Hyperparameter search-space definition
                (used by ``OptunaTask`` and ``SpotOptimTask``).
            dry_run: Report what would be deleted without removing anything
                (used by ``CleanTask``).
            cache_home: Override the cache directory (used by ``CleanTask``).
            **kwargs: Additional task-specific arguments.

        Returns:
            Aggregated prediction package for the task.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.

        Examples:
            `BaseTask.run` is abstract — it raises `NotImplementedError` to
            enforce that every concrete task subclass provides its own
            implementation.  Use `LazyTask`, `OptunaTask`, `SpotOptimTask`,
            `PredictTask`, or `CleanTask` for live pipelines.

            ```{python}
            import tempfile
            import numpy as np
            import pandas as pd
            from spotforecast2_safe.configurator.config_multi import ConfigMulti
            from spotforecast2.multitask.base import BaseTask
            from spotforecast2.multitask import LazyTask

            rng = np.random.default_rng(0)
            idx = pd.date_range("2023-01-01", periods=24 * 14, freq="h", tz="UTC")
            df = pd.DataFrame({"load": rng.normal(100, 10, len(idx))}, index=idx)
            df.index.name = "DateTime"

            # BaseTask.run raises NotImplementedError — use a concrete subclass.
            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(cache_home=tmp)
                base = BaseTask(cfg)
            try:
                base.run()
            except NotImplementedError as exc:
                print("BaseTask.run() raised NotImplementedError (expected).")
                print(str(exc)[:60])

            # LazyTask overrides run() with lazy fitting logic.
            print("LazyTask.run is overridden:", LazyTask.run is not BaseTask.run)
            assert LazyTask.run is not BaseTask.run
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run(). "
            "Use LazyTask, OptunaTask, SpotOptimTask, PredictTask, or CleanTask."
        )
