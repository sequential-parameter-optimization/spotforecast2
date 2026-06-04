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
    PipelineConfig,
    agg_predictor,
)

from spotforecast2.plots.plotter import (
    make_plot,
    plot_with_outliers as _plot_with_outliers,
)

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
    """

    def plot_with_outliers(self) -> None:
        """Visualise original vs. cleaned data with outlier markers.

        Raises:
            RuntimeError: If ``detect_outliers`` has not been called.
        """
        if self.df_pipeline_original is None:  # type: ignore[attr-defined]
            raise RuntimeError("Call detect_outliers() before plot_with_outliers().")
        _plot_with_outliers(
            df_pipeline=self.df_pipeline,  # type: ignore[attr-defined]
            df_pipeline_original=self.df_pipeline_original,  # type: ignore[attr-defined]
            config=self.config,  # type: ignore[attr-defined]
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
                f"Targets {self.config.targets} ({task_name})"  # type: ignore[attr-defined]
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
            markers via ``spotforecast2.plots.plotter.plot_with_outliers``.
        _show_prediction_figure: Calls ``make_plot`` and shows the figure
            interactively.
        _show_prediction_figure_agg: Same for the aggregated prediction.
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
        if len(self.config.targets) == 1:
            target = self.config.targets[0]
            agg_pkg = results[target]
            self.agg_results[task_name] = agg_pkg
            return agg_pkg

        if self.config.agg_weights is not None:
            active_weights = self.config.agg_weights[: len(self.config.targets)]
        else:
            n = len(self.config.targets)
            active_weights = [1.0 / n] * n
            self.logger.info(
                "No agg_weights configured — using equal weights (1/%d each).", n
            )

        agg_pkg = self.agg_predictor(
            results=results,
            targets=self.config.targets,
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
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run(). "
            "Use LazyTask, OptunaTask, SpotOptimTask, PredictTask, or CleanTask."
        )
