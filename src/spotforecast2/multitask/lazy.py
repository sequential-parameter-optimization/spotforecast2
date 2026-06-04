# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lazy-fitting task — Task 1.

Re-exports ``execute_lazy`` from the safe package.  Redefines ``LazyTask``
as a thin subclass of the sf2 ``BaseTask`` (which already carries plotting
support via ``PlottingMixin``) and restores the historical ``show=True``
default for this package.
"""

from typing import Any, Dict, Optional

# Re-export execute_lazy from the safe package so that importers of
# spotforecast2.multitask.lazy.execute_lazy continue to work.
from spotforecast2_safe.multitask.lazy import execute_lazy  # noqa: F401

from spotforecast2.multitask.base import BaseTask

__all__ = ["execute_lazy", "LazyTask"]


class LazyTask(BaseTask):
    """Task 1 — Lazy Fitting with default LightGBM parameters.

    Creates an unfitted forecaster per target and fits with default
    hyperparameters.  No cross-validation or tuning is performed.

    When cached tuning results are available (saved by ``OptunaTask`` or
    ``SpotOptimTask``), they are loaded and applied automatically so that the
    lazy task benefits from prior tuning without re-running the search.

    Examples:
        ```{python}
        from spotforecast2.multitask import LazyTask

        task = LazyTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "lazy"

    def run(
        self,
        show: bool = True,
        use_tuned_params: bool = True,
        max_age_days: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run lazy fitting for all targets.

        Args:
            show: If ``True``, display prediction figures.
            use_tuned_params: If ``True``, load and apply cached tuning
                results for each target.
            max_age_days: Maximum age in days for cached tuning results.
                ``None`` accepts any age.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["lazy"]``.
        """
        return execute_lazy(
            self,
            show=show,
            use_tuned_params=use_tuned_params,
            max_age_days=max_age_days,
        )
