# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Defaults task — Task 2.

Re-exports ``execute_defaults`` from the safe package.  Redefines ``DefaultsTask``
as a thin subclass of the sf2 ``BaseTask`` (which carries plotting support)
and restores the historical ``show=True`` default for this package.
"""

from typing import Any, Dict

# Re-export execute_defaults from the safe package.
from spotforecast2_safe.multitask.defaults import execute_defaults  # noqa: F401

from spotforecast2.multitask.base import BaseTask


class DefaultsTask(BaseTask):
    """Task 2 — Defaults fitting (no tuning, no cached params).

    Creates an unfitted forecaster per target via ``config.forecaster_factory``
    (or the package default) and fits with whatever parameters that factory
    chooses.  Unlike ``LazyTask``, never reads the tuning-result cache.

    Examples:
        ```{python}
        from spotforecast2.multitask import DefaultsTask

        task = DefaultsTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "defaults"

    def run(
        self,
        show: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run defaults fitting for all targets.

        Args:
            show: If ``True``, display prediction figures.
            **kwargs: Forwarded for compatibility with ``BaseTask.run``;
                ``DefaultsTask`` does not consume any extra parameters.

        Returns:
            Aggregated prediction package.  Per-target packages are stored
            on ``self.results["defaults"]``.
        """
        del kwargs  # DefaultsTask has no tuning- or cache-related parameters
        return execute_defaults(self, show=show)
