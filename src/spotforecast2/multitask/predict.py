# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Predict-only task — loads previously saved models and forecasts.

Re-exports ``execute_predict`` from the safe package.  Redefines ``PredictTask``
as a thin subclass of the sf2 ``BaseTask`` (which carries plotting support)
and restores the historical ``show=True`` default for this package.
"""

from typing import Any, Dict, Optional

# Re-export execute_predict from the safe package.
from spotforecast2_safe.multitask.predict import execute_predict  # noqa: F401

from spotforecast2.multitask.base import BaseTask


class PredictTask(BaseTask):
    """Task 5 — Predict-only using previously saved models.

    Loads fitted forecasters that were persisted by a prior
    ``LazyTask``, ``OptunaTask``, or ``SpotOptimTask`` run and produces
    predictions for all configured targets.  No training or tuning is
    performed.

    If no saved models exist in the cache directory the ``run`` method
    raises ``RuntimeError`` with an informative message.

    Examples:
        ```{python}
        from spotforecast2.multitask import PredictTask

        task = PredictTask(data_frame_name="demo10", predict_size=24)
        print(f"Task: {task.TASK}")
        print(f"Predict size: {task.config.predict_size}")
        ```
    """

    _task_name = "predict"

    def run(
        self,
        show: bool = True,
        task_name: Optional[str] = None,
        max_age_days: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run prediction using previously saved models.

        Args:
            show: If ``True``, display prediction figures.
            task_name: Restrict model loading to a specific source task
                (``"lazy"``, ``"optuna"``, or ``"spotoptim"``).
                ``None`` loads the most recent model regardless of source.
            max_age_days: Maximum age in days for saved models.
                Models older than this are ignored.  ``None`` accepts
                any age.

        Returns:
            Aggregated prediction package. Per-target packages are stored
            on ``self.results["predict"]``.

        Raises:
            RuntimeError: If no saved models are found in the cache
                directory, or if a target has no matching model.
        """
        return execute_predict(
            self,
            show=show,
            task_name=task_name,
            max_age_days=max_age_days,
        )
