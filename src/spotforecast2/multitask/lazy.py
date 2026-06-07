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

        Examples:
            ```{python}
            import tempfile
            import numpy as np
            import pandas as pd
            from lightgbm import LGBMRegressor
            from spotforecast2.multitask import LazyTask
            from spotforecast2_safe.configurator.config_multi import ConfigMulti
            from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            from spotforecast2_safe.preprocessing import RollingFeatures

            rng = np.random.default_rng(0)
            n = 24 * 14  # two weeks of hourly data
            idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
            idx.name = "DateTime"
            df = pd.DataFrame({"load": rng.normal(100, 10, n)}, index=idx)

            def _fast_factory(config, *, weight_func=None, target=None):
                return ForecasterRecursive(
                    estimator=LGBMRegressor(
                        n_estimators=10,
                        random_state=config.random_state,
                        verbose=-1,
                    ),
                    lags=6,
                    window_features=RollingFeatures(stats=["mean"], window_sizes=6),
                    weight_func=weight_func,
                )

            with tempfile.TemporaryDirectory() as tmp:
                cfg = ConfigMulti(
                    predict_size=6,
                    use_exogenous_features=False,
                    use_outlier_detection=False,
                    auto_save_models=False,
                    number_folds=2,
                    random_state=42,
                    forecaster_factory=_fast_factory,
                    cache_home=tmp,
                )
                task = LazyTask(cfg, dataframe=df)
                task.prepare_data().detect_outliers().impute().build_exogenous_features()
                result = task.run(show=False, use_tuned_params=False)

            print(f"Future predictions: {len(result['future_pred'])} steps")
            assert len(result["future_pred"]) == 6
            ```
        """
        return execute_lazy(
            self,
            show=show,
            use_tuned_params=use_tuned_params,
            max_age_days=max_age_days,
        )
