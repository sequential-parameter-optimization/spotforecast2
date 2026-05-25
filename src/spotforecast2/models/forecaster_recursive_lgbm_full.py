# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LGBM forecaster with real Bayesian tuning and SHAP.

This module provides `ForecasterRecursiveLGBMFull`, which combines
the LightGBM forecaster from ``spotforecast2-safe`` with Bayesian
hyperparameter optimisation (Optuna) and SHAP-based feature importance
from `ForecasterRecursiveModelFull`.

Examples:
    ```{python}
    from spotforecast2.models import ForecasterRecursiveLGBMFull

    model = ForecasterRecursiveLGBMFull(iteration=0)
    assert model.name == "lgbm"
    assert model.forecaster is not None
    assert model.n_trials == 10
    print(f"name={model.name}, n_trials={model.n_trials}, iteration={model.iteration}")
    ```
"""

from __future__ import annotations

from typing import Any

from spotforecast2_safe.forecaster.wrappers import ForecasterRecursiveLGBM
from spotforecast2.models.forecaster_recursive_model_full import (
    ForecasterRecursiveModelFull,
)


class ForecasterRecursiveLGBMFull(
    ForecasterRecursiveModelFull, ForecasterRecursiveLGBM
):
    """LGBM forecaster with real Bayesian tuning and SHAP.

    Inherits the LightGBM forecaster initialisation from
    `ForecasterRecursiveLGBM`
    (``spotforecast2-safe``) and adds the real `tune()` and
    `get_global_shap_feature_importance()` from
    `ForecasterRecursiveModelFull`.

    The MRO ensures that `tune()` and SHAP methods resolve from
    ``ForecasterRecursiveModelFull``, while the LightGBM-specific
    ``__init__`` (estimator wiring) comes from ``ForecasterRecursiveLGBM``.

    Args:
        iteration: Training iteration index (0-based).
        lags: Number of lag features to use.
        **kwargs: Forwarded to parent classes (e.g., ``n_trials``,
            ``predict_size``, ``train_size``).

    Examples:
        ```{python}
        from spotforecast2.models import ForecasterRecursiveLGBMFull

        model = ForecasterRecursiveLGBMFull(iteration=0)
        assert model.name == "lgbm"
        assert model.forecaster is not None
        assert model.n_trials == 10
        assert model.iteration == 0
        print(f"name={model.name}, n_trials={model.n_trials}, iteration={model.iteration}")
        print(f"tune callable: {callable(model.tune)}")
        print(f"shap callable: {callable(model.get_global_shap_feature_importance)}")
        ```
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        super().__init__(iteration=iteration, lags=lags, **kwargs)
