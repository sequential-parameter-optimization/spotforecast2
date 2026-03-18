# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""XGBoost forecaster with real Bayesian tuning and SHAP.

This module provides :class:`ForecasterRecursiveXGBFull`, which combines
the XGBoost forecaster from ``spotforecast2-safe`` with Bayesian
hyperparameter optimisation (Optuna) and SHAP-based feature importance
from :class:`~spotforecast2.manager.models.ForecasterRecursiveModelFull`.

Examples:
    >>> from spotforecast2.manager.models import ForecasterRecursiveXGBFull
    >>> model = ForecasterRecursiveXGBFull(iteration=0)
    >>> model.name
    'xgb'
    >>> model.forecaster is not None
    True
    >>> model.n_trials
    10
"""

from __future__ import annotations

from typing import Any

from spotforecast2_safe.manager.models.forecaster_recursive_xgb import (
    ForecasterRecursiveXGB,
)
from spotforecast2.manager.models.forecaster_recursive_model_full import (
    ForecasterRecursiveModelFull,
)


class ForecasterRecursiveXGBFull(ForecasterRecursiveModelFull, ForecasterRecursiveXGB):
    """XGBoost forecaster with real Bayesian tuning and SHAP.

    Inherits the XGBoost forecaster initialisation from
    :class:`~spotforecast2_safe.manager.models.forecaster_recursive_xgb.ForecasterRecursiveXGB`
    (``spotforecast2-safe``) and adds the real :meth:`tune` and
    :meth:`get_global_shap_feature_importance` from
    :class:`~spotforecast2.manager.models.ForecasterRecursiveModelFull`.

    The MRO ensures that :meth:`tune` and SHAP methods resolve from
    ``ForecasterRecursiveModelFull``, while the XGBoost-specific
    ``__init__`` (estimator wiring) comes from ``ForecasterRecursiveXGB``.

    Args:
        iteration: Training iteration index (0-based).
        lags: Number of lag features to use.
        **kwargs: Forwarded to parent classes (e.g., ``n_trials``,
            ``predict_size``, ``train_size``).

    Examples:
        >>> from spotforecast2.manager.models import ForecasterRecursiveXGBFull
        >>> model = ForecasterRecursiveXGBFull(iteration=0)
        >>> model.name
        'xgb'
        >>> model.forecaster is not None
        True
        >>> model.n_trials
        10
        >>> model.iteration
        0

    ```{python}
    from spotforecast2.manager.models import ForecasterRecursiveXGBFull
    model = ForecasterRecursiveXGBFull(iteration=0)
    print(f"Model name: {model.name}")
    print(f"Trials: {model.n_trials}")
    print(f"Has tune: {callable(model.tune)}")
    print(f"Has SHAP: {callable(model.get_global_shap_feature_importance)}")
    ```
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        super().__init__(iteration=iteration, lags=lags, **kwargs)
