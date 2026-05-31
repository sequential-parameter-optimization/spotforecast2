# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""XGBoost forecaster with real Bayesian tuning and SHAP.

This module provides `ForecasterRecursiveXGBFull`, which combines
the XGBoost forecaster from ``spotforecast2-safe`` with Bayesian
hyperparameter optimisation (Optuna) and SHAP-based feature importance
from `ForecasterRecursiveModelFull`.

Examples:
    ```{python}
    import numpy as np
    import pandas as pd
    from xgboost import XGBRegressor

    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    from spotforecast2.models import ForecasterRecursiveXGBFull

    model = ForecasterRecursiveXGBFull(iteration=0, lags=3)
    # Swap in a tiny estimator so the example renders quickly.
    model.forecaster = ForecasterRecursive(
        estimator=XGBRegressor(n_estimators=5, max_depth=2, random_state=1234),
        lags=3,
    )
    assert model.name == "xgb"
    assert model.n_trials == 10

    rng = np.random.default_rng(0)
    y = pd.Series(
        rng.random(30),
        index=pd.date_range("2023-01-01", periods=30, freq="h"),
    )
    model.fit(y=y)
    pred = model.forecaster.predict(steps=2)
    assert len(pred) == 2
    print(f"name: {model.name}")
    print(f"n_trials: {model.n_trials}")
    print(f"forecast horizon: {len(pred)} steps")
    ```
"""

from __future__ import annotations

from typing import Any

from spotforecast2_safe.forecaster.wrappers import ForecasterRecursiveXGB

from spotforecast2.models.forecaster_recursive_model_full import (
    ForecasterRecursiveModelFull,
)


class ForecasterRecursiveXGBFull(ForecasterRecursiveModelFull, ForecasterRecursiveXGB):
    """XGBoost forecaster with real Bayesian tuning and SHAP.

    Inherits the XGBoost forecaster initialisation from
    `ForecasterRecursiveXGB`
    (``spotforecast2-safe``) and adds the real `tune()` and
    `get_global_shap_feature_importance()` from
    `ForecasterRecursiveModelFull`.

    The MRO ensures that `tune()` and SHAP methods resolve from
    ``ForecasterRecursiveModelFull``, while the XGBoost-specific
    ``__init__`` (estimator wiring) comes from ``ForecasterRecursiveXGB``.

    Args:
        iteration: Training iteration index (0-based).
        lags: Number of lag features to use.
        **kwargs: Forwarded to parent classes (e.g., ``n_trials``,
            ``predict_size``, ``train_size``).

    Examples:
        ```{python}
        import numpy as np
        import pandas as pd
        from xgboost import XGBRegressor

        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2.models import ForecasterRecursiveXGBFull

        model = ForecasterRecursiveXGBFull(iteration=0, lags=3)
        # Swap in a tiny estimator so the example renders quickly.
        model.forecaster = ForecasterRecursive(
            estimator=XGBRegressor(n_estimators=5, max_depth=2, random_state=1234),
            lags=3,
        )
        assert model.name == "xgb"
        assert model.iteration == 0
        assert model.n_trials == 10
        assert callable(model.tune)
        assert callable(model.get_global_shap_feature_importance)

        rng = np.random.default_rng(0)
        y = pd.Series(
            rng.random(30),
            index=pd.date_range("2023-01-01", periods=30, freq="h"),
        )
        model.fit(y=y)
        pred = model.forecaster.predict(steps=2)
        assert len(pred) == 2
        print(f"name: {model.name}")
        print(f"iteration: {model.iteration}")
        print(f"n_trials: {model.n_trials}")
        print(f"has tune: {callable(model.tune)}")
        print(f"has SHAP: {callable(model.get_global_shap_feature_importance)}")
        print(f"forecast horizon: {len(pred)} steps")
        ```
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        super().__init__(iteration=iteration, lags=lags, **kwargs)
