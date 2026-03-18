# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Full-featured forecasting model classes with Bayesian tuning and SHAP.

This package extends the stub implementations in ``spotforecast2-safe``
with real Bayesian hyperparameter search (Optuna) and SHAP-based
feature importance (``shap.TreeExplainer``).

Classes:
    ForecasterRecursiveModelFull: Base model with Bayesian tuning and SHAP.
    ForecasterRecursiveLGBMFull: LightGBM model with tuning and SHAP.
    ForecasterRecursiveXGBFull: XGBoost model with tuning and SHAP.
"""

from spotforecast2.manager.models.forecaster_recursive_model_full import (
    ForecasterRecursiveModelFull,
)
from spotforecast2.manager.models.forecaster_recursive_lgbm_full import (
    ForecasterRecursiveLGBMFull,
)
from spotforecast2.manager.models.forecaster_recursive_xgb_full import (
    ForecasterRecursiveXGBFull,
)

__all__ = [
    "ForecasterRecursiveModelFull",
    "ForecasterRecursiveLGBMFull",
    "ForecasterRecursiveXGBFull",
]
