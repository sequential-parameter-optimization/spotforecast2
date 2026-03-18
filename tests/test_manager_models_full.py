# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for the manager.models package.

Covers:
- Import paths (package and top-level manager)
- Default parameter values
- Custom initialization
- MRO / inheritance correctness
- from_config classmethod
- get_global_shap_feature_importance returns empty Series when untuned
"""

import pandas as pd

from spotforecast2.manager.models.forecaster_recursive_model_full import (
    ForecasterRecursiveModelFull,
)
from spotforecast2.manager.models.forecaster_recursive_lgbm_full import (
    ForecasterRecursiveLGBMFull,
)
from spotforecast2.manager.models.forecaster_recursive_xgb_full import (
    ForecasterRecursiveXGBFull,
)

# Package-level aliases — used by TestImports to verify re-export paths
from spotforecast2.manager.models import (
    ForecasterRecursiveModelFull as _MFull,
    ForecasterRecursiveLGBMFull as _LFull,
    ForecasterRecursiveXGBFull as _XFull,
)
from spotforecast2.manager import (
    ForecasterRecursiveModelFull as _MFullMgr,
    ForecasterRecursiveLGBMFull as _LFullMgr,
    ForecasterRecursiveXGBFull as _XFullMgr,
)
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)
from spotforecast2_safe.manager.models.forecaster_recursive_lgbm import (
    ForecasterRecursiveLGBM,
)
from spotforecast2_safe.manager.models.forecaster_recursive_xgb import (
    ForecasterRecursiveXGB,
)

# ---------------------------------------------------------------------------
# Import correctness
# ---------------------------------------------------------------------------


class TestImports:
    """Verify that all import paths resolve to the same classes."""

    def test_package_models_model_full(self):
        assert ForecasterRecursiveModelFull is _MFull

    def test_package_models_lgbm_full(self):
        assert ForecasterRecursiveLGBMFull is _LFull

    def test_package_models_xgb_full(self):
        assert ForecasterRecursiveXGBFull is _XFull

    def test_manager_model_full(self):
        assert ForecasterRecursiveModelFull is _MFullMgr

    def test_manager_lgbm_full(self):
        assert ForecasterRecursiveLGBMFull is _LFullMgr

    def test_manager_xgb_full(self):
        assert ForecasterRecursiveXGBFull is _XFullMgr


# ---------------------------------------------------------------------------
# ForecasterRecursiveModelFull defaults
# ---------------------------------------------------------------------------


class TestForecasterRecursiveModelFullDefaults:
    """Verify default values of ForecasterRecursiveModelFull."""

    def test_iteration(self):
        assert ForecasterRecursiveModelFull(iteration=0).iteration == 0

    def test_n_trials_default(self):
        assert ForecasterRecursiveModelFull(iteration=0).n_trials == 10

    def test_custom_n_trials(self):
        assert ForecasterRecursiveModelFull(iteration=0, n_trials=50).n_trials == 50

    def test_has_tune_method(self):
        assert callable(ForecasterRecursiveModelFull(iteration=0).tune)

    def test_has_shap_method(self):
        assert callable(
            ForecasterRecursiveModelFull(iteration=0).get_global_shap_feature_importance
        )

    def test_inherits_from_base(self):
        assert issubclass(ForecasterRecursiveModelFull, ForecasterRecursiveModel)


# ---------------------------------------------------------------------------
# ForecasterRecursiveLGBMFull
# ---------------------------------------------------------------------------


class TestForecasterRecursiveLGBMFull:
    """Verify ForecasterRecursiveLGBMFull construction and MRO."""

    def test_name(self):
        assert ForecasterRecursiveLGBMFull(iteration=0).name == "lgbm"

    def test_forecaster_not_none(self):
        assert ForecasterRecursiveLGBMFull(iteration=0).forecaster is not None

    def test_n_trials_default(self):
        assert ForecasterRecursiveLGBMFull(iteration=0).n_trials == 10

    def test_custom_n_trials(self):
        assert ForecasterRecursiveLGBMFull(iteration=0, n_trials=25).n_trials == 25

    def test_iteration(self):
        assert ForecasterRecursiveLGBMFull(iteration=3).iteration == 3

    def test_inherits_model_full(self):
        assert issubclass(ForecasterRecursiveLGBMFull, ForecasterRecursiveModelFull)

    def test_inherits_lgbm(self):
        assert issubclass(ForecasterRecursiveLGBMFull, ForecasterRecursiveLGBM)

    def test_inherits_base_model(self):
        assert issubclass(ForecasterRecursiveLGBMFull, ForecasterRecursiveModel)

    def test_tune_resolves_from_model_full(self):
        # tune() must come from ForecasterRecursiveModelFull, not the stub
        mro_names = [c.__name__ for c in ForecasterRecursiveLGBMFull.__mro__]
        full_idx = mro_names.index("ForecasterRecursiveModelFull")
        lgbm_idx = mro_names.index("ForecasterRecursiveLGBM")
        assert full_idx < lgbm_idx

    def test_shap_untuned_returns_empty_series(self):
        from unittest.mock import patch

        model = ForecasterRecursiveLGBMFull(iteration=0)
        dummy_X = pd.DataFrame({"a": [1.0, 2.0]})
        dummy_y = pd.Series([1.0, 2.0])
        with patch.object(model, "_get_training_data", return_value=(dummy_X, dummy_y)):
            result = model.get_global_shap_feature_importance()
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_custom_lags(self):
        model = ForecasterRecursiveLGBMFull(iteration=0, lags=24)
        assert len(model.forecaster.lags) == 24

    def test_predict_size_custom(self):
        model = ForecasterRecursiveLGBMFull(iteration=0, predict_size=48)
        assert model.predict_size == 48

    def test_random_state_custom(self):
        model = ForecasterRecursiveLGBMFull(iteration=0, random_state=42)
        assert model.random_state == 42


# ---------------------------------------------------------------------------
# ForecasterRecursiveXGBFull
# ---------------------------------------------------------------------------


class TestForecasterRecursiveXGBFull:
    """Verify ForecasterRecursiveXGBFull construction and MRO."""

    def test_name(self):
        assert ForecasterRecursiveXGBFull(iteration=0).name == "xgb"

    def test_forecaster_not_none(self):
        assert ForecasterRecursiveXGBFull(iteration=0).forecaster is not None

    def test_n_trials_default(self):
        assert ForecasterRecursiveXGBFull(iteration=0).n_trials == 10

    def test_custom_n_trials(self):
        assert ForecasterRecursiveXGBFull(iteration=0, n_trials=5).n_trials == 5

    def test_iteration(self):
        assert ForecasterRecursiveXGBFull(iteration=2).iteration == 2

    def test_inherits_model_full(self):
        assert issubclass(ForecasterRecursiveXGBFull, ForecasterRecursiveModelFull)

    def test_inherits_xgb(self):
        assert issubclass(ForecasterRecursiveXGBFull, ForecasterRecursiveXGB)

    def test_inherits_base_model(self):
        assert issubclass(ForecasterRecursiveXGBFull, ForecasterRecursiveModel)

    def test_tune_resolves_from_model_full(self):
        mro_names = [c.__name__ for c in ForecasterRecursiveXGBFull.__mro__]
        full_idx = mro_names.index("ForecasterRecursiveModelFull")
        xgb_idx = mro_names.index("ForecasterRecursiveXGB")
        assert full_idx < xgb_idx

    def test_shap_untuned_returns_empty_series(self):
        from unittest.mock import patch

        model = ForecasterRecursiveXGBFull(iteration=0)
        dummy_X = pd.DataFrame({"a": [1.0, 2.0]})
        dummy_y = pd.Series([1.0, 2.0])
        with patch.object(model, "_get_training_data", return_value=(dummy_X, dummy_y)):
            result = model.get_global_shap_feature_importance()
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_predict_size_custom(self):
        model = ForecasterRecursiveXGBFull(iteration=0, predict_size=12)
        assert model.predict_size == 12

    def test_random_state_custom(self):
        model = ForecasterRecursiveXGBFull(iteration=0, random_state=99)
        assert model.random_state == 99


# ---------------------------------------------------------------------------
# from_config classmethod
# ---------------------------------------------------------------------------


class TestFromConfig:
    """Verify from_config works on Full model subclasses."""

    def test_lgbm_from_config_returns_instance(self):
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        cfg = ConfigMulti()
        model = ForecasterRecursiveLGBMFull.from_config(iteration=1, config=cfg)
        assert isinstance(model, ForecasterRecursiveLGBMFull)

    def test_lgbm_from_config_predict_size(self):
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        cfg = ConfigMulti(predict_size=48)
        model = ForecasterRecursiveLGBMFull.from_config(iteration=0, config=cfg)
        assert model.predict_size == 48

    def test_lgbm_from_config_n_trials_override(self):
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        cfg = ConfigMulti()
        model = ForecasterRecursiveLGBMFull.from_config(
            iteration=0, config=cfg, n_trials=5
        )
        assert model.n_trials == 5

    def test_xgb_from_config_returns_instance(self):
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        cfg = ConfigMulti()
        model = ForecasterRecursiveXGBFull.from_config(iteration=0, config=cfg)
        assert isinstance(model, ForecasterRecursiveXGBFull)

    def test_xgb_from_config_random_state(self):
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        cfg = ConfigMulti(random_state=7)
        model = ForecasterRecursiveXGBFull.from_config(iteration=0, config=cfg)
        assert model.random_state == 7
