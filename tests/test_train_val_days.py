# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for the config-owned training/validation window and persistence
policy fields after the config-object refactor.

Covers, on ``ConfigMulti`` directly and on the task wrappers:
- Default values of ``train_size``, ``delta_val``, ``auto_save_models``
- Custom values flow through to the constructed task and its ``self.config``
- ``MultiTask`` propagates ``**overrides`` via ``config.set_params``
- ``cv_ts`` returns a fixed-window splitter when ``config.train_size`` is set
"""

import pandas as pd
import pytest
from spotforecast2_safe.configurator.config_multi import ConfigMulti

from spotforecast2.multitask import (
    LazyTask,
    MultiTask,
    OptunaTask,
    SpotOptimTask,
)

_ALL_CLASSES = [LazyTask, OptunaTask, SpotOptimTask, MultiTask]
_ALL_IDS = ["LazyTask", "OptunaTask", "SpotOptimTask", "MultiTask"]


def _make(cls, **config_overrides):
    """Instantiate *cls* with a ``ConfigMulti`` carrying the supplied overrides."""
    return cls(ConfigMulti(**config_overrides))


class TestDefaults:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_train_size_default(self, cls):
        t = _make(cls)
        assert t.config.train_size == pd.Timedelta(days=3 * 365)

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_delta_val_default(self, cls):
        t = _make(cls)
        assert t.config.delta_val == pd.Timedelta(hours=24 * 7 * 10)

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_auto_save_models_default(self, cls):
        t = _make(cls)
        assert t.config.auto_save_models is True


class TestTrainSize:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_custom_train_size_stored(self, cls):
        t = _make(cls, train_size=pd.Timedelta(days=180))
        assert t.config.train_size == pd.Timedelta(days=180)

    def test_multitask_overrides_path(self):
        mt = MultiTask(train_size=pd.Timedelta(days=90))
        assert mt.config.train_size == pd.Timedelta(days=90)


class TestDeltaVal:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_custom_delta_val_stored(self, cls):
        t = _make(cls, delta_val=pd.Timedelta(days=14))
        assert t.config.delta_val == pd.Timedelta(days=14)

    def test_multitask_overrides_path(self):
        mt = MultiTask(delta_val=pd.Timedelta(days=21))
        assert mt.config.delta_val == pd.Timedelta(days=21)


class TestAutoSaveModels:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_auto_save_models_false(self, cls):
        t = _make(cls, auto_save_models=False)
        assert t.config.auto_save_models is False

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_auto_save_models_true(self, cls):
        t = _make(cls, auto_save_models=True)
        assert t.config.auto_save_models is True

    def test_multitask_overrides_false(self):
        mt = MultiTask(auto_save_models=False)
        assert mt.config.auto_save_models is False

    def test_multitask_overrides_true(self):
        mt = MultiTask(auto_save_models=True)
        assert mt.config.auto_save_models is True


class TestFixedVsSlidingWindow:
    def test_config_train_size_is_timedelta_by_default(self):
        t = LazyTask()
        assert isinstance(t.config.train_size, pd.Timedelta)

    def test_config_train_size_can_be_overridden_to_none(self):
        t = LazyTask()
        t.config.train_size = None
        assert t.config.train_size is None

    def test_cv_ts_fixed_train_size_true_when_train_size_set(self):
        t = LazyTask(ConfigMulti(train_size=pd.Timedelta(days=365)))
        t.run_state.end_train_ts = pd.Timestamp("2025-01-01", tz="UTC")
        n = 4000
        idx = pd.date_range(end=t.run_state.end_train_ts, periods=n, freq="h", tz="UTC")
        y = pd.Series(range(n), index=idx, dtype=float)
        cv = t.cv_ts(y)
        assert cv.fixed_train_size is True
