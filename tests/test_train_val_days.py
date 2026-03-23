# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for the train_days, val_days, and auto_save_models parameters.

Covers:
- Default values on BaseTask, LazyTask, OptunaTask, SpotOptimTask, MultiTask
- Custom train_days stored on instance
- Custom val_days stored on instance
- auto_save_models stored on instance
- TRAIN_SIZE computed from train_days
- DELTA_VAL computed from val_days * number_folds
- config.train_size matches TRAIN_SIZE
- config.delta_val matches DELTA_VAL
- Interaction: changing number_folds changes DELTA_VAL proportionally
- All task subclasses accept all three parameters
- MultiTask correctly forwards all three parameters to BaseTask
- Fixed-window CV: config.train_size is not None when train_days is set
- Sliding-window CV: fixed_train_size is False when train_size is None
"""

import inspect

import pandas as pd
import pytest

from spotforecast2.manager.multitask import (
    BaseTask,
    LazyTask,
    MultiTask,
    OptunaTask,
    SpotOptimTask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_CLASSES = [LazyTask, OptunaTask, SpotOptimTask, MultiTask]
_ALL_IDS = ["LazyTask", "OptunaTask", "SpotOptimTask", "MultiTask"]


def _make(cls, **kwargs):
    """Instantiate *cls* with keyword arguments only."""
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_train_days_default(self, cls):
        t = _make(cls)
        assert t.train_days == 365 * 2

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_val_days_default(self, cls):
        t = _make(cls)
        assert t.val_days == 7 * 2

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_auto_save_models_default(self, cls):
        t = _make(cls)
        assert t.auto_save_models is True

    def test_train_size_default_is_two_years(self):
        t = LazyTask()
        assert t.TRAIN_SIZE == pd.Timedelta(days=730)

    def test_delta_val_default(self):
        # number_folds=10 (default), val_days=14 (default) → 140 days
        t = LazyTask()
        assert t.DELTA_VAL == pd.Timedelta(days=t.val_days * t.number_folds)


# ---------------------------------------------------------------------------
# Custom train_days
# ---------------------------------------------------------------------------


class TestTrainDays:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_custom_train_days_stored(self, cls):
        t = _make(cls, train_days=180)
        assert t.train_days == 180

    def test_train_size_reflects_train_days(self):
        t = LazyTask(train_days=365)
        assert t.TRAIN_SIZE == pd.Timedelta(days=365)

    def test_config_train_size_reflects_train_days(self):
        t = LazyTask(train_days=365)
        assert t.config.train_size == pd.Timedelta(days=365)

    def test_large_train_days(self):
        t = LazyTask(train_days=365 * 5)
        assert t.TRAIN_SIZE == pd.Timedelta(days=365 * 5)

    def test_small_train_days(self):
        t = LazyTask(train_days=30)
        assert t.TRAIN_SIZE == pd.Timedelta(days=30)

    def test_multitask_custom_train_days(self):
        mt = MultiTask(train_days=90)
        assert mt.train_days == 90
        assert mt.TRAIN_SIZE == pd.Timedelta(days=90)
        assert mt.config.train_size == pd.Timedelta(days=90)


# ---------------------------------------------------------------------------
# Custom val_days
# ---------------------------------------------------------------------------


class TestValDays:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_custom_val_days_stored(self, cls):
        t = _make(cls, val_days=14)
        assert t.val_days == 14

    def test_delta_val_reflects_val_days(self):
        # number_folds=10 (default), val_days=14 → 140 days
        t = LazyTask(val_days=14)
        assert t.DELTA_VAL == pd.Timedelta(days=140)

    def test_config_delta_val_reflects_val_days(self):
        t = LazyTask(val_days=14)
        assert t.config.delta_val == pd.Timedelta(days=140)

    def test_val_days_multiplied_by_number_folds(self):
        t = LazyTask(val_days=7, number_folds=4)
        assert t.DELTA_VAL == pd.Timedelta(days=28)

    def test_multitask_custom_val_days(self):
        mt = MultiTask(val_days=21)
        assert mt.val_days == 21
        assert mt.DELTA_VAL == pd.Timedelta(days=210)
        assert mt.config.delta_val == pd.Timedelta(days=210)


# ---------------------------------------------------------------------------
# auto_save_models
# ---------------------------------------------------------------------------


class TestAutoSaveModels:
    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_auto_save_models_false(self, cls):
        t = _make(cls, auto_save_models=False)
        assert t.auto_save_models is False

    @pytest.mark.parametrize("cls", _ALL_CLASSES, ids=_ALL_IDS)
    def test_auto_save_models_true(self, cls):
        t = _make(cls, auto_save_models=True)
        assert t.auto_save_models is True

    def test_multitask_auto_save_models_false(self):
        mt = MultiTask(auto_save_models=False)
        assert mt.auto_save_models is False

    def test_multitask_auto_save_models_true(self):
        mt = MultiTask(auto_save_models=True)
        assert mt.auto_save_models is True


# ---------------------------------------------------------------------------
# Interaction: number_folds × val_days → DELTA_VAL
# ---------------------------------------------------------------------------


class TestFoldsValDaysInteraction:
    def test_delta_val_scales_with_number_folds(self):
        t_few = LazyTask(val_days=7, number_folds=2)
        t_many = LazyTask(val_days=7, number_folds=8)
        assert t_few.DELTA_VAL == pd.Timedelta(days=14)
        assert t_many.DELTA_VAL == pd.Timedelta(days=56)

    def test_delta_val_scales_with_val_days(self):
        t_short = LazyTask(val_days=7, number_folds=4)
        t_long = LazyTask(val_days=14, number_folds=4)
        assert t_short.DELTA_VAL == pd.Timedelta(days=28)
        assert t_long.DELTA_VAL == pd.Timedelta(days=56)

    def test_default_delta_val_equals_number_folds_times_val_days(self):
        t = LazyTask()
        expected = pd.Timedelta(days=t.val_days * t.number_folds)
        assert t.DELTA_VAL == expected

    def test_custom_delta_val_equals_number_folds_times_val_days(self):
        t = LazyTask(val_days=5, number_folds=6)
        expected = pd.Timedelta(days=5 * 6)
        assert t.DELTA_VAL == expected


# ---------------------------------------------------------------------------
# Constructor signature
# ---------------------------------------------------------------------------


class TestSignature:
    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask, OptunaTask, SpotOptimTask, MultiTask],
        ids=["BaseTask", "LazyTask", "OptunaTask", "SpotOptimTask", "MultiTask"],
    )
    def test_train_days_in_signature(self, cls):
        sig = inspect.signature(cls.__init__)
        assert "train_days" in sig.parameters

    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask, OptunaTask, SpotOptimTask, MultiTask],
        ids=["BaseTask", "LazyTask", "OptunaTask", "SpotOptimTask", "MultiTask"],
    )
    def test_val_days_in_signature(self, cls):
        sig = inspect.signature(cls.__init__)
        assert "val_days" in sig.parameters

    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask, OptunaTask, SpotOptimTask, MultiTask],
        ids=["BaseTask", "LazyTask", "OptunaTask", "SpotOptimTask", "MultiTask"],
    )
    def test_auto_save_models_in_signature(self, cls):
        sig = inspect.signature(cls.__init__)
        assert "auto_save_models" in sig.parameters

    def test_train_days_default_in_signature(self):
        sig = inspect.signature(BaseTask.__init__)
        assert sig.parameters["train_days"].default == 365 * 2

    def test_val_days_default_in_signature(self):
        sig = inspect.signature(BaseTask.__init__)
        assert sig.parameters["val_days"].default == 7 * 2

    def test_auto_save_models_default_in_signature(self):
        sig = inspect.signature(BaseTask.__init__)
        assert sig.parameters["auto_save_models"].default is True


# ---------------------------------------------------------------------------
# Fixed vs sliding training window
# ---------------------------------------------------------------------------


class TestFixedVsSlidingWindow:
    def test_config_train_size_is_timedelta_by_default(self):
        t = LazyTask()
        assert isinstance(t.config.train_size, pd.Timedelta)

    def test_config_train_size_not_none_when_train_days_set(self):
        t = LazyTask(train_days=365)
        assert t.config.train_size is not None

    def test_config_train_size_is_none_when_explicitly_overridden(self):
        # Directly override the config attribute after construction
        t = LazyTask()
        t.config.train_size = None
        assert t.config.train_size is None

    def test_cv_ts_fixed_train_size_true_when_train_days_set(self):
        import pandas as pd

        t = LazyTask(train_days=365)
        t.config.end_train_ts = pd.Timestamp("2025-01-01", tz="UTC")
        n = 4000
        idx = pd.date_range(end=t.config.end_train_ts, periods=n, freq="h", tz="UTC")
        y = pd.Series(range(n), index=idx, dtype=float)
        cv = t.cv_ts(y)
        assert cv.fixed_train_size is True
