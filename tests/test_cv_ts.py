# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for BaseTask.cv_ts().

Covers:
- Return type is TimeSeriesFold
- steps equals config.predict_size
- refit is False
- gap is 0
- allow_incomplete_fold is True
- initial_train_size matches y_train slice up to validation boundary
- fixed_train_size is True when config.train_size is not None
- fixed_train_size is False when config.train_size is None
- cv_ts is available on all task subclasses
- calling cv_ts twice yields identical configuration
"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from spotforecast2.manager.multitask import (
    BaseTask,
    LazyTask,
    OptunaTask,
    SpotOptimTask,
)
from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_END_TRAIN = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
_FREQ = "h"
_N = 2000  # two-thousand hourly observations


def _make_task(tmp_path: Path, **kwargs) -> LazyTask:
    """Return a LazyTask whose config.end_train_ts is set without loading data.

    val_days is pinned to 7 so that DELTA_VAL = 7 * number_folds days, keeping
    the test series length requirements manageable.  The val_days parameter
    itself is exercised in test_train_val_days.py.
    """
    kwargs.setdefault("val_days", 7)
    t = LazyTask(data_frame_name="test_data", cache_home=tmp_path, **kwargs)
    t.config.end_train_ts = _END_TRAIN
    return t


def _make_y_train(end: pd.Timestamp = _END_TRAIN, n: int = _N) -> pd.Series:
    """Synthetic hourly Series ending at *end* with *n* observations."""
    idx = pd.date_range(end=end, periods=n, freq=_FREQ, tz="UTC")
    return pd.Series(range(n), index=idx, dtype=float, name="target_0")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task(tmp_path: Path) -> LazyTask:
    return _make_task(tmp_path)


@pytest.fixture()
def y_train() -> pd.Series:
    return _make_y_train()


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestCvTsReturnType:
    def test_returns_timeseries_fold(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert isinstance(cv, TimeSeriesFold)

    def test_steps_equals_predict_size(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.steps == task.config.predict_size

    def test_refit_is_false(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.refit is False

    def test_gap_is_zero(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.gap == 0

    def test_allow_incomplete_fold_is_true(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.allow_incomplete_fold is True


# ---------------------------------------------------------------------------
# initial_train_size
# ---------------------------------------------------------------------------


class TestCvTsInitialTrainSize:
    def test_initial_train_size_matches_slice(self, task, y_train):
        end_cv = task.config.end_train_ts - task.config.delta_val
        expected = len(y_train.loc[:end_cv])
        cv = task.cv_ts(y_train)
        assert cv.initial_train_size == expected

    def test_initial_train_size_positive(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.initial_train_size > 0

    def test_initial_train_size_less_than_total(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.initial_train_size < len(y_train)

    def test_initial_train_size_changes_with_different_series(self, task):
        """Longer series (both reaching before end_cv) → larger initial_train_size."""
        # default number_folds=10 → delta_val=70 days=1680 h.
        # Both series must start before end_cv so initial_train_size > 0.
        y_short = _make_y_train(n=2000)
        y_long = _make_y_train(n=4000)
        cv_short = task.cv_ts(y_short)
        cv_long = task.cv_ts(y_long)
        assert cv_long.initial_train_size > cv_short.initial_train_size


# ---------------------------------------------------------------------------
# fixed_train_size
# ---------------------------------------------------------------------------


class TestCvTsFixedTrainSize:
    def test_fixed_train_size_true_when_train_size_set(self, task, y_train):
        assert task.config.train_size is not None
        cv = task.cv_ts(y_train)
        assert cv.fixed_train_size is True

    def test_fixed_train_size_false_when_train_size_none(self, tmp_path, y_train):
        t = _make_task(tmp_path)
        t.config.train_size = None  # bypass ConfigMulti default
        cv = t.cv_ts(y_train)
        assert cv.fixed_train_size is False


# ---------------------------------------------------------------------------
# Method availability on all task classes
# ---------------------------------------------------------------------------


class TestCvTsAvailability:
    @pytest.mark.parametrize(
        "cls",
        [BaseTask, LazyTask, OptunaTask, SpotOptimTask],
        ids=["BaseTask", "LazyTask", "OptunaTask", "SpotOptimTask"],
    )
    def test_has_cv_ts(self, cls):
        assert hasattr(cls, "cv_ts")
        assert callable(getattr(cls, "cv_ts"))

    def test_cv_ts_signature_has_y_train(self):
        sig = inspect.signature(BaseTask.cv_ts)
        assert "y_train" in sig.parameters

    def test_cv_ts_return_annotation(self):
        sig = inspect.signature(BaseTask.cv_ts)
        assert sig.return_annotation is TimeSeriesFold


# ---------------------------------------------------------------------------
# Idempotency — calling twice yields identical configuration
# ---------------------------------------------------------------------------


class TestCvTsIdempotency:
    def test_two_calls_same_steps(self, task, y_train):
        assert task.cv_ts(y_train).steps == task.cv_ts(y_train).steps

    def test_two_calls_same_initial_train_size(self, task, y_train):
        assert (
            task.cv_ts(y_train).initial_train_size
            == task.cv_ts(y_train).initial_train_size
        )

    def test_two_calls_same_fixed_train_size(self, task, y_train):
        assert (
            task.cv_ts(y_train).fixed_train_size == task.cv_ts(y_train).fixed_train_size
        )
