# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for the sklearn.model_selection.TimeSeriesSplit integration
in BaseTask.cv_ts().

Verifies that:
- The underlying sklearn TimeSeriesSplit respects temporal ordering (no leakage).
- The number of folds matches ``config.number_folds``.
- Each test fold has exactly ``config.predict_size`` observations.
- ``initial_train_size`` in the returned TimeSeriesFold matches the first
  sklearn training-fold length.
- Fixed (sliding) vs expanding window behaves correctly with respect to
  ``config.train_size``.
- Train and test index sets are disjoint within every fold.
- Changing ``number_folds`` or ``predict_size`` propagates correctly.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

from spotforecast2.manager.multitask import LazyTask
from spotforecast2.model_selection.split_ts_cv import TimeSeriesFold

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_END_TRAIN = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
_FREQ = "h"
_N = 3000  # three-thousand hourly observations — comfortably more than any fold


def _make_task(tmp_path, **kwargs) -> LazyTask:
    """Return a LazyTask with ``end_train_ts`` set to ``_END_TRAIN``.

    val_days is pinned to 7 so that DELTA_VAL = 7 * number_folds days, keeping
    the test series length requirements manageable.  The val_days parameter
    itself is exercised in test_train_val_days.py.
    """
    kwargs.setdefault("val_days", 7)
    t = LazyTask(data_frame_name="test_data", cache_home=tmp_path, **kwargs)
    t.config.end_train_ts = _END_TRAIN
    return t


def _make_y_train(end: pd.Timestamp = _END_TRAIN, n: int = _N) -> pd.Series:
    """Synthetic hourly Series of length *n* ending at *end*."""
    idx = pd.date_range(end=end, periods=n, freq=_FREQ, tz="UTC")
    return pd.Series(range(n), index=idx, dtype=float, name="target_0")


def _skl_cv(task: LazyTask, y_train: pd.Series) -> SklearnTimeSeriesSplit:
    """Build the equivalent sklearn TimeSeriesSplit for *task*."""
    end_cv = task.config.end_train_ts - task.config.delta_val
    n_train_cv = len(y_train.loc[:end_cv])
    max_train_size = n_train_cv if task.config.train_size is not None else None
    return SklearnTimeSeriesSplit(
        n_splits=task.number_folds,
        max_train_size=max_train_size,
        test_size=task.config.predict_size,
        gap=0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task(tmp_path) -> LazyTask:
    return _make_task(tmp_path)


@pytest.fixture()
def y_train() -> pd.Series:
    return _make_y_train()


# ---------------------------------------------------------------------------
# Temporal ordering — no data leakage
# ---------------------------------------------------------------------------


class TestTemporalOrdering:
    """All training indices must come strictly before their test indices."""

    def test_all_train_indices_precede_test_indices(self, task, y_train):
        for train_idx, test_idx in _skl_cv(task, y_train).split(y_train):
            assert (
                train_idx.max() < test_idx.min()
            ), "Data leakage: a training index is not strictly before its test index"

    def test_test_folds_are_ordered_over_time(self, task, y_train):
        """Consecutive test folds must cover later time periods."""
        splits = list(_skl_cv(task, y_train).split(y_train))
        for i in range(len(splits) - 1):
            _, test_current = splits[i]
            _, test_next = splits[i + 1]
            assert (
                test_current.min() < test_next.min()
            ), f"Fold {i} test starts after fold {i+1} — ordering violated"

    def test_no_overlap_between_train_and_test(self, task, y_train):
        """Train and test index sets must be fully disjoint in every fold."""
        for train_idx, test_idx in _skl_cv(task, y_train).split(y_train):
            assert (
                len(np.intersect1d(train_idx, test_idx)) == 0
            ), "Train and test index sets overlap — data leakage detected"


# ---------------------------------------------------------------------------
# Number of splits
# ---------------------------------------------------------------------------


class TestNumberOfSplits:
    def test_split_count_equals_number_folds_default(self, task, y_train):
        splits = list(_skl_cv(task, y_train).split(y_train))
        assert len(splits) == task.number_folds

    @pytest.mark.parametrize("n_folds", [3, 5, 7, 10, 15])
    def test_split_count_with_various_number_folds(self, tmp_path, n_folds):
        task = _make_task(tmp_path, number_folds=n_folds)
        y = _make_y_train()
        splits = list(_skl_cv(task, y).split(y))
        assert len(splits) == n_folds

    def test_cv_ts_steps_equal_predict_size_for_different_folds(self, tmp_path):
        for n_folds in (5, 10):
            task = _make_task(tmp_path, number_folds=n_folds)
            y = _make_y_train()
            cv = task.cv_ts(y)
            assert cv.steps == task.config.predict_size


# ---------------------------------------------------------------------------
# Test-fold size
# ---------------------------------------------------------------------------


class TestFoldTestSize:
    def test_each_test_fold_has_predict_size_observations(self, task, y_train):
        for _, test_idx in _skl_cv(task, y_train).split(y_train):
            assert len(test_idx) == task.config.predict_size

    @pytest.mark.parametrize("predict_size", [12, 24, 48])
    def test_test_fold_size_with_various_predict_sizes(self, tmp_path, predict_size):
        task = _make_task(tmp_path, predict_size=predict_size)
        y = _make_y_train()
        for _, test_idx in _skl_cv(task, y).split(y):
            assert len(test_idx) == predict_size


# ---------------------------------------------------------------------------
# initial_train_size derived from sklearn first fold
# ---------------------------------------------------------------------------


class TestInitialTrainSize:
    def test_initial_train_size_matches_sklearn_first_fold(self, task, y_train):
        splits = list(_skl_cv(task, y_train).split(y_train))
        expected = len(splits[0][0])
        cv = task.cv_ts(y_train)
        assert cv.initial_train_size == expected

    def test_initial_train_size_is_positive(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.initial_train_size > 0

    def test_initial_train_size_less_than_n_samples(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.initial_train_size < len(y_train)

    def test_initial_train_size_stable_across_two_calls(self, task, y_train):
        assert (
            task.cv_ts(y_train).initial_train_size
            == task.cv_ts(y_train).initial_train_size
        )


# ---------------------------------------------------------------------------
# Fixed sliding window (max_train_size) vs expanding window
# ---------------------------------------------------------------------------


class TestSlidingVsExpandingWindow:
    def test_fixed_window_each_fold_same_train_size(self, task, y_train):
        """With ``max_train_size`` set every fold has the same training size."""
        assert task.config.train_size is not None  # default: 365 days
        splits = list(_skl_cv(task, y_train).split(y_train))
        train_sizes = [len(t) for t, _ in splits]
        assert all(
            s == train_sizes[0] for s in train_sizes
        ), f"Expected uniform fold sizes but got {train_sizes}"

    def test_expanding_window_train_size_grows(self, tmp_path, y_train):
        """Without ``max_train_size`` each fold sees more history than the last."""
        task = _make_task(tmp_path)
        task.config.train_size = None  # force expanding window

        skl = SklearnTimeSeriesSplit(
            n_splits=task.number_folds,
            max_train_size=None,
            test_size=task.config.predict_size,
            gap=0,
        )
        splits = list(skl.split(y_train))
        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Fold {i} train size {train_sizes[i]} not larger than "
                f"fold {i-1} train size {train_sizes[i-1]}"
            )

    def test_fixed_train_size_flag_true_when_train_size_set(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert cv.fixed_train_size is True

    def test_fixed_train_size_flag_false_when_train_size_none(self, tmp_path, y_train):
        task = _make_task(tmp_path)
        task.config.train_size = None
        cv = task.cv_ts(y_train)
        assert cv.fixed_train_size is False

    def test_max_train_size_equals_n_train_cv(self, task, y_train):
        """When ``train_size`` is set, ``max_train_size`` equals ``n_train_cv``."""
        end_cv = task.config.end_train_ts - task.config.delta_val
        n_train_cv = len(y_train.loc[:end_cv])
        # Each fold's training set should be at most n_train_cv observations.
        for train_idx, _ in _skl_cv(task, y_train).split(y_train):
            assert len(train_idx) <= n_train_cv


# ---------------------------------------------------------------------------
# Varying number_folds affects cv_ts output
# ---------------------------------------------------------------------------


class TestNumberFoldsEffect:
    def test_more_folds_same_initial_train_size_fixed_window(self, tmp_path, y_train):
        """With a fixed sliding window, initial_train_size is independent of fold count."""
        task_5 = _make_task(tmp_path, number_folds=5)
        task_10 = _make_task(tmp_path, number_folds=10)
        # Both use the same n_train_cv derived from end_cv
        end_cv_5 = task_5.config.end_train_ts - task_5.config.delta_val
        end_cv_10 = task_10.config.end_train_ts - task_10.config.delta_val
        n5 = len(y_train.loc[:end_cv_5])
        n10 = len(y_train.loc[:end_cv_10])
        # With fixed window: initial_train_size == max_train_size == n_train_cv
        assert task_5.cv_ts(y_train).initial_train_size == n5
        assert task_10.cv_ts(y_train).initial_train_size == n10

    def test_more_folds_consumes_more_validation_data(self, tmp_path, y_train):
        """Larger number_folds enlarges delta_val, shrinking n_train_cv."""
        task_5 = _make_task(tmp_path, number_folds=5)
        task_15 = _make_task(tmp_path, number_folds=15)
        end_cv_5 = task_5.config.end_train_ts - task_5.config.delta_val
        end_cv_15 = task_15.config.end_train_ts - task_15.config.delta_val
        n5 = len(y_train.loc[:end_cv_5])
        n15 = len(y_train.loc[:end_cv_15])
        assert (
            n15 < n5
        ), "Larger number_folds should push end_cv earlier, reducing n_train_cv"


# ---------------------------------------------------------------------------
# Return type and TimeSeriesFold attributes
# ---------------------------------------------------------------------------


class TestReturnTypeAndAttributes:
    def test_cv_ts_returns_timeseries_fold(self, task, y_train):
        cv = task.cv_ts(y_train)
        assert isinstance(cv, TimeSeriesFold)

    def test_cv_ts_gap_is_zero(self, task, y_train):
        assert task.cv_ts(y_train).gap == 0

    def test_cv_ts_refit_is_false(self, task, y_train):
        assert task.cv_ts(y_train).refit is False

    def test_cv_ts_allow_incomplete_fold_true(self, task, y_train):
        assert task.cv_ts(y_train).allow_incomplete_fold is True
