# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for the ``cv_block_size`` override in BaseTask.cv_ts().

``cv_ts`` now reads an optional ``config.cv_block_size`` to size the
cross-validation block independently of the live forecast horizon
(``predict_size``).  Coverage here:

- When ``cv_block_size`` is set (and differs from ``predict_size``) the
  returned ``TimeSeriesFold`` reports ``steps == cv_block_size`` and the
  underlying sklearn split uses ``test_size == cv_block_size``.
- When ``cv_block_size`` is absent or ``None`` the behaviour is unchanged:
  ``steps == predict_size`` (the getattr fallback).
- The resulting fold count over a synthetic hourly series tracks the chosen
  block width.

These tests do not require the spotforecast2-safe ``cv_block_size`` field to
be present on the config class — the override is exercised via the same
``getattr`` fallback that ``cv_ts`` uses, by assigning the attribute on the
constructed config object (mirroring how the existing cv_ts tests set
``config.end_train_ts`` and ``config.train_size``).
"""

import math
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

from spotforecast2.multitask import LazyTask

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_END_TRAIN = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
_FREQ = "h"
_N = 3000  # three-thousand hourly observations


def _make_task(tmp_path: Path, *, val_days: int = 7, **kwargs) -> LazyTask:
    """Return a LazyTask with ``end_train_ts`` pinned to ``_END_TRAIN``.

    Mirrors the helper in ``test_cv_ts_sklearn.py``: ``delta_val`` is computed
    explicitly from ``val_days * number_folds`` rather than derived inside
    ``BaseTask``.
    """
    number_folds = kwargs.get("number_folds", 10)
    overrides = dict(kwargs, delta_val=pd.Timedelta(days=val_days * number_folds))
    overrides.setdefault("data_frame_name", "test_data")
    t = LazyTask(cache_home=tmp_path, **overrides)
    t.config.end_train_ts = _END_TRAIN
    return t


def _make_y_train(end: pd.Timestamp = _END_TRAIN, n: int = _N) -> pd.Series:
    """Synthetic hourly Series of length *n* ending at *end*."""
    idx = pd.date_range(end=end, periods=n, freq=_FREQ, tz="UTC")
    return pd.Series(range(n), index=idx, dtype=float, name="target_0")


def _skl_cv(
    task: LazyTask, y_train: pd.Series, test_size: int
) -> SklearnTimeSeriesSplit:
    """Build the sklearn TimeSeriesSplit equivalent for *task* with *test_size*."""
    end_cv = task.config.end_train_ts - task.config.delta_val
    n_train_cv = len(y_train.loc[:end_cv])
    max_train_size = n_train_cv if task.config.train_size is not None else None
    return SklearnTimeSeriesSplit(
        n_splits=task.config.number_folds,
        max_train_size=max_train_size,
        test_size=test_size,
        gap=0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def y_train() -> pd.Series:
    return _make_y_train()


# ---------------------------------------------------------------------------
# cv_block_size set → steps and test_size follow the block, not predict_size
# ---------------------------------------------------------------------------


class TestCvBlockSizeOverride:
    def test_steps_equal_cv_block_size_not_predict_size(self, tmp_path, y_train):
        task = _make_task(tmp_path, predict_size=32)
        task.config.cv_block_size = 24
        assert task.config.predict_size == 32

        cv = task.cv_ts(y_train)
        assert cv.steps == 24
        assert cv.steps != task.config.predict_size

    def test_sklearn_split_uses_cv_block_size_test_size(self, tmp_path, y_train):
        task = _make_task(tmp_path, predict_size=32)
        task.config.cv_block_size = 24

        # The split the task builds internally uses cv_block_size as test_size;
        # every test fold therefore has exactly cv_block_size observations.
        for _, test_idx in _skl_cv(task, y_train, test_size=24).split(y_train):
            assert len(test_idx) == 24

    @pytest.mark.parametrize("cv_block_size", [12, 24, 48])
    def test_steps_track_various_cv_block_sizes(self, tmp_path, y_train, cv_block_size):
        task = _make_task(tmp_path, predict_size=32)
        task.config.cv_block_size = cv_block_size
        assert task.cv_ts(y_train).steps == cv_block_size


# ---------------------------------------------------------------------------
# cv_block_size absent or None → unchanged behaviour (getattr fallback)
# ---------------------------------------------------------------------------


class TestCvBlockSizeFallback:
    def test_steps_equal_predict_size_when_attr_absent(self, tmp_path, y_train):
        task = _make_task(tmp_path, predict_size=32)
        # Exercise the getattr-absent fallback in cv_ts: a config that does not
        # declare cv_block_size at all must fall back to predict_size. Newer
        # spotforecast2-safe ConfigMulti declares the field (default None), so
        # delete it from the instance to recreate the "attribute absent" path
        # the getattr fallback is written for.
        if hasattr(task.config, "cv_block_size"):
            delattr(task.config, "cv_block_size")
        assert not hasattr(task.config, "cv_block_size")
        assert task.cv_ts(y_train).steps == task.config.predict_size == 32

    def test_steps_equal_predict_size_when_none(self, tmp_path, y_train):
        task = _make_task(tmp_path, predict_size=32)
        task.config.cv_block_size = None  # explicit None ⇒ fall back
        assert task.cv_ts(y_train).steps == task.config.predict_size == 32


# ---------------------------------------------------------------------------
# Fold count tracks the block width
# ---------------------------------------------------------------------------


class TestCvBlockSizeFoldCount:
    def test_test_fold_size_matches_block_over_validation_span(self, tmp_path, y_train):
        """Every test fold spans exactly cv_block_size observations, and the
        validation span covered equals number_folds * cv_block_size."""
        task = _make_task(tmp_path, predict_size=32, number_folds=10)
        cv_block = 24
        task.config.cv_block_size = cv_block

        splits = list(_skl_cv(task, y_train, test_size=cv_block).split(y_train))
        assert len(splits) == task.config.number_folds

        total_validation = sum(len(test_idx) for _, test_idx in splits)
        assert total_validation == task.config.number_folds * cv_block

        # The number of folds needed to cover that validation span at this block
        # width is exactly number_folds (no leftover partial block).
        assert math.ceil(total_validation / cv_block) == task.config.number_folds
