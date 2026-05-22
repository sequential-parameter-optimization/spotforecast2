# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``LAGS_CONSIDER`` and ``WINDOW_FEATURES`` in
``multitask/search_spaces.py``.

Single canonical home (ADR-002 Step 5 deleted the legacy
``spotforecast2.trainer.trainer_full`` module).
"""

from spotforecast2_safe.preprocessing import RollingFeatures

from spotforecast2.multitask.search_spaces import LAGS_CONSIDER, WINDOW_FEATURES


def test_lags_consider_value():
    assert LAGS_CONSIDER == list(range(1, 24))


def test_window_features_is_five_rolling_features():
    """The classic chag25a five-window configuration."""
    assert len(WINDOW_FEATURES) == 5
    for entry in WINDOW_FEATURES:
        assert isinstance(entry, RollingFeatures)


def test_window_features_window_sizes():
    """Each ``RollingFeatures`` was constructed with the documented window."""

    def _scalar(ws):
        # ``RollingFeatures`` normalises a single int to a 1-tuple internally.
        if hasattr(ws, "__len__") and len(ws) == 1:
            return ws[0]
        return ws

    sizes = [_scalar(rf.window_sizes) for rf in WINDOW_FEATURES]
    assert sizes == [24, 24 * 7, 24 * 30, 24, 24]
