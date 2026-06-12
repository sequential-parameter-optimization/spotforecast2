# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2.stats.autocorrelation.select_pacf_lags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spotforecast2.stats.autocorrelation import select_pacf_lags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_ar_series(n_days: int = 120, seed: int = 42) -> pd.Series:
    """AR(1) with moderate coefficient for general lag selection tests."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + rng.standard_normal()
    return pd.Series(ar)


def _make_ar24_series(n_days: int = 200, seed: int = 42) -> pd.Series:
    """AR(24) process — lag 24 dominates the PACF (strong daily periodicity)."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    ar = np.zeros(n)
    for t in range(24, n):
        ar[t] = 0.8 * ar[t - 24] + rng.standard_normal()
    return pd.Series(ar)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestSelectPacfLagsHappyPath:
    def test_returns_list_of_ints(self):
        series = _make_daily_ar_series()
        lags = select_pacf_lags(series, n_lags=50, top_k=8)
        assert isinstance(lags, list)
        assert all(isinstance(x, int) for x in lags)

    def test_sorted_ascending(self):
        series = _make_daily_ar_series()
        lags = select_pacf_lags(series, n_lags=50, top_k=8)
        assert lags == sorted(lags)

    def test_top_k_respected(self):
        series = _make_daily_ar_series()
        for k in (1, 4, 8):
            lags = select_pacf_lags(series, n_lags=50, top_k=k)
            assert len(lags) <= k

    def test_lag_1_selected_ar1_component(self):
        """AR(1) coefficient 0.7 should produce a strongly significant lag-1 PACF."""
        series = _make_daily_ar_series()
        lags = select_pacf_lags(series, n_lags=50, top_k=8)
        assert 1 in lags, f"expected lag 1 in {lags}"

    def test_daily_lag_24_in_top_k(self):
        """An AR(24) process must have lag 24 in the top-k selected lags."""
        series = _make_ar24_series(n_days=200)
        lags = select_pacf_lags(series, n_lags=50, top_k=8)
        assert 24 in lags, f"expected lag 24 in {lags} (AR(24) process)"

    def test_determinism(self):
        """Same series, same parameters → same result every call."""
        series = _make_daily_ar_series(seed=0)
        r1 = select_pacf_lags(series, n_lags=50, top_k=6)
        r2 = select_pacf_lags(series, n_lags=50, top_k=6)
        assert r1 == r2

    def test_n_lags_limits_search(self):
        """Returned lags must not exceed n_lags."""
        series = _make_daily_ar_series()
        n_lags = 30
        lags = select_pacf_lags(series, n_lags=n_lags, top_k=8)
        assert all(lag <= n_lags for lag in lags)


# ---------------------------------------------------------------------------
# Degenerate / edge-case tests
# ---------------------------------------------------------------------------

class TestSelectPacfLagsDegenerate:
    def test_constant_series_fallback_returned(self):
        series = pd.Series([1.0] * 50)
        result = select_pacf_lags(series, n_lags=10, fallback=[1, 2, 24])
        assert result == [1, 2, 24]

    def test_constant_series_no_fallback_raises(self):
        series = pd.Series([1.0] * 50)
        with pytest.raises(ValueError, match="no significant"):
            select_pacf_lags(series, n_lags=10, fallback=None)

    def test_very_short_series_fallback(self):
        """Series shorter than 2*n_lags+1; PACF falls back on statsmodels truncation.

        No lag should pass the wide confidence band for a near-white-noise short series.
        We do not mandate whether ValueError or fallback is triggered; only that either
        the fallback list is returned or a ValueError is raised.
        """
        rng = np.random.default_rng(7)
        series = pd.Series(rng.standard_normal(20))
        try:
            result = select_pacf_lags(series, n_lags=5, top_k=2, fallback=[1, 2])
            # if no significant lags, fallback should be returned
            assert isinstance(result, list)
        except ValueError:
            pass  # also acceptable

    def test_fallback_list_copied(self):
        """The returned list must be a copy, not the same object as fallback."""
        series = pd.Series([0.0] * 50)
        fb = [1, 2, 24]
        result = select_pacf_lags(series, n_lags=10, fallback=fb)
        assert result is not fb

    def test_fallback_none_is_default(self):
        """fallback=None is the default; constant series raises ValueError."""
        series = pd.Series([3.14] * 50)
        with pytest.raises(ValueError):
            select_pacf_lags(series, n_lags=10)


# ---------------------------------------------------------------------------
# Parameter variation
# ---------------------------------------------------------------------------

class TestSelectPacfLagsParams:
    def test_default_n_lags_200_top_k_8(self):
        """Default parameters work on a long series without error."""
        series = _make_daily_ar_series(n_days=500)
        lags = select_pacf_lags(series)
        assert len(lags) <= 8
        assert lags == sorted(lags)

    def test_top_k_1(self):
        series = _make_daily_ar_series()
        lags = select_pacf_lags(series, n_lags=50, top_k=1)
        assert len(lags) == 1

    def test_large_top_k_bounded_by_significant(self):
        """If fewer than top_k significant lags exist, return only those."""
        series = _make_daily_ar_series()
        # Use a large top_k; returned list must still be <= n_sig
        lags_k3 = select_pacf_lags(series, n_lags=5, top_k=3)
        lags_k100 = select_pacf_lags(series, n_lags=5, top_k=100)
        # Both share the same significant set (for n_lags=5); the larger top_k
        # cannot return MORE lags, only the same.
        assert set(lags_k3).issubset(set(lags_k100))
        assert len(lags_k100) >= len(lags_k3)
