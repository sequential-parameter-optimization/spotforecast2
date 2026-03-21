# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test suite for WeightFunction and get_missing_weights in preprocessing.imputation.

Safety-critical validation scope:
- WeightFunction returns correct weights for valid windows
- WeightFunction returns None (not an all-zero array) when the entire
  requested window has zero weights — preventing ForecasterRecursive from
  raising "sample_weight cannot be normalized because the sum is zero"
- get_missing_weights produces all-ones series when data has no missing values
- Pickle round-trip preserves behaviour
- custom_weights raises ValueError for unknown indices
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.imputation import (
    WeightFunction,
    custom_weights,
    get_missing_weights,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mixed_weights():
    """Series with a mix of 0.0 and 1.0 weights."""
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    values = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    return pd.Series(values, index=idx)


@pytest.fixture
def all_one_weights():
    """Series where every weight is 1.0 (no gaps)."""
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    return pd.Series(1.0, index=idx)


@pytest.fixture
def all_zero_weights():
    """Series where every weight is 0.0 (entire window penalised)."""
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    return pd.Series(0.0, index=idx)


@pytest.fixture
def no_gap_dataframe():
    """Small DataFrame with no missing values — simulates demo02.csv."""
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"A": rng.normal(10, 2, 200), "B": rng.normal(5, 1, 200)}, index=idx
    )


@pytest.fixture
def gap_dataframe():
    """DataFrame with deliberate NaN gaps."""
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {"A": rng.normal(10, 2, 200), "B": rng.normal(5, 1, 200)}, index=idx
    )
    data.iloc[50:55, 0] = np.nan
    data.iloc[120:123, 1] = np.nan
    return data


# ============================================================================
# WeightFunction — normal (positive-sum) windows
# ============================================================================


class TestWeightFunctionNormalWindow:
    """WeightFunction behaviour when weights sum to a positive value."""

    def test_returns_ndarray_for_positive_sum(self, mixed_weights):
        wf = WeightFunction(mixed_weights)
        idx = mixed_weights.index[3:8]  # all-1.0 slice
        result = wf(idx)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.ones(5))

    def test_values_match_weights_series(self, mixed_weights):
        wf = WeightFunction(mixed_weights)
        idx = mixed_weights.index[:5]
        result = wf(idx)
        expected = mixed_weights.iloc[:5].values
        np.testing.assert_array_equal(result, expected)

    def test_all_ones_series_returns_ones(self, all_one_weights):
        wf = WeightFunction(all_one_weights)
        result = wf(all_one_weights.index)
        assert result is not None
        np.testing.assert_array_equal(result, np.ones(10))

    def test_mixed_window_partial_zeros_still_returns_array(self, mixed_weights):
        """A window that contains *some* zeros but sums > 0 returns an array."""
        wf = WeightFunction(mixed_weights)
        idx = mixed_weights.index  # includes zeros at ends, sum > 0
        result = wf(idx)
        assert result is not None
        assert isinstance(result, np.ndarray)


# ============================================================================
# WeightFunction — degenerate (zero-sum) windows
# ============================================================================


class TestWeightFunctionZeroSumWindow:
    """WeightFunction must return None when all weights in the window are zero.

    This prevents ForecasterRecursive.create_sample_weights from raising
    ``ValueError: sample_weight cannot be normalized because sum is zero``.
    """

    def test_returns_none_for_all_zero_weights(self, all_zero_weights):
        wf = WeightFunction(all_zero_weights)
        result = wf(all_zero_weights.index)
        assert result is None

    def test_returns_none_for_zero_subwindow(self, mixed_weights):
        """Selecting only the zero-weight prefix must return None."""
        wf = WeightFunction(mixed_weights)
        zero_idx = mixed_weights.index[:3]  # all 0.0
        result = wf(zero_idx)
        assert result is None

    def test_does_not_raise_on_zero_sum(self, all_zero_weights):
        """Calling WeightFunction on a zero-sum window must not raise."""
        wf = WeightFunction(all_zero_weights)
        try:
            result = wf(all_zero_weights.index)
            assert result is None
        except Exception as exc:
            pytest.fail(f"WeightFunction raised unexpectedly: {exc}")

    def test_none_allows_forecaster_to_skip_weighting(self, all_zero_weights):
        """Simulate what ForecasterRecursive.create_sample_weights does."""
        wf = WeightFunction(all_zero_weights)
        sample_weight = wf(all_zero_weights.index)
        # ForecasterRecursive only validates when sample_weight is not None
        assert sample_weight is None  # → no validation, no crash


# ============================================================================
# WeightFunction — pickle round-trip
# ============================================================================


class TestWeightFunctionPickle:
    """WeightFunction must survive pickle/unpickle with the same behaviour."""

    def test_pickle_roundtrip_positive_sum(self, all_one_weights):
        wf = WeightFunction(all_one_weights)
        wf2 = pickle.loads(pickle.dumps(wf))
        result = wf2(all_one_weights.index)
        assert result is not None
        np.testing.assert_array_equal(result, np.ones(10))

    def test_pickle_roundtrip_zero_sum(self, all_zero_weights):
        wf = WeightFunction(all_zero_weights)
        wf2 = pickle.loads(pickle.dumps(wf))
        result = wf2(all_zero_weights.index)
        assert result is None

    def test_repr_is_informative(self, all_one_weights):
        wf = WeightFunction(all_one_weights)
        r = repr(wf)
        assert "WeightFunction" in r
        assert "10" in r


# ============================================================================
# get_missing_weights — no-gap data (demo02.csv scenario)
# ============================================================================


class TestGetMissingWeightsNoGap:
    """When the input has no NaN values, weights_series must be all 1.0.

    Regression test for the demo02.csv scenario: IsolationForest marks some
    rows as outliers (NaN), but if the data truly has no gaps before that
    step, get_missing_weights must produce uniform 1.0 weights so that
    WeightFunction correctly returns an array (not None) for the full window.
    """

    def test_no_gap_weights_are_all_ones(self, no_gap_dataframe):
        _, weights = get_missing_weights(no_gap_dataframe, window_size=10)
        assert (weights == 1.0).all(), "Expected all-ones weights for gap-free data"

    def test_no_gap_weight_sum_positive(self, no_gap_dataframe):
        _, weights = get_missing_weights(no_gap_dataframe, window_size=10)
        assert weights.sum() > 0

    def test_no_gap_weight_function_returns_array(self, no_gap_dataframe):
        _, weights = get_missing_weights(no_gap_dataframe, window_size=10)
        wf = WeightFunction(weights)
        result = wf(weights.index)
        assert result is not None
        assert isinstance(result, np.ndarray)


# ============================================================================
# get_missing_weights — data with gaps
# ============================================================================


class TestGetMissingWeightsWithGaps:
    """Verify gap-penalty zones are created correctly."""

    def test_gap_rows_have_zero_weight(self, gap_dataframe):
        _, weights = get_missing_weights(gap_dataframe, window_size=5)
        # The gap rows themselves (before ffill) should be in zero-weight zone
        assert (weights == 0.0).any()

    def test_non_gap_rows_have_one_weight(self, gap_dataframe):
        _, weights = get_missing_weights(gap_dataframe, window_size=5)
        assert (weights == 1.0).any()

    def test_filled_data_has_no_nans(self, gap_dataframe):
        filled, _ = get_missing_weights(gap_dataframe, window_size=5)
        assert filled.isnull().sum().sum() == 0

    def test_weights_series_length_matches_data(self, gap_dataframe):
        filled, weights = get_missing_weights(gap_dataframe, window_size=5)
        assert len(weights) == len(filled)

    def test_weight_func_returns_none_for_all_zero_subwindow(self, gap_dataframe):
        """A training window that sits entirely inside a gap-penalty zone → None."""
        filled, weights = get_missing_weights(gap_dataframe, window_size=72)
        wf = WeightFunction(weights)
        # Rows 50–54 were set to NaN; with window_size=72 the zone extends far
        zero_idx = weights.index[weights == 0.0]
        if len(zero_idx) > 0:
            result = wf(zero_idx)
            assert result is None

    def test_weight_func_returns_array_for_positive_subwindow(self, gap_dataframe):
        _, weights = get_missing_weights(gap_dataframe, window_size=5)
        wf = WeightFunction(weights)
        positive_idx = weights.index[weights == 1.0]
        if len(positive_idx) > 0:
            result = wf(positive_idx)
            assert result is not None
            assert isinstance(result, np.ndarray)


# ============================================================================
# get_missing_weights — input validation
# ============================================================================


class TestGetMissingWeightsValidation:
    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            get_missing_weights(pd.DataFrame())

    def test_raises_on_nonpositive_window_size(self, no_gap_dataframe):
        with pytest.raises(ValueError, match="positive"):
            get_missing_weights(no_gap_dataframe, window_size=0)

    def test_raises_when_window_ge_nrows(self, no_gap_dataframe):
        with pytest.raises(ValueError, match="smaller"):
            get_missing_weights(no_gap_dataframe, window_size=len(no_gap_dataframe))


# ============================================================================
# custom_weights — index validation
# ============================================================================


class TestCustomWeights:
    def test_raises_for_unknown_index(self, all_one_weights):
        bad_idx = pd.date_range("2025-01-01", periods=3, freq="h")
        with pytest.raises(ValueError, match="Index not found"):
            custom_weights(bad_idx, all_one_weights)

    def test_raises_for_unknown_scalar(self, all_one_weights):
        with pytest.raises(ValueError, match="Index not found"):
            custom_weights(pd.Timestamp("2025-01-01"), all_one_weights)

    def test_known_index_returns_correct_values(self, mixed_weights):
        idx = mixed_weights.index[3:6]
        result = custom_weights(idx, mixed_weights)
        np.testing.assert_array_equal(result, np.ones(3))
