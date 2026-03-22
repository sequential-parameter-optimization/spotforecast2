# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for custom_weights / get_missing_weights import consolidation and apply_imputation.

Validates that:
- custom_weights and get_missing_weights live exclusively in spotforecast2_safe
- spotforecast2.preprocessing re-exports them (no duplicate implementation)
- All public import paths for custom_weights resolve to the same function
- apply_imputation works correctly for both "weighted" and "linear" strategies
"""

import inspect
import logging
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

# ============================================================================
# Import paths under test
# ============================================================================

from spotforecast2_safe.preprocessing.imputation import custom_weights as cw_safe_direct
from spotforecast2_safe.preprocessing.imputation import (
    get_missing_weights as gmw_safe_direct,
)
from spotforecast2_safe.preprocessing import custom_weights as cw_safe_pkg
from spotforecast2_safe.preprocessing import get_missing_weights as gmw_safe_pkg
from spotforecast2.preprocessing import custom_weights as cw_sf2_pkg
from spotforecast2.preprocessing import get_missing_weights as gmw_sf2_pkg

from spotforecast2_safe.preprocessing.imputation import apply_imputation
from spotforecast2_safe.preprocessing import WeightFunction

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_index():
    return pd.date_range("2024-01-01", periods=10, freq="h")


@pytest.fixture
def all_one_weights(small_index):
    return pd.Series(1.0, index=small_index)


@pytest.fixture
def mixed_weights(small_index):
    values = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    return pd.Series(values, index=small_index)


@pytest.fixture
def no_gap_df():
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"A": rng.normal(10, 1, 200), "B": rng.normal(5, 1, 200)}, index=idx
    )


@pytest.fixture
def gap_df():
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"A": rng.normal(10, 1, 200), "B": rng.normal(5, 1, 200)}, index=idx
    )
    df.iloc[40:45, 0] = np.nan
    return df


@pytest.fixture
def logger():
    return logging.getLogger("test_imputation")


# ============================================================================
# Single source of truth: import path verification
# ============================================================================


class TestCustomWeightsSingleSource:
    """custom_weights lives in spotforecast2_safe; sf2.preprocessing re-exports it."""

    def test_sf2_pkg_is_same_object_as_safe_direct(self):
        assert cw_sf2_pkg is cw_safe_direct

    def test_sf2_pkg_is_same_object_as_safe_pkg(self):
        assert cw_sf2_pkg is cw_safe_pkg

    def test_source_file_is_in_safe_package(self):
        src = inspect.getfile(cw_safe_direct)
        assert "spotforecast2_safe" in src
        assert "spotforecast2/" not in src.replace("spotforecast2_safe", "")

    def test_sf2_preprocessing_has_no_own_imputation_module(self):
        """sf2's imputation.py has been deleted; no spotforecast2.preprocessing.imputation submodule."""

        # Trying to import the submodule should fail or resolve to sf2-safe
        try:
            import spotforecast2.preprocessing.imputation as mod

            # If it somehow imports, verify it comes from sf2-safe
            src = inspect.getfile(mod)
            assert "spotforecast2_safe" in src
        except ModuleNotFoundError:
            pass  # correct — the file no longer exists


class TestGetMissingWeightsSingleSource:
    """get_missing_weights lives in spotforecast2_safe; sf2.preprocessing re-exports it."""

    def test_sf2_pkg_is_same_object_as_safe_direct(self):
        assert gmw_sf2_pkg is gmw_safe_direct

    def test_sf2_pkg_is_same_object_as_safe_pkg(self):
        assert gmw_sf2_pkg is gmw_safe_pkg

    def test_source_file_is_in_safe_package(self):
        src = inspect.getfile(gmw_safe_direct)
        assert "spotforecast2_safe" in src
        assert "spotforecast2/" not in src.replace("spotforecast2_safe", "")

    def test_returns_numeric_series_not_boolean(self):
        """Returns 0.0/1.0, not boolean — previous sf2 bug fixed."""
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"A": rng.normal(10, 1, 100)}, index=idx)
        df.iloc[20:25, 0] = np.nan
        _, weights = gmw_sf2_pkg(df, window_size=5)
        assert weights.dtype != bool
        assert set(weights.unique()).issubset({0.0, 1.0})


# ============================================================================
# custom_weights behaviour (via sf2 import path)
# ============================================================================


class TestCustomWeightsBehaviour:
    """Functional tests using the sf2 import path."""

    def test_pd_index_returns_ndarray(self, all_one_weights, small_index):
        result = cw_sf2_pkg(small_index, all_one_weights)
        assert isinstance(result, np.ndarray)

    def test_pd_index_values_match(self, mixed_weights, small_index):
        result = cw_sf2_pkg(small_index, mixed_weights)
        np.testing.assert_array_equal(result, mixed_weights.values)

    def test_scalar_index_returns_scalar(self, all_one_weights, small_index):
        result = cw_sf2_pkg(small_index[0], all_one_weights)
        assert result == 1.0

    def test_raises_on_unknown_pd_index(self, all_one_weights):
        bad_idx = pd.date_range("2030-01-01", periods=3, freq="h")
        with pytest.raises(ValueError, match="Index not found"):
            cw_sf2_pkg(bad_idx, all_one_weights)

    def test_raises_on_unknown_scalar(self, all_one_weights):
        with pytest.raises(ValueError, match="Index not found"):
            cw_sf2_pkg(pd.Timestamp("2030-01-01"), all_one_weights)


# ============================================================================
# apply_imputation — "linear" strategy
# ============================================================================


class TestApplyImputationLinear:
    def _config(self, targets):
        return SimpleNamespace(
            imputation_method="linear", targets=targets, window_size=10
        )

    def test_fills_nans_in_target_columns(self, gap_df, logger):
        config = self._config(["A"])
        imputed, weight_func = apply_imputation(gap_df.copy(), config, logger)
        assert imputed["A"].isnull().sum() == 0

    def test_returns_none_weight_func(self, gap_df, logger):
        config = self._config(["A"])
        _, weight_func = apply_imputation(gap_df.copy(), config, logger)
        assert weight_func is None

    def test_no_gap_data_unchanged_values(self, no_gap_df, logger):
        config = self._config(["A", "B"])
        original_A = no_gap_df["A"].copy()
        imputed, _ = apply_imputation(no_gap_df.copy(), config, logger)
        pd.testing.assert_series_equal(imputed["A"], original_A)

    def test_returns_dataframe(self, no_gap_df, logger):
        config = self._config(["A", "B"])
        result, _ = apply_imputation(no_gap_df.copy(), config, logger)
        assert isinstance(result, pd.DataFrame)

    def test_shape_preserved(self, gap_df, logger):
        config = self._config(["A"])
        imputed, _ = apply_imputation(gap_df.copy(), config, logger)
        assert imputed.shape == gap_df.shape


# ============================================================================
# apply_imputation — "weighted" strategy
# ============================================================================


class TestApplyImputationWeighted:
    def _config(self):
        return SimpleNamespace(
            imputation_method="weighted", targets=["A", "B"], window_size=10
        )

    def test_returns_weight_function_instance(self, gap_df, logger):
        config = self._config()
        _, weight_func = apply_imputation(gap_df.copy(), config, logger)
        assert isinstance(weight_func, WeightFunction)

    def test_fills_nans_in_all_columns(self, gap_df, logger):
        config = self._config()
        imputed, _ = apply_imputation(gap_df.copy(), config, logger)
        assert imputed.isnull().sum().sum() == 0

    def test_no_gap_weight_func_returns_ones(self, no_gap_df, logger):
        """With sf2-safe's get_missing_weights, gap-free data produces all-1.0 weights."""
        config = self._config()
        _, weight_func = apply_imputation(no_gap_df.copy(), config, logger)
        result = weight_func(no_gap_df.index)
        assert result is not None
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.ones(len(no_gap_df)))

    def test_gap_weight_func_has_zero_zones(self, gap_df, logger):
        config = self._config()
        _, weight_func = apply_imputation(gap_df.copy(), config, logger)
        # The WeightFunction wraps a weights_series; check it has some non-one weights
        assert (weight_func.weights_series != 1.0).any()

    def test_shape_preserved(self, gap_df, logger):
        config = self._config()
        imputed, _ = apply_imputation(gap_df.copy(), config, logger)
        assert imputed.shape == gap_df.shape


# ============================================================================
# apply_imputation — unknown method raises
# ============================================================================


class TestApplyImputationUnknownMethod:
    def test_raises_on_unknown_method(self, no_gap_df, logger):
        config = SimpleNamespace(
            imputation_method="magic", targets=["A"], window_size=5
        )
        with pytest.raises(ValueError, match="Unknown imputation_method"):
            apply_imputation(no_gap_df.copy(), config, logger)
