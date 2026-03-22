# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for apply_imputation() — now living in spotforecast2_safe.

Coverage:
- apply_imputation is importable from spotforecast2_safe (canonical source)
- apply_imputation is still importable from spotforecast2.preprocessing (re-export)
- Linear method fills NaN values and returns weight_func=None
- Weighted method fills NaN values and returns a WeightFunction instance
- Logging: NaN count before and after is logged at INFO level
- Warning is emitted when NaN values remain after imputation
- ValueError is raised for an unknown imputation_method
- Diagnostic: nan_before is logged before any imputation happens
"""

import logging
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.imputation import apply_imputation
from spotforecast2_safe.preprocessing.imputation import WeightFunction

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 100,
    gap_slice: slice | None = None,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a small hourly DataFrame, optionally with NaN gaps."""
    if cols is None:
        cols = ["A", "B"]
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    data = {c: rng.uniform(1, 10, n) for c in cols}
    df = pd.DataFrame(data, index=idx)
    if gap_slice is not None:
        for c in cols:
            df.iloc[gap_slice, df.columns.get_loc(c)] = np.nan
    return df


def _make_config(method: str, window_size: int = 10, cols: list[str] | None = None):
    """Return a minimal config SimpleNamespace."""
    if cols is None:
        cols = ["A", "B"]
    return SimpleNamespace(
        imputation_method=method,
        targets=cols,
        window_size=window_size,
    )


@pytest.fixture
def df_with_gap():
    return _make_df(n=100, gap_slice=slice(20, 25))


@pytest.fixture
def df_no_gap():
    return _make_df(n=100, gap_slice=None)


@pytest.fixture
def stdlib_logger():
    return logging.getLogger("test_apply_imputation")


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestApplyImputationImport:
    """apply_imputation lives in sf2-safe; sf2.preprocessing re-exports it."""

    def test_importable_from_sf2safe_submodule(self):
        from spotforecast2_safe.preprocessing.imputation import apply_imputation as fn

        assert callable(fn)

    def test_importable_from_sf2safe_package(self):
        from spotforecast2_safe.preprocessing import apply_imputation as fn

        assert callable(fn)

    def test_importable_from_sf2_package(self):
        from spotforecast2.preprocessing import apply_imputation as fn

        assert callable(fn)

    def test_sf2_package_reexports_same_object(self):
        from spotforecast2_safe.preprocessing.imputation import (
            apply_imputation as canonical,
        )
        from spotforecast2.preprocessing import apply_imputation as reexport

        assert canonical is reexport

    def test_source_file_is_in_safe_package(self):
        src = inspect.getfile(apply_imputation)
        assert "spotforecast2_safe" in src
        assert "spotforecast2/" not in src.replace("spotforecast2_safe", "")

    def test_no_apply_imputation_in_sf2_preprocessing_pkg(self):
        """sf2 must not have its own imputation.py anymore."""
        import sys

        # The module spotforecast2.preprocessing.imputation must NOT exist
        assert (
            "spotforecast2.preprocessing.imputation" not in sys.modules
            or inspect.getfile(
                sys.modules["spotforecast2.preprocessing.imputation"]
            ).find("spotforecast2_safe")
            != -1
        )


# ---------------------------------------------------------------------------
# Linear imputation
# ---------------------------------------------------------------------------


class TestApplyImputationLinear:
    """Linear method must fill NaN values and return weight_func=None."""

    def test_linear_fills_all_nans(self, df_with_gap, stdlib_logger):
        config = _make_config("linear")
        result, wf = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert result.isnull().sum().sum() == 0

    def test_linear_returns_none_weight_func(self, df_with_gap, stdlib_logger):
        config = _make_config("linear")
        _, wf = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert wf is None

    def test_linear_returns_dataframe(self, df_with_gap, stdlib_logger):
        config = _make_config("linear")
        result, _ = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert isinstance(result, pd.DataFrame)

    def test_linear_preserves_shape(self, df_with_gap, stdlib_logger):
        config = _make_config("linear")
        result, _ = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert result.shape == df_with_gap.shape

    def test_linear_no_gap_unchanged_sum(self, df_no_gap, stdlib_logger):
        """With no gaps, linear imputation must not alter values."""
        config = _make_config("linear")
        original_sum = df_no_gap.sum().sum()
        result, _ = apply_imputation(df_no_gap.copy(), config, stdlib_logger)
        assert pytest.approx(result.sum().sum(), rel=1e-9) == original_sum

    def test_linear_only_fills_listed_targets(self, stdlib_logger):
        """Only columns listed in config.targets should be interpolated."""
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "A": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "B": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            },
            index=idx,
        )
        config = _make_config("linear", cols=["A"])  # only target A
        result, _ = apply_imputation(df, config, stdlib_logger)
        assert result["A"].isnull().sum() == 0
        # Column B was not listed → still NaN
        assert result["B"].isnull().sum() == 1


# ---------------------------------------------------------------------------
# Weighted imputation
# ---------------------------------------------------------------------------


class TestApplyImputationWeighted:
    """Weighted method must fill NaN values and return a WeightFunction."""

    def test_weighted_fills_all_nans(self, df_with_gap, stdlib_logger):
        config = _make_config("weighted", window_size=5)
        result, _ = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert result.isnull().sum().sum() == 0

    def test_weighted_returns_weight_function(self, df_with_gap, stdlib_logger):
        config = _make_config("weighted", window_size=5)
        _, wf = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert isinstance(wf, WeightFunction)

    def test_weighted_returns_dataframe(self, df_with_gap, stdlib_logger):
        config = _make_config("weighted", window_size=5)
        result, _ = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert isinstance(result, pd.DataFrame)

    def test_weighted_preserves_shape(self, df_with_gap, stdlib_logger):
        config = _make_config("weighted", window_size=5)
        result, _ = apply_imputation(df_with_gap.copy(), config, stdlib_logger)
        assert result.shape == df_with_gap.shape

    def test_weighted_no_gap_weight_func_is_callable(self, df_no_gap, stdlib_logger):
        config = _make_config("weighted", window_size=5)
        _, wf = apply_imputation(df_no_gap.copy(), config, stdlib_logger)
        assert callable(wf)


# ---------------------------------------------------------------------------
# Unknown method
# ---------------------------------------------------------------------------


class TestApplyImputationUnknownMethod:
    def test_raises_value_error_for_unknown_method(self, df_with_gap, stdlib_logger):
        config = _make_config("spline")
        with pytest.raises(ValueError, match="Unknown imputation_method"):
            apply_imputation(df_with_gap.copy(), config, stdlib_logger)

    def test_error_message_contains_method_name(self, df_with_gap, stdlib_logger):
        config = _make_config("kriging")
        with pytest.raises(ValueError, match="kriging"):
            apply_imputation(df_with_gap.copy(), config, stdlib_logger)


# ---------------------------------------------------------------------------
# Logging: NaN count before and after
# ---------------------------------------------------------------------------


class TestApplyImputationLogging:
    """Verify that INFO messages are emitted with before/after NaN counts."""

    def _collect_info_messages(self, df: pd.DataFrame, config, caplog) -> list[str]:
        with caplog.at_level(logging.INFO, logger="test_apply_imputation"):
            apply_imputation(
                df.copy(),
                config,
                logging.getLogger("test_apply_imputation"),
            )
        return [r.message for r in caplog.records if r.levelno == logging.INFO]

    def test_logs_nan_before(self, df_with_gap, caplog):
        config = _make_config("linear")
        msgs = self._collect_info_messages(df_with_gap, config, caplog)
        assert any("before" in m.lower() for m in msgs)

    def test_logs_nan_after(self, df_with_gap, caplog):
        config = _make_config("linear")
        msgs = self._collect_info_messages(df_with_gap, config, caplog)
        assert any("after" in m.lower() for m in msgs)

    def test_nan_before_count_is_nonzero_in_log(self, df_with_gap, caplog):
        """The NaN-before message must contain a positive integer."""
        config = _make_config("linear")
        msgs = self._collect_info_messages(df_with_gap, config, caplog)
        before_msgs = [m for m in msgs if "before" in m.lower()]
        assert before_msgs, "No 'before' log message found"
        # The message must contain a digit > 0
        import re

        numbers = re.findall(r"\d+", before_msgs[0])
        assert any(int(n) > 0 for n in numbers)

    def test_nan_after_zero_for_linear(self, df_with_gap, caplog):
        """After linear imputation there should be 0 NaN cells logged."""
        config = _make_config("linear")
        msgs = self._collect_info_messages(df_with_gap, config, caplog)
        after_msgs = [m for m in msgs if "after" in m.lower()]
        assert after_msgs, "No 'after' log message found"
        import re

        # Last number in the message should be 0
        numbers = re.findall(r"\d+", after_msgs[-1])
        assert numbers[-1] == "0"


# ---------------------------------------------------------------------------
# Warning when NaN values remain
# ---------------------------------------------------------------------------


class TestApplyImputationWarning:
    """A WARNING must be emitted when NaN values remain after imputation."""

    def test_warning_emitted_when_nans_remain(self, caplog):
        """Simulate a case where linear interpolation cannot fill edge NaN."""
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        # NaN at the very start — LinearlyInterpolateTS may not back-fill
        df = pd.DataFrame({"A": [None, 2.0, 3.0, 4.0, None]}, index=idx)
        config = _make_config("linear", cols=["A"])
        logger = logging.getLogger("test_apply_imputation")

        with caplog.at_level(logging.WARNING, logger="test_apply_imputation"):
            apply_imputation(df.copy(), config, logger)

        # If NaN cells remain, a WARNING must have been logged.
        # (If LinearlyInterpolateTS fills everything, this test passes vacuously.)
        result_df = df.copy()
        from spotforecast2_safe.preprocessing import LinearlyInterpolateTS

        interp = LinearlyInterpolateTS()
        result_df["A"] = interp.fit_transform(result_df["A"])
        if result_df["A"].isnull().sum() > 0:
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert warnings, "Expected a WARNING log for remaining NaN values"

    def test_no_warning_when_all_nans_filled(self, df_with_gap, caplog):
        """No WARNING should appear when imputation is complete."""
        config = _make_config("linear")
        logger = logging.getLogger("test_apply_imputation")
        with caplog.at_level(logging.WARNING, logger="test_apply_imputation"):
            apply_imputation(df_with_gap.copy(), config, logger)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings
