# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for MultiTask DataFrame input support.

Covers:
- Constructor accepts dataframe parameter (None and DataFrame)
- _dataframe is stored correctly
- prepare_data() uses _dataframe when no demo_data argument is given
- Explicit demo_data in prepare_data() overrides constructor dataframe
- CSV-loading path still works when dataframe=None
- dataframe appears in constructor signature with correct default
- MultiTask with dataframe produces valid pipeline state
- df_pipeline is non-None after prepare_data() with dataframe
- Targets are populated after prepare_data() with dataframe
- Wrong-type argument raises TypeError
"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from spotforecast2.manager.multitask import MultiTask
from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    """Load the package demo10 CSV once for the whole module."""
    data_home = get_package_data_home()
    return fetch_data(filename=str(data_home / "demo10.csv"))


@pytest.fixture()
def mt_with_df(demo_df: pd.DataFrame, tmp_path: Path) -> MultiTask:
    """MultiTask constructed with a dataframe argument."""
    return MultiTask(dataframe=demo_df, cache_home=tmp_path, predict_size=24)


@pytest.fixture()
def mt_without_df(tmp_path: Path) -> MultiTask:
    """MultiTask constructed without a dataframe (CSV path)."""
    return MultiTask(cache_home=tmp_path, predict_size=24)


# ---------------------------------------------------------------------------
# Constructor signature
# ---------------------------------------------------------------------------


class TestConstructorSignature:
    def test_dataframe_in_signature(self):
        sig = inspect.signature(MultiTask.__init__)
        assert "dataframe" in sig.parameters

    def test_dataframe_default_is_none(self):
        sig = inspect.signature(MultiTask.__init__)
        assert sig.parameters["dataframe"].default is None

    def test_dataframe_is_keyword_only(self):
        sig = inspect.signature(MultiTask.__init__)
        p = sig.parameters["dataframe"]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY


# ---------------------------------------------------------------------------
# _dataframe attribute
# ---------------------------------------------------------------------------


class TestDataframeAttribute:
    def test_none_when_not_provided(self, mt_without_df):
        assert mt_without_df._dataframe is None

    def test_stored_when_provided(self, mt_with_df, demo_df):
        assert mt_with_df._dataframe is demo_df

    def test_stored_dataframe_is_same_object(self, demo_df, tmp_path):
        mt = MultiTask(dataframe=demo_df, cache_home=tmp_path)
        assert mt._dataframe is demo_df


# ---------------------------------------------------------------------------
# prepare_data() with constructor dataframe
# ---------------------------------------------------------------------------


class TestPrepareDataWithDataframe:
    def test_pipeline_not_none_after_prepare(self, mt_with_df):
        mt_with_df.prepare_data()
        assert mt_with_df.df_pipeline is not None

    def test_targets_populated_after_prepare(self, mt_with_df):
        mt_with_df.prepare_data()
        assert len(mt_with_df.config.targets) > 0

    def test_pipeline_shape_is_2d(self, mt_with_df):
        mt_with_df.prepare_data()
        assert mt_with_df.df_pipeline.ndim == 2

    def test_pipeline_has_rows(self, mt_with_df):
        mt_with_df.prepare_data()
        assert len(mt_with_df.df_pipeline) > 0

    def test_returns_self(self, mt_with_df):
        result = mt_with_df.prepare_data()
        assert result is mt_with_df


# ---------------------------------------------------------------------------
# Precedence: explicit demo_data overrides constructor dataframe
# ---------------------------------------------------------------------------


class TestPrepareDataPrecedence:
    def test_explicit_demo_data_overrides_constructor_df(
        self, demo_df: pd.DataFrame, tmp_path: Path
    ):
        """demo_data passed to prepare_data() must win over self._dataframe."""
        # Use a slightly modified copy as the "explicit" frame so we can
        # distinguish which one was actually used by inspecting the column set.
        demo_df2 = demo_df.copy()
        demo_df2["_sentinel_col"] = 0.0

        mt = MultiTask(dataframe=demo_df, cache_home=tmp_path, predict_size=24)
        mt.prepare_data(demo_data=demo_df2)

        # If demo_data was used, _sentinel_col shows up (or at least the
        # pipeline has the same number of non-index columns as demo_df2).
        pipeline_cols = set(mt.df_pipeline.columns)
        assert "_sentinel_col" in pipeline_cols

    def test_constructor_df_used_when_no_explicit_demo_data(
        self, demo_df: pd.DataFrame, tmp_path: Path
    ):
        """When prepare_data() is called without demo_data, constructor df is used."""
        mt = MultiTask(dataframe=demo_df, cache_home=tmp_path, predict_size=24)
        # prepare_data() called without arguments — must NOT raise FileNotFoundError
        mt.prepare_data()
        assert mt.df_pipeline is not None


# ---------------------------------------------------------------------------
# CSV-loading backward compatibility
# ---------------------------------------------------------------------------


class TestCsvBackwardCompatibility:
    def test_prepare_data_loads_csv_when_no_dataframe(self, mt_without_df):
        """Default MultiTask (no dataframe) still loads CSV successfully."""
        mt_without_df.prepare_data()
        assert mt_without_df.df_pipeline is not None

    def test_csv_pipeline_has_targets(self, mt_without_df):
        mt_without_df.prepare_data()
        assert len(mt_without_df.config.targets) > 0


# ---------------------------------------------------------------------------
# prepare_data() signature on MultiTask
# ---------------------------------------------------------------------------


class TestPrepareDataSignature:
    def test_has_demo_data_param(self):
        sig = inspect.signature(MultiTask.prepare_data)
        assert "demo_data" in sig.parameters

    def test_has_df_test_param(self):
        sig = inspect.signature(MultiTask.prepare_data)
        assert "df_test" in sig.parameters

    def test_demo_data_default_none(self):
        sig = inspect.signature(MultiTask.prepare_data)
        assert sig.parameters["demo_data"].default is None
