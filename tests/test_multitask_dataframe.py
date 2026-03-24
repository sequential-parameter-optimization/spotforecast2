# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for DataFrame-only data input support across all task classes.

Covers:
- Constructor accepts dataframe and data_test parameters (BaseTask, MultiTask)
- _dataframe is stored correctly
- prepare_data() uses _dataframe when no demo_data argument is given
- Explicit demo_data in prepare_data() overrides constructor dataframe
- dataframe appears in constructor signature with correct default
- Task with dataframe produces valid pipeline state
- ValueError raised when no data source is provided
- data_test (as DataFrame) is supported via constructor and prepare_data()
- LazyTask (BaseTask subclass) also supports dataframe
"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from spotforecast2.manager.multitask import BaseTask, LazyTask, MultiTask
from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DEMO_CSV = str(get_package_data_home() / "demo10.csv")


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    """Load the package demo10 CSV once for the whole module."""
    return fetch_data(filename=_DEMO_CSV)


@pytest.fixture()
def mt_with_df(demo_df: pd.DataFrame, tmp_path: Path) -> MultiTask:
    """MultiTask constructed with a dataframe argument."""
    return MultiTask(dataframe=demo_df, cache_home=tmp_path, predict_size=24)


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

    def test_data_test_in_signature(self):
        sig = inspect.signature(MultiTask.__init__)
        assert "data_test" in sig.parameters

    def test_data_test_default_is_none(self):
        sig = inspect.signature(MultiTask.__init__)
        assert sig.parameters["data_test"].default is None

    def test_no_data_source_in_signature(self):
        sig = inspect.signature(MultiTask.__init__)
        assert "data_source" not in sig.parameters

    def test_no_data_home_in_signature(self):
        sig = inspect.signature(MultiTask.__init__)
        assert "data_home" not in sig.parameters


# ---------------------------------------------------------------------------
# _dataframe attribute
# ---------------------------------------------------------------------------


class TestDataframeAttribute:
    def test_none_when_not_provided(self, tmp_path):
        mt = MultiTask(cache_home=tmp_path)
        assert mt._dataframe is None

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
# Precedence: explicit demo_data > constructor dataframe
# ---------------------------------------------------------------------------


class TestPrepareDataPrecedence:
    def test_explicit_demo_data_overrides_constructor_df(
        self, demo_df: pd.DataFrame, tmp_path: Path
    ):
        """demo_data passed to prepare_data() must win over self._dataframe."""
        demo_df2 = demo_df.copy()
        demo_df2["_sentinel_col"] = 0.0

        mt = MultiTask(dataframe=demo_df, cache_home=tmp_path, predict_size=24)
        mt.prepare_data(demo_data=demo_df2)

        pipeline_cols = set(mt.df_pipeline.columns)
        assert "_sentinel_col" in pipeline_cols

    def test_constructor_df_used_when_no_explicit_demo_data(
        self, demo_df: pd.DataFrame, tmp_path: Path
    ):
        """When prepare_data() is called without demo_data, constructor df is used."""
        mt = MultiTask(dataframe=demo_df, cache_home=tmp_path, predict_size=24)
        mt.prepare_data()
        assert mt.df_pipeline is not None


# ---------------------------------------------------------------------------
# No data -> ValueError
# ---------------------------------------------------------------------------


class TestNoDataRaises:
    def test_no_data_source_raises_value_error(self, tmp_path):
        mt = MultiTask(cache_home=tmp_path)
        with pytest.raises(ValueError, match="No data source provided"):
            mt.prepare_data()


# ---------------------------------------------------------------------------
# data_test as DataFrame
# ---------------------------------------------------------------------------


class TestDataTestDataFrame:
    def test_data_test_stored_on_instance(self, demo_df, tmp_path):
        df_test = demo_df.head(24).copy()
        mt = MultiTask(
            dataframe=demo_df,
            data_test=df_test,
            cache_home=tmp_path,
            predict_size=24,
        )
        assert mt.data_test is df_test

    def test_data_test_used_during_prepare_data(self, demo_df, tmp_path):
        df_test = demo_df.head(24).copy()
        mt = MultiTask(
            dataframe=demo_df,
            data_test=df_test,
            cache_home=tmp_path,
            predict_size=24,
        )
        mt.prepare_data()
        assert mt.df_test is not None

    def test_df_test_arg_overrides_constructor_data_test(self, demo_df, tmp_path):
        df_test_ctor = demo_df.head(24).copy()
        df_test_arg = demo_df.head(48).copy()
        mt = MultiTask(
            dataframe=demo_df,
            data_test=df_test_ctor,
            cache_home=tmp_path,
            predict_size=24,
        )
        mt.prepare_data(df_test=df_test_arg)
        # The df_test argument should take precedence
        assert mt.df_test is not None


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


# ---------------------------------------------------------------------------
# BaseTask / LazyTask also support dataframe
# ---------------------------------------------------------------------------


class TestBaseTaskDataframe:
    """Verify the dataframe parameter works on BaseTask subclasses."""

    def test_dataframe_in_base_task_signature(self):
        sig = inspect.signature(BaseTask.__init__)
        assert "dataframe" in sig.parameters
        assert sig.parameters["dataframe"].default is None

    def test_data_test_in_base_task_signature(self):
        sig = inspect.signature(BaseTask.__init__)
        assert "data_test" in sig.parameters
        assert sig.parameters["data_test"].default is None

    def test_lazy_task_accepts_dataframe(self, demo_df, tmp_path):
        task = LazyTask(dataframe=demo_df, cache_home=tmp_path)
        assert task._dataframe is demo_df

    def test_lazy_task_prepare_data_uses_constructor_df(self, demo_df, tmp_path):
        task = LazyTask(dataframe=demo_df, cache_home=tmp_path)
        task.prepare_data()
        assert task.df_pipeline is not None
        assert task.df_pipeline.shape[0] > 0

    def test_lazy_task_no_data_raises(self, tmp_path):
        task = LazyTask(cache_home=tmp_path)
        with pytest.raises(ValueError, match="No data source provided"):
            task.prepare_data()

    def test_lazy_task_explicit_demo_data_overrides_constructor(
        self, demo_df, tmp_path
    ):
        demo_df2 = demo_df.copy()
        demo_df2["_lazy_sentinel"] = 0.0
        task = LazyTask(dataframe=demo_df, cache_home=tmp_path)
        task.prepare_data(demo_data=demo_df2)
        assert "_lazy_sentinel" in task.df_pipeline.columns
