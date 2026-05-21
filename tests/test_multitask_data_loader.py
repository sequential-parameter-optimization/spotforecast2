# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``config.data_loader`` hook on ``BaseTask.prepare_data``
(ADR-001 Step 7).

The ENTSO-E integration that follows this ADR carries its data-acquisition
step (``download_new_data`` + ``merge_build_manual``) in a ``data_loader``
callable rather than passing a DataFrame in.  These tests exercise the hook
directly without needing the actual ENTSO-E API.
"""

import pandas as pd
import pytest

from spotforecast2.multitask import LazyTask


def _synthetic_df() -> pd.DataFrame:
    """Tiny hourly DataFrame with a single target column.

    Indexed by a tz-aware DateTime index — the shape ``reset_index`` (from
    ``spotforecast2_safe.preprocessing.curate_data``) expects.
    """
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame({"load": list(range(48))}, index=idx)


def test_data_loader_invoked_when_no_dataframe_supplied():
    """``prepare_data`` falls back to ``config.data_loader(config)`` when
    neither a constructor ``dataframe`` nor a ``demo_data`` argument is
    supplied.  The callable is invoked exactly once."""
    calls = []

    def loader(config):
        calls.append(config)
        return _synthetic_df()

    task = LazyTask(predict_size=24, use_outlier_detection=False)
    task.config.data_loader = loader

    task.prepare_data()

    assert len(calls) == 1
    assert calls[0] is task.config
    assert task.df_pipeline is not None
    assert "load" in task.df_pipeline.columns


def test_explicit_dataframe_skips_data_loader():
    """When a DataFrame is supplied via the constructor, the loader hook is
    not invoked (avoiding silent extra work)."""
    calls = []

    def loader(config):  # pragma: no cover — must not run
        calls.append(config)
        return _synthetic_df()

    task = LazyTask(
        dataframe=_synthetic_df(),
        predict_size=24,
        use_outlier_detection=False,
    )
    task.config.data_loader = loader

    task.prepare_data()

    assert calls == []
    assert task.df_pipeline is not None


def test_no_data_and_no_loader_raises_value_error():
    """When neither a DataFrame nor a loader is supplied, the historical
    ValueError is preserved (with an updated message mentioning the hook)."""
    task = LazyTask(predict_size=24)
    with pytest.raises(ValueError, match=r"data_loader"):
        task.prepare_data()
