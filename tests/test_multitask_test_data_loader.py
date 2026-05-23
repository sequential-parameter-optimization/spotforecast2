# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ``config.test_data_loader`` hook on ``BaseTask.prepare_data``.

Mirrors ``test_multitask_data_loader`` but for the test/ground-truth slice:
``test_data_loader(config) -> pd.DataFrame`` is invoked iff neither a
constructor ``data_test`` nor a ``prepare_data(df_test=...)`` argument was
supplied.  Used by the demo10 / ENTSO-E Quarto notebooks so the prediction
package gains a non-empty ``test_actual`` series and populated
``metrics_future`` without manual plumbing at the call site.
"""

import pandas as pd

from spotforecast2.multitask import LazyTask


def _synthetic_train_df() -> pd.DataFrame:
    """48 hourly observations of a single target — minimal training frame."""
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame({"load": list(range(48))}, index=idx)


def _synthetic_test_df() -> pd.DataFrame:
    """24 hourly observations immediately following the training frame."""
    idx = pd.date_range("2024-01-03", periods=24, freq="h", tz="UTC")
    return pd.DataFrame({"load": list(range(100, 124))}, index=idx)


def test_test_data_loader_invoked_when_no_data_test_supplied():
    """``prepare_data`` falls back to ``config.test_data_loader(config)`` when
    neither a constructor ``data_test`` nor a ``df_test`` argument is supplied."""
    calls = []

    def loader(config):
        calls.append(config)
        return _synthetic_test_df()

    task = LazyTask(
        dataframe=_synthetic_train_df(),
        predict_size=24,
        use_outlier_detection=False,
    )
    task.config.test_data_loader = loader

    task.prepare_data()

    assert len(calls) == 1
    assert calls[0] is task.config
    assert task.df_test is not None
    assert "load" in task.df_test.columns
    assert len(task.df_test) == 24


def test_explicit_data_test_skips_test_data_loader():
    """When ``data_test`` is supplied via the constructor, the loader hook is
    not invoked (matches the precedence rule)."""
    calls = []

    def loader(config):  # pragma: no cover — must not run
        calls.append(config)
        return _synthetic_test_df()

    task = LazyTask(
        dataframe=_synthetic_train_df(),
        data_test=_synthetic_test_df(),
        predict_size=24,
        use_outlier_detection=False,
    )
    task.config.test_data_loader = loader

    task.prepare_data()

    assert calls == []
    assert task.df_test is not None


def test_prepare_data_df_test_arg_skips_test_data_loader():
    """The ``df_test=`` argument to ``prepare_data`` wins over the loader hook."""
    calls = []

    def loader(config):  # pragma: no cover — must not run
        calls.append(config)
        return _synthetic_test_df()

    task = LazyTask(
        dataframe=_synthetic_train_df(),
        predict_size=24,
        use_outlier_detection=False,
    )
    task.config.test_data_loader = loader

    explicit = _synthetic_test_df()
    task.prepare_data(df_test=explicit)

    assert calls == []
    assert task.df_test is not None
    assert len(task.df_test) == 24


def test_no_data_test_and_no_loader_leaves_df_test_none():
    """Default behaviour is unchanged: with neither source, ``df_test`` stays
    ``None`` and the rest of the pipeline runs without a test-actuals slice."""
    task = LazyTask(
        dataframe=_synthetic_train_df(),
        predict_size=24,
        use_outlier_detection=False,
    )

    task.prepare_data()

    assert task.df_test is None
