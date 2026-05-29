# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the ``poly_features_degree`` / ``max_poly_features`` wiring.

Guards two behaviours that were previously broken or absent:

1. ``build_exogenous_features`` must forward ``config.poly_features_degree`` to
   ``create_interaction_features(degree=...)``. Historically the degree was never
   passed, so the builder ran at its default ``degree=1`` and produced *zero*
   interaction columns regardless of configuration.
2. When more than ``config.max_poly_features`` ``poly_*`` columns are generated,
   the pipeline must cap them to the top-K ranked by mutual information.

All heavy feature builders are patched at their import site in
``spotforecast2.multitask.base`` so no network / leaf library is touched.
``select_top_poly_features`` is intentionally NOT patched — the real mutual
information cap runs.
"""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd

from spotforecast2.multitask import MultiTask
from spotforecast2_safe.configurator.config_multi import ConfigMulti


def _make_task(poly_features_degree: int, max_poly_features: int) -> MultiTask:
    cfg = ConfigMulti(
        poly_features_degree=poly_features_degree,
        max_poly_features=max_poly_features,
    )
    mt = MultiTask(cfg)
    idx = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    mt.df_pipeline = pd.DataFrame({"target_0": np.arange(96.0)}, index=idx)
    mt.config.targets = ["target_0"]
    mt.config.data_start = idx[0]
    mt.config.cov_end = idx[-1]
    return mt


def _frame(index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: 0.0 for c in columns}, index=index)


def _patch_builders(stack: ExitStack, idx: pd.DatetimeIndex) -> None:
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.get_weather_features",
            return_value=(_frame(idx, ["temperature_2m"]), _frame(idx, ["temperature_2m"])),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.get_calendar_features",
            return_value=_frame(idx, ["hour"]),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.get_day_night_features",
            return_value=_frame(idx, ["is_daylight"]),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.get_holiday_features",
            return_value=_frame(idx, ["is_holiday"]),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.apply_cyclical_encoding",
            side_effect=lambda data, drop_original: data,
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.select_exogenous_features",
            side_effect=lambda exogenous_features, **_: list(exogenous_features.columns),
        )
    )
    stack.enter_context(
        patch(
            "spotforecast2.multitask.base.merge_data_and_covariates",
            side_effect=lambda data, exogenous_features, **_: (
                data,
                exogenous_features,
                exogenous_features,
            ),
        )
    )


def _fake_interactions(n_poly: int):
    """Return a create_interaction_features stub that injects ``n_poly`` poly_* columns."""

    def _impl(exogenous_features, weather_aligned, **kwargs):
        out = exogenous_features.copy()
        target = np.arange(len(out), dtype=float)
        rng = np.random.default_rng(0)
        for i in range(n_poly):
            # Decreasing signal-to-noise so mutual information has a clear order.
            out[f"poly_x{i}"] = target * (n_poly - i) + rng.normal(0, 1, len(out))
        return out

    return _impl


def test_degree_is_threaded_into_create_interaction_features():
    mt = _make_task(poly_features_degree=2, max_poly_features=10)
    idx = mt.df_pipeline.index
    with ExitStack() as stack:
        _patch_builders(stack, idx)
        mock_create = stack.enter_context(
            patch(
                "spotforecast2.multitask.base.create_interaction_features",
                side_effect=_fake_interactions(3),
            )
        )
        mt.build_exogenous_features()

    assert mock_create.call_args.kwargs["degree"] == 2


def test_poly_columns_capped_to_max():
    mt = _make_task(poly_features_degree=2, max_poly_features=3)
    idx = mt.df_pipeline.index
    with ExitStack() as stack:
        _patch_builders(stack, idx)
        stack.enter_context(
            patch(
                "spotforecast2.multitask.base.create_interaction_features",
                side_effect=_fake_interactions(8),
            )
        )
        mt.build_exogenous_features()

    poly_cols = [c for c in mt.exogenous_features.columns if c.startswith("poly_")]
    assert len(poly_cols) == 3


def test_no_cap_below_threshold():
    mt = _make_task(poly_features_degree=2, max_poly_features=10)
    idx = mt.df_pipeline.index
    with ExitStack() as stack:
        _patch_builders(stack, idx)
        stack.enter_context(
            patch(
                "spotforecast2.multitask.base.create_interaction_features",
                side_effect=_fake_interactions(4),
            )
        )
        mt.build_exogenous_features()

    poly_cols = [c for c in mt.exogenous_features.columns if c.startswith("poly_")]
    assert len(poly_cols) == 4
