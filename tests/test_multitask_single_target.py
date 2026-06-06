# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Single-target awareness in the multitask pipeline (ADR-001 Step 6).

The ENTSO-E integration that follows this ADR will instantiate the
multitask pipeline with a single-column DataFrame.  Before that, the
``_aggregate_and_show`` step unconditionally invoked ``agg_predictor``
across multiple targets, which either errored or silently returned a
meaningless single-target "aggregation".  Step 6 makes the aggregate step
a pass-through for single-target configurations.  These tests exercise
that seam directly without requiring the full data-acquisition / weather /
calendar machinery.
"""

import pandas as pd
import pytest

from spotforecast2.multitask import LazyTask


@pytest.fixture
def single_target_lazy() -> LazyTask:
    """A LazyTask with a single configured target and a deterministic result.

    Bypasses ``prepare_data`` by setting only what ``_aggregate_and_show``
    requires.  The goal is to verify the single-target branch in isolation,
    not to integration-test the full pipeline.
    """
    task = LazyTask(predict_size=24, auto_save_models=False)
    task.run_state.targets = ["only_target"]
    return task


def test_aggregate_and_show_passes_through_single_target(single_target_lazy):
    """For a one-target config, the aggregator must return the per-target
    package unchanged and must not call the multi-target ``agg_predictor``."""
    pkg = {"future_pred": pd.Series([1.0, 2.0, 3.0]), "marker": "single"}
    results = {"only_target": pkg}

    out = single_target_lazy._aggregate_and_show(
        results, task_name="single-target lazy", show=False
    )

    assert out is pkg
    assert (
        single_target_lazy.agg_results["single-target lazy"] is pkg
    ), "single-target aggregation must still register the result in agg_results"


def test_aggregate_and_show_skips_extra_figure_in_single_target(
    single_target_lazy, monkeypatch
):
    """The single-target fast path should not produce a duplicate aggregated
    figure even with ``show=True``: the per-target figure is the only one.
    """
    import spotforecast2.multitask.base as base_mod

    calls = {"made_plot": 0}

    def _recording_make_plot(*args, **kwargs):  # pragma: no cover — should not run
        calls["made_plot"] += 1
        raise AssertionError("single-target aggregation must not draw an extra figure")

    monkeypatch.setattr(base_mod, "make_plot", _recording_make_plot)
    pkg = {"future_pred": pd.Series([1.0])}
    single_target_lazy._aggregate_and_show(
        {"only_target": pkg}, task_name="t", show=True
    )
    assert calls["made_plot"] == 0


def test_multi_target_aggregation_still_runs(monkeypatch):
    """Sanity: multi-target configurations still invoke the aggregator
    (regression guard for Step 6's single-target short-circuit)."""
    task = LazyTask(predict_size=24, auto_save_models=False)
    task.run_state.targets = ["a", "b"]
    task.config.agg_weights = [0.5, 0.5]

    seen = {}

    def fake_agg_predictor(*, results, targets, weights):
        seen["targets"] = list(targets)
        seen["weights"] = list(weights)
        return {"future_pred": "aggregated"}

    monkeypatch.setattr(task, "agg_predictor", fake_agg_predictor)
    out = task._aggregate_and_show(
        {"a": {"future_pred": "A"}, "b": {"future_pred": "B"}},
        task_name="multi",
        show=False,
    )
    assert out == {"future_pred": "aggregated"}
    assert seen["targets"] == ["a", "b"]
    assert seen["weights"] == [0.5, 0.5]
