# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for the public ``agg_predictor`` function and related API.

Covers:
- Import paths for the standalone ``agg_predictor`` function
- Public vs. private API contract (no ``_agg_predictor`` on BaseTask)
- Aggregation logic (weighted sums, equal-weight fallback, test_actual handling)
- ``agg_results`` attribute initialisation and population
- ``_aggregate_and_show`` always-aggregate guarantee (even without agg_weights)
- Every task (lazy, train, optuna, spotoptim) concludes with aggregated values
- ``agg_predictor`` method on BaseTask and all subclasses
"""

import numpy as np
import pandas as pd

from spotforecast2.manager.multitask import (
    BaseTask,
    LazyTask,
    MultiTask,
    OptunaTask,
    SpotOptimTask,
    agg_predictor,
)
from spotforecast2.manager.multitask.base import agg_predictor as _agg_predictor_base
from spotforecast2.manager import agg_predictor as _agg_predictor_manager

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_pred_pkg(n_train: int = 100, n_future: int = 24, seed: int = 0) -> dict:
    """Return a minimal prediction package as built by build_prediction_package."""
    rng = np.random.default_rng(seed)
    idx_train = pd.date_range("2024-01-01", periods=n_train, freq="h")
    idx_future = pd.date_range("2024-05-01", periods=n_future, freq="h")
    return {
        "train_actual": pd.Series(rng.random(n_train), index=idx_train),
        "train_pred": pd.Series(rng.random(n_train), index=idx_train),
        "future_actual": pd.Series(dtype="float64"),
        "future_pred": pd.Series(rng.random(n_future), index=idx_future),
        "test_actual": pd.Series(rng.random(n_future), index=idx_future),
        "metrics_train": {},
        "metrics_future": {},
        "metrics_future_one_day": {},
        "validation_passed": True,
    }


def _make_pred_pkg_no_test(
    n_train: int = 100, n_future: int = 24, seed: int = 0
) -> dict:
    """Prediction package without a test_actual key."""
    pkg = _make_pred_pkg(n_train, n_future, seed)
    del pkg["test_actual"]
    return pkg


# ---------------------------------------------------------------------------
# 1. Import path tests
# ---------------------------------------------------------------------------


class TestImportPaths:
    """agg_predictor must be importable from all expected locations."""

    def test_import_from_multitask_package(self):
        from spotforecast2.manager.multitask import agg_predictor as ap

        assert callable(ap)

    def test_import_from_multitask_base(self):
        from spotforecast2.manager.multitask.base import agg_predictor as ap

        assert callable(ap)

    def test_import_from_manager_package(self):
        from spotforecast2.manager import agg_predictor as ap

        assert callable(ap)

    def test_all_paths_same_object(self):
        assert agg_predictor is _agg_predictor_base
        assert agg_predictor is _agg_predictor_manager

    def test_in_multitask_all(self):
        from spotforecast2.manager.multitask import __all__

        assert "agg_predictor" in __all__

    def test_in_manager_all(self):
        from spotforecast2.manager import __all__

        assert "agg_predictor" in __all__


# ---------------------------------------------------------------------------
# 2. Public API contract — no more _agg_predictor
# ---------------------------------------------------------------------------


class TestPublicAPIContract:
    """agg_predictor is public; _agg_predictor no longer exists."""

    def test_base_task_has_agg_predictor_method(self):
        assert hasattr(BaseTask, "agg_predictor")

    def test_base_task_has_no_private_agg_predictor(self):
        assert not hasattr(
            BaseTask, "_agg_predictor"
        ), "_agg_predictor must not exist — it was renamed to agg_predictor"

    def test_lazy_task_inherits_method(self):
        assert hasattr(LazyTask, "agg_predictor")

    def test_optuna_task_inherits_method(self):
        assert hasattr(OptunaTask, "agg_predictor")

    def test_spotoptim_task_inherits_method(self):
        assert hasattr(SpotOptimTask, "agg_predictor")

    def test_multi_task_inherits_method(self):
        assert hasattr(MultiTask, "agg_predictor")

    def test_method_callable_on_instance(self):
        task = MultiTask()
        assert callable(task.agg_predictor)


# ---------------------------------------------------------------------------
# 3. Standalone agg_predictor function — aggregation logic
# ---------------------------------------------------------------------------


class TestAggPredictorFunction:
    """Verify the standalone agg_predictor aggregation logic."""

    def setup_method(self):
        self.pkg_a = _make_pred_pkg(seed=1)
        self.pkg_b = _make_pred_pkg(seed=2)
        self.results = {"A": self.pkg_a, "B": self.pkg_b}

    def test_returns_dict(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert isinstance(out, dict)

    def test_required_keys_present(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        for key in (
            "train_actual",
            "train_pred",
            "future_pred",
            "future_actual",
            "metrics_train",
            "metrics_future",
            "metrics_future_one_day",
            "validation_passed",
        ):
            assert key in out, f"Missing key: {key}"

    def test_future_pred_is_series(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert isinstance(out["future_pred"], pd.Series)

    def test_train_actual_is_series(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert isinstance(out["train_actual"], pd.Series)

    def test_future_pred_length(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert len(out["future_pred"]) == 24

    def test_train_actual_length(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert len(out["train_actual"]) == 100

    def test_validation_passed_is_true(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert out["validation_passed"] is True

    def test_metrics_are_empty_dicts(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert out["metrics_train"] == {}
        assert out["metrics_future"] == {}

    def test_test_actual_included_when_present(self):
        out = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        assert "test_actual" in out

    def test_test_actual_absent_when_missing_from_all(self):
        pkg_no_test_a = _make_pred_pkg_no_test(seed=10)
        pkg_no_test_b = _make_pred_pkg_no_test(seed=20)
        out = agg_predictor(
            {"A": pkg_no_test_a, "B": pkg_no_test_b}, ["A", "B"], [0.5, 0.5]
        )
        assert "test_actual" not in out

    def test_single_target_weight_one_reproduces_future_pred(self):
        """A single target with weight=1 must reproduce its future_pred."""
        out = agg_predictor({"A": self.pkg_a}, ["A"], [1.0])
        pd.testing.assert_series_equal(
            out["future_pred"], self.pkg_a["future_pred"], check_names=False
        )

    def test_equal_weights_differ_from_full_weight_on_a(self):
        """Equal weights must differ from putting 100% weight on A."""
        out_eq = agg_predictor(self.results, ["A", "B"], [0.5, 0.5])
        out_a_only = agg_predictor(self.results, ["A", "B"], [1.0, 0.0])
        assert not out_eq["future_pred"].equals(out_a_only["future_pred"])

    def test_weights_sum_to_different_values_is_allowed(self):
        """Non-unit-sum weights are delegated to agg_predict — no error raised."""
        out = agg_predictor(self.results, ["A", "B"], [2.0, 1.0])
        assert isinstance(out["future_pred"], pd.Series)

    def test_three_targets(self):
        pkg_c = _make_pred_pkg(seed=3)
        results3 = {"A": self.pkg_a, "B": self.pkg_b, "C": pkg_c}
        out = agg_predictor(results3, ["A", "B", "C"], [1 / 3, 1 / 3, 1 / 3])
        assert isinstance(out["future_pred"], pd.Series)
        assert len(out["future_pred"]) == 24


# ---------------------------------------------------------------------------
# 4. BaseTask.agg_predictor method
# ---------------------------------------------------------------------------


class TestAggPredictorMethod:
    """BaseTask.agg_predictor delegates to the module-level function."""

    def setup_method(self):
        self.task = MultiTask()
        self.pkg_a = _make_pred_pkg(seed=10)
        self.pkg_b = _make_pred_pkg(seed=20)
        self.results = {"X": self.pkg_a, "Y": self.pkg_b}

    def test_method_returns_same_as_function(self):
        method_out = self.task.agg_predictor(self.results, ["X", "Y"], [0.5, 0.5])
        function_out = agg_predictor(self.results, ["X", "Y"], [0.5, 0.5])
        pd.testing.assert_series_equal(
            method_out["future_pred"], function_out["future_pred"]
        )

    def test_method_returns_dict(self):
        out = self.task.agg_predictor(self.results, ["X", "Y"], [0.5, 0.5])
        assert isinstance(out, dict)

    def test_method_returns_required_keys(self):
        out = self.task.agg_predictor(self.results, ["X", "Y"], [0.5, 0.5])
        assert "future_pred" in out
        assert "train_actual" in out


# ---------------------------------------------------------------------------
# 5. agg_results attribute
# ---------------------------------------------------------------------------


class TestAggResultsAttribute:
    """Every BaseTask subclass stores aggregation output in self.agg_results."""

    def test_multi_task_has_agg_results(self):
        assert hasattr(MultiTask(), "agg_results")

    def test_lazy_task_has_agg_results(self):
        assert hasattr(LazyTask(), "agg_results")

    def test_optuna_task_has_agg_results(self):
        assert hasattr(OptunaTask(), "agg_results")

    def test_spotoptim_task_has_agg_results(self):
        assert hasattr(SpotOptimTask(), "agg_results")

    def test_initialises_as_empty_dict(self):
        task = MultiTask()
        assert task.agg_results == {}

    def test_is_dict_type(self):
        task = MultiTask()
        assert isinstance(task.agg_results, dict)


# ---------------------------------------------------------------------------
# 6. _aggregate_and_show — always-aggregate guarantee
# ---------------------------------------------------------------------------


class TestAggregateAndShowAlwaysAggregates:
    """_aggregate_and_show must return a dict even without agg_weights."""

    def _make_task_with_targets(self, targets):
        task = MultiTask()
        task.config.targets = targets
        return task

    def test_returns_dict_without_agg_weights(self):
        task = self._make_task_with_targets(["A", "B"])
        task.config.agg_weights = None
        results = {"A": _make_pred_pkg(seed=1), "B": _make_pred_pkg(seed=2)}
        out = task._aggregate_and_show(results, "test_task_no_weights", show=False)
        assert isinstance(out, dict)

    def test_returns_dict_with_agg_weights(self):
        task = self._make_task_with_targets(["A", "B"])
        task.config.agg_weights = [0.6, 0.4]
        results = {"A": _make_pred_pkg(seed=1), "B": _make_pred_pkg(seed=2)}
        out = task._aggregate_and_show(results, "test_task_weights", show=False)
        assert isinstance(out, dict)

    def test_stores_in_agg_results_without_weights(self):
        task = self._make_task_with_targets(["A", "B"])
        task.config.agg_weights = None
        results = {"A": _make_pred_pkg(seed=1), "B": _make_pred_pkg(seed=2)}
        task._aggregate_and_show(results, "my_task_key", show=False)
        assert "my_task_key" in task.agg_results

    def test_stores_in_agg_results_with_weights(self):
        task = self._make_task_with_targets(["A", "B"])
        task.config.agg_weights = [0.7, 0.3]
        results = {"A": _make_pred_pkg(seed=1), "B": _make_pred_pkg(seed=2)}
        task._aggregate_and_show(results, "weighted_task", show=False)
        assert "weighted_task" in task.agg_results
        assert isinstance(task.agg_results["weighted_task"], dict)

    def test_equal_weight_fallback_uses_1_over_n(self):
        """Without agg_weights, each target should get weight 1/n."""
        task = self._make_task_with_targets(["A", "B"])
        task.config.agg_weights = None
        pkg_a = _make_pred_pkg(seed=5)
        pkg_b = _make_pred_pkg(seed=6)
        results = {"A": pkg_a, "B": pkg_b}

        out_eq = task._aggregate_and_show(results, "eq", show=False)
        out_manual = agg_predictor(results, ["A", "B"], [0.5, 0.5])
        pd.testing.assert_series_equal(out_eq["future_pred"], out_manual["future_pred"])

    def test_single_target_equal_weight_is_one(self):
        task = self._make_task_with_targets(["A"])
        task.config.agg_weights = None
        results = {"A": _make_pred_pkg(seed=7)}
        out = task._aggregate_and_show(results, "single_target", show=False)
        assert isinstance(out, dict)

    def test_result_is_stored_keyed_by_task_name(self):
        task = self._make_task_with_targets(["A"])
        task.config.agg_weights = None
        results = {"A": _make_pred_pkg(seed=8)}
        task._aggregate_and_show(results, "sentinel_key", show=False)
        assert "sentinel_key" in task.agg_results
        assert task.agg_results["sentinel_key"]["validation_passed"] is True


# ---------------------------------------------------------------------------
# 7. Task-level contract: run() returns aggregated values
# ---------------------------------------------------------------------------


def _inject_pipeline(task, n_rows=300):
    """Inject a synthetic pipeline state so a task can run without I/O."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=n_rows, freq="h", tz="UTC")
    df = pd.DataFrame(rng.random((n_rows, 2)), index=idx, columns=["A", "B"])
    # df_test must have DateTime as a column (matching prepare_data / reset_index format)
    df_test = df.iloc[-24:].copy().reset_index().rename(columns={"index": "DateTime"})
    task.df_pipeline = df
    task.df_test = df_test
    task.config.targets = ["A", "B"]
    task.config.agg_weights = [0.5, 0.5]
    task.data_with_exog = None
    task.exo_pred = None
    task.exog_feature_names = []
    task.config.end_train_ts = idx[-25]
    task.config.start_train_ts = idx[0]
    return task


class TestTaskRunReturnsAggregatedValues:
    """execute_* functions and run() must return Dict[str, Any] (aggregated pkg)."""

    def test_execute_lazy_returns_aggregated_keys(self):
        from spotforecast2.manager.multitask.lazy import execute_lazy

        task = _inject_pipeline(LazyTask(use_exogenous_features=False, predict_size=24))
        result = execute_lazy(task, show=False)
        assert isinstance(result, dict)
        assert (
            "future_pred" in result
        ), "execute_lazy must return aggregated pkg with 'future_pred'"
        assert "train_actual" in result

    def test_execute_lazy_stores_per_target_in_results(self):
        from spotforecast2.manager.multitask.lazy import execute_lazy

        task = _inject_pipeline(LazyTask(use_exogenous_features=False, predict_size=24))
        execute_lazy(task, show=False)
        assert "lazy" in task.results
        assert "A" in task.results["lazy"]
        assert "B" in task.results["lazy"]

    def test_execute_lazy_populates_agg_results(self):
        from spotforecast2.manager.multitask.lazy import execute_lazy

        task = _inject_pipeline(LazyTask(use_exogenous_features=False, predict_size=24))
        execute_lazy(task, show=False)
        assert len(task.agg_results) > 0


# ---------------------------------------------------------------------------
# 8. Quartodoc / documentation — agg_predictor has a proper docstring
# ---------------------------------------------------------------------------


class TestDocumentation:
    """agg_predictor must have a meaningful docstring for quartodoc."""

    def test_module_level_function_has_docstring(self):
        assert agg_predictor.__doc__ is not None
        assert len(agg_predictor.__doc__.strip()) > 30

    def test_method_has_docstring(self):
        task = MultiTask()
        doc = task.agg_predictor.__doc__
        assert doc is not None
        assert len(doc.strip()) > 20

    def test_function_signature_has_results_param(self):
        import inspect

        sig = inspect.signature(agg_predictor)
        assert "results" in sig.parameters

    def test_function_signature_has_targets_param(self):
        import inspect

        sig = inspect.signature(agg_predictor)
        assert "targets" in sig.parameters

    def test_function_signature_has_weights_param(self):
        import inspect

        sig = inspect.signature(agg_predictor)
        assert "weights" in sig.parameters
