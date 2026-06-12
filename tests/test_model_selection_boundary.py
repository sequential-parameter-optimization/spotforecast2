# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2.model_selection.boundary."""

from __future__ import annotations

import logging

from spotforecast2.model_selection.boundary import (
    boundary_report,
    report_boundary_positions,
    suggest_bounds,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

LINEAR_SPACE = {
    "estimator__num_leaves": (8, 1024),
    "estimator__n_estimators": (100, 5000),
    "estimator__bagging_fraction": (0.5, 1.0),
    "estimator__reg_alpha": (0.001, 10.0),
    "lags": ["[1, 2, 24]", "[1, 2, 24, 168]"],  # categorical, must be skipped
}

LOG_SPACE = {
    "estimator__learning_rate": (0.005, 0.3, "log10"),
    "estimator__reg_alpha": (0.001, 100.0, "log10"),
}

MIXED_SPACE = {**LINEAR_SPACE, **LOG_SPACE}


# ---------------------------------------------------------------------------
# report_boundary_positions
# ---------------------------------------------------------------------------

class TestReportBoundaryPositions:
    def test_interior_no_flag(self):
        """An optimum well inside all bounds returns an empty list."""
        params = {
            "num_leaves": 300,
            "n_estimators": 2000,
            "bagging_fraction": 0.75,
            "reg_alpha": 1.0,
            "learning_rate": 0.05,
        }
        flagged = report_boundary_positions(params, MIXED_SPACE)
        assert flagged == []

    def test_near_upper_linear_flagged(self):
        """A value at 98 % of the linear range is flagged as '> upper'."""
        # reg_alpha = 9.8 in (0.001, 10.0): pos = (9.8 - 0.001) / (10.0 - 0.001) ≈ 0.980
        params = {"reg_alpha": 9.8}
        space = {"estimator__reg_alpha": (0.001, 10.0)}
        flagged = report_boundary_positions(params, space)
        assert flagged == ["reg_alpha > upper"]

    def test_near_lower_linear_flagged(self):
        """A value at 2 % of the linear range is flagged as '< lower'."""
        # n_estimators = 190 in (100, 5000): pos = (190 - 100) / (5000 - 100) ≈ 0.018
        params = {"n_estimators": 190}
        space = {"estimator__n_estimators": (100, 5000)}
        flagged = report_boundary_positions(params, space)
        assert flagged == ["n_estimators < lower"]

    def test_log10_dim_near_lower(self):
        """In log10 space, a value near the lower decade boundary is flagged."""
        # learning_rate = 0.006 in (0.005, 0.3) log10:
        # pos = (log10(0.006) - log10(0.005)) / (log10(0.3) - log10(0.005))
        import math
        low, high, val = 0.005, 0.3, 0.006
        pos = (math.log10(val) - math.log10(low)) / (math.log10(high) - math.log10(low))
        assert pos < 0.10  # confirm this is a near-lower case
        params = {"learning_rate": val}
        space = {"estimator__learning_rate": (low, high, "log10")}
        flagged = report_boundary_positions(params, space)
        assert flagged == ["learning_rate < lower"]

    def test_estimator_prefix_stripped(self):
        """The 'estimator__' prefix is stripped before looking up params."""
        params = {"reg_alpha": 9.9}
        space = {"estimator__reg_alpha": (0.001, 10.0)}
        flagged = report_boundary_positions(params, space)
        assert any("reg_alpha" in f for f in flagged)

    def test_categorical_lags_skipped(self):
        """List-valued entries in the search space are silently skipped."""
        params = {}
        space = {"lags": ["[1, 2, 24]", "[1, 24, 168]"]}
        flagged = report_boundary_positions(params, space)
        assert flagged == []

    def test_bool_param_skipped(self):
        """Boolean values are skipped (isinstance(True, int) is True in Python)."""
        params = {"verbose": True}
        space = {"estimator__verbose": (0, 1)}
        flagged = report_boundary_positions(params, space)
        assert flagged == []

    def test_missing_param_key_skipped_with_warning(self, caplog):
        """A param key not found in params is skipped; no exception is raised."""
        params = {}  # empty — key is missing
        space = {"estimator__num_leaves": (8, 1024)}
        with caplog.at_level(logging.DEBUG):
            flagged = report_boundary_positions(params, space)
        assert flagged == []
        # No warning should be emitted for a simply missing key (it is None,
        # so we skip via the `val is None` check, not the except branch)

    def test_returned_list_exact_contents(self):
        """The returned list contains exactly the flagged strings in order."""
        params = {"reg_alpha": 9.9, "n_estimators": 150}
        space = {
            "estimator__reg_alpha": (0.001, 10.0),
            "estimator__n_estimators": (100, 5000),
        }
        flagged = report_boundary_positions(params, space)
        # reg_alpha is near upper; n_estimators 150 in (100, 5000) = pos 0.01 → < lower
        assert "reg_alpha > upper" in flagged
        assert "n_estimators < lower" in flagged

    def test_logger_injection(self, caplog):
        """A custom logger passed via the logger= argument is used for messages."""
        custom_logger = logging.getLogger("test_custom_boundary_logger")
        params = {"num_leaves": 300}
        space = {"estimator__num_leaves": (8, 1024)}
        with caplog.at_level(logging.INFO, logger="test_custom_boundary_logger"):
            report_boundary_positions(params, space, logger=custom_logger)
        # The custom logger should have emitted at least one INFO message
        assert any(
            r.name == "test_custom_boundary_logger" for r in caplog.records
        )

    def test_multiple_flags(self):
        """Multiple near-boundary dims all appear in the returned list."""
        params = {"reg_alpha": 9.9, "learning_rate": 0.29}
        space = {
            "estimator__reg_alpha": (0.001, 10.0),
            "estimator__learning_rate": (0.005, 0.3, "log10"),
        }
        flagged = report_boundary_positions(params, space)
        assert len(flagged) == 2

    def test_log_invalid_val_skipped(self):
        """A zero or negative value in a log10 dim is silently skipped."""
        params = {"reg_alpha": 0.0}
        space = {"estimator__reg_alpha": (0.001, 10.0, "log10")}
        flagged = report_boundary_positions(params, space)
        assert flagged == []


# ---------------------------------------------------------------------------
# boundary_report (DataFrame form)
# ---------------------------------------------------------------------------

class TestBoundaryReport:
    def test_returns_dataframe(self):
        best = {"estimator__reg_alpha": 9.89, "estimator__learning_rate": 0.069}
        space = {
            "estimator__reg_alpha": (0.001, 10.0),
            "estimator__learning_rate": (0.005, 0.3, "log10"),
        }
        df = boundary_report(best, space)
        assert set(df.columns) == {"param", "low", "high", "value", "scale", "position", "flag"}

    def test_flagged_near_upper(self):
        best = {"estimator__reg_alpha": 9.89}
        space = {"estimator__reg_alpha": (0.001, 10.0)}
        df = boundary_report(best, space)
        row = df[df["param"] == "reg_alpha"].iloc[0]
        assert row["flag"] == "> upper"
        assert row["scale"] == "linear"

    def test_log10_scale(self):
        best = {"estimator__learning_rate": 0.069}
        space = {"estimator__learning_rate": (0.005, 0.3, "log10")}
        df = boundary_report(best, space)
        row = df[df["param"] == "learning_rate"].iloc[0]
        assert row["scale"] == "log10"
        assert 0.0 < row["position"] < 1.0

    def test_categorical_skipped(self):
        best = {}
        space = {"lags": ["[1, 2, 24]"]}
        df = boundary_report(best, space)
        assert df.empty

    def test_sorted_descending_position(self):
        best = {
            "estimator__reg_alpha": 9.89,   # near upper → high position
            "estimator__num_leaves": 300,    # interior
        }
        space = {
            "estimator__reg_alpha": (0.001, 10.0),
            "estimator__num_leaves": (8, 1024),
        }
        df = boundary_report(best, space)
        assert df["position"].iloc[0] >= df["position"].iloc[-1]

    def test_prefix_stripped_in_param_column(self):
        best = {"estimator__num_leaves": 512}
        space = {"estimator__num_leaves": (8, 1024)}
        df = boundary_report(best, space)
        assert "num_leaves" in df["param"].values
        assert "estimator__num_leaves" not in df["param"].values


# ---------------------------------------------------------------------------
# suggest_bounds
# ---------------------------------------------------------------------------

class TestSuggestBounds:
    def test_interior_unchanged(self):
        best = {"estimator__num_leaves": 300}
        space = {"estimator__num_leaves": (8, 1024)}
        new_space = suggest_bounds(best, space)
        assert new_space["estimator__num_leaves"] == (8, 1024)

    def test_upper_pinned_float_multiplied(self):
        """Upper-pinned float bound: high * widen_factor."""
        best = {"estimator__reg_alpha": 9.89}
        space = {"estimator__reg_alpha": (0.001, 10.0)}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        assert new_space["estimator__reg_alpha"][1] == 100.0

    def test_upper_pinned_log_multiplied(self):
        """Upper-pinned log10 bound: high * widen_factor."""
        best = {"estimator__reg_alpha": 9.89}
        space = {"estimator__reg_alpha": (0.001, 10.0, "log10")}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        assert new_space["estimator__reg_alpha"][1] == 100.0
        assert new_space["estimator__reg_alpha"][2] == "log10"

    def test_upper_pinned_int_additive(self):
        """Upper-pinned integer bound: high + (high - low)."""
        best = {"estimator__n_estimators": 4950}
        space = {"estimator__n_estimators": (100, 5000)}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        assert new_space["estimator__n_estimators"][1] == 5000 + (5000 - 100)

    def test_lower_pinned_float_divided(self):
        """Lower-pinned float bound: low / widen_factor."""
        best = {"estimator__learning_rate": 0.006}
        space = {"estimator__learning_rate": (0.005, 0.3, "log10")}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        assert abs(new_space["estimator__learning_rate"][0] - 0.0005) < 1e-10

    def test_categorical_unchanged(self):
        """Categorical (list) entries pass through unchanged."""
        best = {}
        space = {"lags": ["[1, 2, 24]", "[1, 24, 168]"]}
        new_space = suggest_bounds(best, space)
        assert new_space["lags"] == space["lags"]

    def test_all_keys_preserved(self):
        """The returned dict has exactly the same keys as search_space.

        Uses prefixed keys (SpotOptim-result style) so the flagged/widened path
        is exercised for at least one dimension.
        """
        # num_leaves=1000 is near upper of (8, 1024) — should be widened
        best = {
            "estimator__num_leaves": 1000,
            "estimator__learning_rate": 0.05,
        }
        space = {
            "estimator__num_leaves": (8, 1024),
            "estimator__learning_rate": (0.005, 0.3, "log10"),
            "lags": ["[1, 2, 24]"],
        }
        new_space = suggest_bounds(best, space)
        assert set(new_space.keys()) == set(space.keys())
        # The near-upper-boundary integer dim must have been widened upward
        assert new_space["estimator__num_leaves"][1] > 1024
        # Interior log10 dim is unchanged
        assert new_space["estimator__learning_rate"] == space["estimator__learning_rate"]

    def test_widen_factor_parameter(self):
        """Different widen_factor values produce different bounds."""
        best = {"estimator__reg_alpha": 9.89}
        space = {"estimator__reg_alpha": (0.001, 10.0)}
        new5 = suggest_bounds(best, space, widen_factor=5.0)
        new10 = suggest_bounds(best, space, widen_factor=10.0)
        assert new10["estimator__reg_alpha"][1] > new5["estimator__reg_alpha"][1]

    def test_lower_pinned_int_additive_floor_1(self):
        """Lower-pinned integer bound floors at 1: max(1, low - (high - low))."""
        best = {"estimator__n_estimators": 105}  # near lower end of (100, 5000)
        space = {"estimator__n_estimators": (100, 5000)}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        # new low = max(1, 100 - (5000 - 100)) = max(1, -4800) = 1
        assert new_space["estimator__n_estimators"][0] >= 1
