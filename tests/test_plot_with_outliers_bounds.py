# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the updated plot_with_outliers() function with bounds support.

Verifies that horizontal bound lines are correctly added to the Plotly figure
when config.bounds is provided, and that the function degrades gracefully when
bounds are absent.
"""

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from spotforecast2.plots.plotter import plot_with_outliers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n: int = 100, targets: list[str] | None = None):
    """Return (df_pipeline, df_original, dates) with synthetic data."""
    if targets is None:
        targets = ["target1"]
    dates = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    data = {t: np.random.default_rng(42).uniform(10, 90, n) for t in targets}
    df_orig = pd.DataFrame(data, index=dates)
    df_pipe = df_orig.copy()
    # Mark row 10 of every target as NaN to simulate outlier removal
    for t in targets:
        df_pipe.loc[dates[10], t] = np.nan
    return df_pipe, df_orig, dates


def _captured_figures(
    df_pipeline: pd.DataFrame,
    df_original: pd.DataFrame,
    config,
    targets: list[str] | None = None,
) -> list[go.Figure]:
    """Run plot_with_outliers and collect figures without displaying them."""
    figures = []

    def fake_show(self, *args, **kwargs):
        figures.append(self)

    with patch.object(go.Figure, "show", fake_show):
        plot_with_outliers(df_pipeline, df_original, config, targets=targets)

    return figures


# ---------------------------------------------------------------------------
# Tests: import and basic call
# ---------------------------------------------------------------------------


class TestPlotWithOutliersImport:
    """Verify the function is importable from the plots package."""

    def test_importable_from_plotter(self):
        from spotforecast2.plots.plotter import plot_with_outliers as fn  # noqa: F401

        assert callable(fn)


# ---------------------------------------------------------------------------
# Tests: without bounds
# ---------------------------------------------------------------------------


class TestPlotWithOutliersNoBounds:
    """Function must work correctly when config.bounds is None or missing."""

    def test_no_bounds_returns_one_figure_per_target(self):
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config)
        assert len(figs) == 1

    def test_no_bounds_two_targets_two_figures(self):
        df_pipe, df_orig, _ = _make_data(targets=["A", "B"])
        config = SimpleNamespace(targets=["A", "B"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config)
        assert len(figs) == 2

    def test_no_bounds_attribute_does_not_raise(self):
        """Config without a bounds attribute should not raise AttributeError."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"])  # no bounds attr
        figs = _captured_figures(df_pipe, df_orig, config)
        assert len(figs) == 1

    def test_no_bounds_no_hlines_in_layout(self):
        """Without bounds there should be no shapes in the layout."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config)
        layout = figs[0].layout
        shapes = layout.shapes if layout.shapes else []
        assert len(shapes) == 0

    def test_outlier_trace_present(self):
        """Red outlier marker trace must be included when outliers exist."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config)
        names = [t.name for t in figs[0].data]
        assert "Outliers" in names

    def test_regular_data_trace_present(self):
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config)
        names = [t.name for t in figs[0].data]
        assert "Regular Data" in names


# ---------------------------------------------------------------------------
# Tests: with bounds
# ---------------------------------------------------------------------------


class TestPlotWithOutliersBounds:
    """Horizontal bound lines must be added when config.bounds is provided."""

    def test_bounds_adds_two_shapes_per_figure(self):
        """Each figure should have exactly 2 shapes (lower + upper hlines)."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=[(0.0, 100.0)])
        figs = _captured_figures(df_pipe, df_orig, config)
        shapes = figs[0].layout.shapes
        assert len(shapes) == 2

    def test_bounds_shape_y_values_match_config(self):
        """Shapes must be placed at exactly the configured y-values."""
        lower, upper = -50.0, 250.0
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=[(lower, upper)])
        figs = _captured_figures(df_pipe, df_orig, config)
        y_vals = {s.y0 for s in figs[0].layout.shapes}
        assert lower in y_vals
        assert upper in y_vals

    def test_bounds_shapes_are_horizontal_lines(self):
        """Both shapes must have type='line' and span the full x range."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=[(0.0, 100.0)])
        figs = _captured_figures(df_pipe, df_orig, config)
        for shape in figs[0].layout.shapes:
            assert shape.type == "line"

    def test_bounds_shape_color_is_lightblue(self):
        """Bound lines must use lightblue color."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=[(0.0, 100.0)])
        figs = _captured_figures(df_pipe, df_orig, config)
        for shape in figs[0].layout.shapes:
            assert shape.line.color == "lightblue"

    def test_bounds_two_targets_correct_shapes_each(self):
        """Each of two figures gets its own pair of bound lines."""
        df_pipe, df_orig, _ = _make_data(targets=["A", "B"])
        config = SimpleNamespace(
            targets=["A", "B"],
            bounds=[(0.0, 100.0), (-10.0, 60.0)],
        )
        figs = _captured_figures(df_pipe, df_orig, config)
        assert len(figs) == 2
        # First figure: bounds (0, 100)
        y_vals_0 = {s.y0 for s in figs[0].layout.shapes}
        assert 0.0 in y_vals_0 and 100.0 in y_vals_0
        # Second figure: bounds (-10, 60)
        y_vals_1 = {s.y0 for s in figs[1].layout.shapes}
        assert -10.0 in y_vals_1 and 60.0 in y_vals_1

    def test_bounds_partial_list_no_shapes_for_missing_entry(self):
        """If bounds list is shorter than targets, extra targets get no shapes."""
        df_pipe, df_orig, _ = _make_data(targets=["A", "B"])
        config = SimpleNamespace(
            targets=["A", "B"],
            bounds=[(0.0, 100.0)],  # only one entry for two targets
        )
        figs = _captured_figures(df_pipe, df_orig, config)
        # First figure has shapes
        assert len(figs[0].layout.shapes) == 2
        # Second figure has no shapes
        assert len(figs[1].layout.shapes) == 0

    def test_bounds_annotations_contain_bound_values(self):
        """Layout annotations should mention the bound values."""
        lower, upper = 5.0, 95.0
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=[(lower, upper)])
        figs = _captured_figures(df_pipe, df_orig, config)
        texts = [a.text for a in figs[0].layout.annotations if a.text]
        assert any(str(lower) in t for t in texts)
        assert any(str(upper) in t for t in texts)


# ---------------------------------------------------------------------------
# Tests: title includes outlier percentage
# ---------------------------------------------------------------------------


class TestPlotWithOutliersTitle:
    """Plot title must always include the outlier percentage."""

    def test_title_includes_target_name(self):
        df_pipe, df_orig, _ = _make_data(targets=["my_target"])
        config = SimpleNamespace(targets=["my_target"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config)
        assert "my_target" in figs[0].layout.title.text

    def test_title_includes_percentage(self):
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(targets=["load"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config)
        # Title must contain a percentage value (e.g. "1.00%")
        assert "%" in figs[0].layout.title.text


# ---------------------------------------------------------------------------
# Tests: no outliers case
# ---------------------------------------------------------------------------


class TestPlotWithOutliersNoOutliers:
    """When there are no NaN rows, no Outliers trace should be added."""

    def test_no_outliers_no_outlier_trace(self):
        dates = pd.date_range("2023-01-01", periods=50, freq="h", tz="UTC")
        data = pd.DataFrame({"load": np.ones(50)}, index=dates)
        config = SimpleNamespace(targets=["load"], bounds=None)
        figs = _captured_figures(data, data, config)
        names = [t.name for t in figs[0].data]
        assert "Outliers" not in names


# ---------------------------------------------------------------------------
# Tests: explicit targets argument (primary pipeline path)
# ---------------------------------------------------------------------------


class TestPlotWithOutliersExplicitTargets:
    """The explicit ``targets=`` argument is the primary (run_state) path.

    ``PlottingMixin`` passes ``task.run_state.targets`` explicitly; the
    ``config.targets`` fallback exists only for legacy standalone callers.
    """

    def test_explicit_targets_used_when_config_lacks_targets(self):
        """A config without a targets attribute must work with explicit targets."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(bounds=None)  # no targets attribute at all
        figs = _captured_figures(df_pipe, df_orig, config, targets=["load"])
        assert len(figs) == 1
        assert "load" in figs[0].layout.title.text

    def test_explicit_targets_take_precedence_over_config(self):
        """When both are present, the explicit argument wins."""
        df_pipe, df_orig, _ = _make_data(targets=["A", "B"])
        config = SimpleNamespace(targets=["A", "B"], bounds=None)
        figs = _captured_figures(df_pipe, df_orig, config, targets=["A"])
        assert len(figs) == 1  # only the explicit list is plotted
        assert "A" in figs[0].layout.title.text

    def test_explicit_targets_with_bounds(self):
        """Bounds are applied positionally against the explicit target list."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(bounds=[(0.0, 100.0)])
        figs = _captured_figures(df_pipe, df_orig, config, targets=["load"])
        assert len(figs[0].layout.shapes) == 2
        y_vals = {s.y0 for s in figs[0].layout.shapes}
        assert 0.0 in y_vals and 100.0 in y_vals


# ---------------------------------------------------------------------------
# Tests: config.targets fallback (supported standalone/notebook path)
# ---------------------------------------------------------------------------


class TestPlotWithOutliersConfigTargetsFallback:
    """Pin config.targets fallback as a supported standalone/notebook API.

    WHY THIS IS PINNED:
    The consumer-contract artifact ``bart26k-lecture/14_team_4_submission.qmd``
    (line 1888) calls ``plot_with_outliers`` WITHOUT an explicit ``targets``
    kwarg, relying on ``config.targets`` being used as the fallback.  This
    usage pattern is load-bearing and must not be removed.  The previously
    planned removal in 6.0.0 is cancelled.

    Reference: ADR adr-multitask-configmulti-merge follow-up reassessment
    2026-06-07.
    """

    def test_no_targets_kwarg_uses_config_targets(self):
        """Calling without targets kwarg produces figures for config.targets."""
        df_pipe, df_orig, _ = _make_data(targets=["alpha", "beta"])
        config = SimpleNamespace(bounds=None, targets=["alpha", "beta"])
        # Call WITHOUT explicit targets kwarg — passes targets=None through to plot_with_outliers, triggering the config.targets fallback inside the function
        figs = _captured_figures(df_pipe, df_orig, config)
        assert len(figs) == 2
        titles = [figs[i].layout.title.text for i in range(2)]
        assert any("alpha" in t for t in titles)
        assert any("beta" in t for t in titles)

    def test_no_targets_kwarg_emits_no_deprecation_warning(self):
        """The config.targets fallback must not emit DeprecationWarning or FutureWarning."""
        df_pipe, df_orig, _ = _make_data(targets=["load"])
        config = SimpleNamespace(bounds=None, targets=["load"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _captured_figures(df_pipe, df_orig, config)
        deprecation_warnings = [
            w
            for w in caught
            if issubclass(w.category, (DeprecationWarning, FutureWarning))
        ]
        assert (
            deprecation_warnings == []
        ), f"Unexpected deprecation warnings: {[str(w.message) for w in deprecation_warnings]}"

    def test_explicit_targets_wins_over_config_targets(self):
        """When explicit targets= differs from config.targets, explicit wins."""
        df_pipe, df_orig, _ = _make_data(targets=["series_a", "series_b"])
        config = SimpleNamespace(bounds=None, targets=["series_a", "series_b"])
        # Pass only ["series_a"] explicitly — only one figure should be produced
        figs = _captured_figures(df_pipe, df_orig, config, targets=["series_a"])
        assert len(figs) == 1
        title = figs[0].layout.title.text
        assert "series_a" in title
        assert "series_b" not in title
