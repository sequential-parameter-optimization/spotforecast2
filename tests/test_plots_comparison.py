# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2.plots.comparison.

Headless matplotlib backend must be set BEFORE pyplot is imported.  The
`matplotlib.use()` call at module level (before any pyplot import) satisfies
that requirement.
"""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from spotforecast2.plots.comparison import (
    plot_critical_difference,
    plot_rank_stability,
)

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close every matplotlib figure after each test to avoid resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_rank_stability
# ---------------------------------------------------------------------------


def _make_ranks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mae": [1, 2, 3, 4],
            "rmse": [1, 3, 2, 4],
            "mape": [2, 1, 3, 4],
        },
        index=["team_a", "team_b", "team_c", "baseline"],
    )


def test_rank_stability_returns_figure():
    fig = plot_rank_stability(_make_ranks())
    assert isinstance(fig, Figure)


def test_rank_stability_line_count_matches_entries():
    ranks = _make_ranks()
    fig = plot_rank_stability(ranks)
    ax = fig.axes[0]
    assert len(ax.lines) == len(ranks)


def test_rank_stability_highlight_color_applied():
    ranks = _make_ranks()
    fig = plot_rank_stability(ranks, highlight={"team_a": "#2a78d6"})
    ax = fig.axes[0]
    assert "#2a78d6" in [line.get_color() for line in ax.lines]


def test_rank_stability_labels_rename_annotation_text():
    ranks = _make_ranks()
    fig = plot_rank_stability(ranks, labels={"team_a": "Team A"})
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("Team A" in t for t in texts)
    assert not any("team_a  " in t or "  team_a" in t for t in texts)


def test_rank_stability_annotate_false_suppresses_texts():
    ranks = _make_ranks()
    fig = plot_rank_stability(ranks, annotate=False)
    ax = fig.axes[0]
    assert len(ax.texts) == 0


def test_rank_stability_xticks_are_column_names():
    ranks = _make_ranks()
    fig = plot_rank_stability(ranks)
    ax = fig.axes[0]
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == list(ranks.columns)


def test_rank_stability_yaxis_inverted_and_no_yticks():
    ranks = _make_ranks()
    fig = plot_rank_stability(ranks)
    ax = fig.axes[0]
    ymin, ymax = ax.get_ylim()
    assert ymin > ymax  # inverted
    assert len(ax.get_yticks()) == 0


def test_rank_stability_all_spines_hidden():
    ranks = _make_ranks()
    fig = plot_rank_stability(ranks)
    ax = fig.axes[0]
    assert all(not spine.get_visible() for spine in ax.spines.values())


def test_rank_stability_ax_injection_no_new_figure():
    _fig0, ax = plt.subplots()
    n_figs_before = len(plt.get_fignums())
    fig = plot_rank_stability(_make_ranks(), ax=ax)
    assert fig is ax.figure
    assert len(plt.get_fignums()) == n_figs_before


# ---------------------------------------------------------------------------
# plot_critical_difference
# ---------------------------------------------------------------------------


def _make_cd_panel():
    positions = pd.Series(
        {"team_a": 1.1, "team_b": 1.9, "team_c": 2.4, "baseline": 3.2}
    )
    entries = list(positions.index)
    sig_matrix = pd.DataFrame(1.0, index=entries, columns=entries)
    # one significant pair
    sig_matrix.loc["team_a", "baseline"] = 0.001
    sig_matrix.loc["baseline", "team_a"] = 0.001
    # one non-significant pair (already 1.0 by construction, keep explicit)
    sig_matrix.loc["team_b", "team_c"] = 0.8
    sig_matrix.loc["team_c", "team_b"] = 0.8
    return positions, sig_matrix


def test_critical_difference_returns_figure():
    positions, sig_matrix = _make_cd_panel()
    fig = plot_critical_difference(positions, sig_matrix)
    assert isinstance(fig, Figure)


def test_critical_difference_runs_on_four_entry_panel():
    positions, sig_matrix = _make_cd_panel()
    fig = plot_critical_difference(positions, sig_matrix, alpha=0.05)
    ax = fig.axes[0]
    # rank markers + crossbars/elbows produce at least one line per entry
    assert len(ax.lines) >= len(positions)


def test_critical_difference_labels_rename():
    positions, sig_matrix = _make_cd_panel()
    fig = plot_critical_difference(
        positions,
        sig_matrix,
        labels={"team_a": "Team A", "baseline": "Baseline"},
    )
    ax = fig.axes[0]
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "Team A" in texts
    assert "Baseline" in texts


def test_critical_difference_color_palette_renamed_with_labels():
    positions, sig_matrix = _make_cd_panel()
    # scikit_posthocs requires a color for every entry; the palette below
    # uses the ORIGINAL ids as keys and must be renamed to match the
    # relabelled `positions`/`sig_matrix` under the hood.
    palette = {entry_id: "#888888" for entry_id in positions.index}
    palette["baseline"] = "#008300"
    fig = plot_critical_difference(
        positions,
        sig_matrix,
        labels={"baseline": "Baseline"},
        color_palette=palette,
    )
    assert isinstance(fig, Figure)


def test_critical_difference_ax_injection_no_new_figure():
    positions, sig_matrix = _make_cd_panel()
    _fig0, ax = plt.subplots()
    n_figs_before = len(plt.get_fignums())
    fig = plot_critical_difference(positions, sig_matrix, ax=ax)
    assert fig is ax.figure
    assert len(plt.get_fignums()) == n_figs_before
