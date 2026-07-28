# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cross-entry comparison plots for forecasting campaigns / leaderboards.

Generalises the rank-stability bump chart and the critical-difference
diagram of a manuscript's results section into a reusable, stateless API.
Both functions return a `matplotlib.figure.Figure`; the caller is
responsible for saving and closing it. Neither function calls `plt.show()`
nor mutates `matplotlib.rcParams` — styling and figure lifecycle stay with
the caller (set `matplotlib.use("Agg")` before importing `pyplot` in
headless environments).
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spotforecast2.plots._axes import ensure_axes

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def plot_rank_stability(
    ranks: pd.DataFrame,
    *,
    labels: Mapping[str, str] | None = None,
    highlight: Mapping[str, str] | None = None,
    base_color: str = "0.7",
    highlight_linewidth: float = 2.0,
    base_linewidth: float = 1.0,
    marker_size: float = 2.0,
    highlight_marker_size: float = 3.0,
    label_fontsize: float = 7.5,
    label_color: str | None = None,
    muted_label_color: str = "0.35",
    annotate: bool = True,
    tick_labelsize: float | None = None,
    tick_labelcolor: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.3, 5.2),
) -> Figure:
    """Bump chart of entry ranks across a set of metrics.

    `ranks` has one row per entry (the index holds the entry id) and one
    column per metric, in the order the metrics should appear on the x
    axis. Cell values are the entry's integer rank under that metric. A
    line connects each entry's rank across the metrics; entries listed in
    `highlight` are drawn with a thicker line, a larger marker, and their
    given color, while every other entry shares `base_color`.

    Args:
        ranks: DataFrame indexed by entry id, one integer-rank column per
            metric, column order defines the x order.
        labels: Mapping of entry id to display name used in the
            `annotate=True` text labels. Entries missing from the mapping
            fall back to their raw id.
        highlight: Mapping of entry id to the color used for that entry's
            line, marker, and (in full ink) its labels. Entries not present
            here are drawn in `base_color`.
        base_color: Line/marker color for entries not in `highlight`.
        highlight_linewidth: Line width for highlighted entries.
        base_linewidth: Line width for non-highlighted entries.
        marker_size: Marker size for non-highlighted entries.
        highlight_marker_size: Marker size for highlighted entries.
        label_fontsize: Font size of the `annotate=True` text labels.
        label_color: Text color of the labels of highlighted entries.
            Defaults to None, which uses the ambient
            ``matplotlib.rcParams["text.color"]``.
        muted_label_color: Text color of the labels of non-highlighted
            entries.
        annotate: When `True`, write `"label rank"` to the left of the
            first column and `"rank label"` to the right of the last
            column for every entry. The labels are drawn with
            `clip_on=False`, so they render outside the axes box — widen
            the figure margins (e.g. via ``fig.subplots_adjust``) to make
            room for them.
        tick_labelsize: Font size of the x tick labels. Defaults to None
            (leave the ambient size).
        tick_labelcolor: Color of the x tick labels. Defaults to None
            (leave the ambient color).
        ax: Existing axes to draw into. When given, no new figure is
            created and the function returns `ax.figure`.
        figsize: Figure size used when `ax` is not given.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2.plots.comparison import plot_rank_stability

        ranks = pd.DataFrame(
            {
                "mae": [1, 2, 3, 4],
                "rmse": [1, 3, 2, 4],
                "mape": [2, 1, 3, 4],
            },
            index=["team_a", "team_b", "team_c", "baseline"],
        )
        fig = plot_rank_stability(
            ranks,
            labels={"team_a": "Team A", "baseline": "Baseline"},
            highlight={"team_a": "#2a78d6", "baseline": "#008300"},
        )
        print(type(fig).__name__)
        ```
    """
    labels = labels or {}
    highlight = highlight or {}
    if label_color is None:
        label_color = plt.rcParams["text.color"]

    fig, ax, _ = ensure_axes(ax, figsize)

    metrics = list(ranks.columns)
    n_metrics = len(metrics)
    x = list(range(n_metrics))

    for entry_id, row in ranks.iterrows():
        y = row.to_numpy()
        is_highlighted = entry_id in highlight
        color = highlight[entry_id] if is_highlighted else base_color
        lw = highlight_linewidth if is_highlighted else base_linewidth
        ms = highlight_marker_size if is_highlighted else marker_size
        ax.plot(
            x,
            y,
            color=color,
            lw=lw,
            marker="o",
            ms=ms,
            zorder=3 if is_highlighted else 2,
        )
        if annotate:
            label = labels.get(entry_id, entry_id)
            text_color = label_color if is_highlighted else muted_label_color
            ax.text(
                -0.12,
                y[0],
                f"{label}  {y[0]}",
                ha="right",
                va="center",
                fontsize=label_fontsize,
                color=text_color,
                clip_on=False,
            )
            ax.text(
                n_metrics - 1 + 0.12,
                y[-1],
                f"{y[-1]}  {label}",
                ha="left",
                va="center",
                fontsize=label_fontsize,
                color=text_color,
                clip_on=False,
            )

    ax.set_xlim(-0.15, n_metrics - 1 + 0.15)
    ax.set_xticks(x, metrics)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    tick_kwargs: dict[str, object] = {"length": 0}
    if tick_labelsize is not None:
        tick_kwargs["labelsize"] = tick_labelsize
    if tick_labelcolor is not None:
        tick_kwargs["labelcolor"] = tick_labelcolor
    ax.tick_params(**tick_kwargs)

    return fig


def plot_critical_difference(
    positions: pd.Series,
    sig_matrix: pd.DataFrame,
    *,
    labels: Mapping[str, str] | None = None,
    color_palette: Mapping[str, str] | None = None,
    alpha: float = 0.05,
    label_fmt_left: str = "{label} ({rank:.0f})",
    label_fmt_right: str = "({rank:.0f}) {label}",
    label_props: Mapping[str, object] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.3, 2.8),
) -> Figure:
    """Critical-difference diagram (Demsar layout) over paired entries.

    Thin wrapper over `scikit_posthocs.critical_difference_diagram`.
    `scikit_posthocs` pulls in `seaborn` at import time, so it is imported
    lazily inside this function rather than at module scope (the same
    precedent as `plots.diagnostics.plot_shap_summary` for `shap`).

    Args:
        positions: Lower-is-better summary statistic (e.g. mean MAE) per
            entry, indexed by entry id.
        sig_matrix: Square, symmetric p-value frame indexed and columned
            by the same entry ids as `positions`.
        labels: Mapping of entry id to display name. When given, both
            `positions` and `sig_matrix` are renamed before plotting, and
            `color_palette` keys (which refer to the original ids) are
            renamed to match.
        color_palette: Mapping of entry id (original id, renamed
            automatically when `labels` is given) to a color for that
            entry's marker and label. `scikit_posthocs` requires a color
            for every entry in `positions` once a palette is given.
        alpha: Significance level used to draw crossbars between entries
            that are not significantly different.
        label_fmt_left: Format string for labels left of the first rank
            marker; `{label}` and `{rank}` are available.
        label_fmt_right: Format string for labels right of the last rank
            marker.
        label_props: Extra keyword arguments forwarded to the label text
            objects (e.g. `{"fontsize": 8}`).
        ax: Existing axes to draw into. When given, no new figure is
            created and the function returns `ax.figure`.
        figsize: Figure size used when `ax` is not given.

    Returns:
        A `matplotlib.figure.Figure`.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2.plots.comparison import plot_critical_difference

        positions = pd.Series(
            {"team_a": 1.2, "team_b": 1.8, "team_c": 2.1, "baseline": 3.0}
        )
        entries = list(positions.index)
        sig_matrix = pd.DataFrame(1.0, index=entries, columns=entries)
        sig_matrix.loc["team_a", "baseline"] = 0.01
        sig_matrix.loc["baseline", "team_a"] = 0.01

        # scikit-posthocs requires a color for every entry when a palette is
        # given.
        palette = {entry_id: "#888888" for entry_id in entries}
        palette["baseline"] = "#008300"
        fig = plot_critical_difference(
            positions,
            sig_matrix,
            labels={"team_a": "Team A", "baseline": "Baseline"},
            color_palette=palette,
        )
        print(type(fig).__name__)
        ```
    """
    import scikit_posthocs as sp

    if color_palette is not None:
        missing = [e for e in positions.index if e not in color_palette]
        if missing:
            raise ValueError(
                "color_palette must cover every entry in positions; "
                f"missing: {missing}."
            )

    fig, ax, _ = ensure_axes(ax, figsize)

    if labels:
        positions = positions.rename(index=labels)
        sig_matrix = sig_matrix.rename(index=labels, columns=labels)
        if color_palette:
            color_palette = {
                labels.get(entry_id, entry_id): color
                for entry_id, color in color_palette.items()
            }

    sp.critical_difference_diagram(
        positions,
        sig_matrix,
        ax=ax,
        alpha=alpha,
        label_fmt_left=label_fmt_left,
        label_fmt_right=label_fmt_right,
        label_props=dict(label_props) if label_props is not None else None,
        color_palette=dict(color_palette) if color_palette is not None else None,
    )

    return fig
