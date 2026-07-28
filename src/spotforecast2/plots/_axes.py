# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared axes-injection helper for the plots modules."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def ensure_axes(
    ax: Axes | None, figsize: tuple[float, float]
) -> tuple[Figure, Axes, bool]:
    """Return ``(fig, ax, owns_figure)``, creating a figure only when needed.

    When ``ax`` is None a new figure of ``figsize`` is created and
    ``owns_figure`` is True — the calling plot function may then apply
    figure-level layout (``tight_layout`` etc.). When ``ax`` is given, the
    caller owns the figure: no new figure is created and ``owns_figure``
    is False.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False
