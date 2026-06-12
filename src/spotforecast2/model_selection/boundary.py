# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Search-space boundary management helpers for hyperparameter tuning.

After a SpotOptim (or any optimizer) run, the tuned optimum may press against
a search-space boundary — meaning the optimizer wanted to go further but was
constrained. These helpers make that visible and actionable.

Motivation: KB entry 2026-06-08-hyperparameter-boundary-management documents an
operational case where ``reg_alpha`` pinned at its old ceiling (98.9 % of the
linear range), inflating L1 regularization and flattening the live forecast.
Widening that bound and re-running resolved the issue. The helpers below
systematize that diagnostic loop.

Three functions are provided:

- `report_boundary_positions` — logs each dimension's position and returns a
  list of flagged names. Decoupled from ``MultiTask``: the caller extracts
  params (e.g. ``estimator.get_params()``). Primary operational diagnostic.
- `boundary_report` — returns a ``pd.DataFrame`` with one row per numeric
  dimension: position, scale, flag. Companion to `suggest_bounds`.
- `suggest_bounds` — returns a copy of the search space with flagged bounds
  widened on the pressed side; pass it straight back to
  ``run_task_spotoptim(search_space=...)``.

**Key convention difference — prefix handling:**

``report_boundary_positions`` strips the ``"estimator__"`` prefix from each
search-space key before looking up the value in ``params``.  The ``params``
dict is expected to come from ``estimator.get_params()`` (scikit-learn style),
which returns UN-prefixed keys such as ``"reg_alpha"`` — not
``"estimator__reg_alpha"``.

``boundary_report`` and ``suggest_bounds`` look up ``best_params`` using the
search-space key **as-is**, including any ``"estimator__"`` prefix.  The
``best_params`` dict is expected to come from a SpotOptim result, which stores
FULL search-space keys (e.g. ``"estimator__reg_alpha"``).

Mixing conventions (passing ``get_params()``-style keys to ``boundary_report``,
or SpotOptim result keys to ``report_boundary_positions``) will silently produce
an empty result because no key matches.  If ``boundary_report`` returns an empty
DataFrame for a space with numeric dimensions, a key-convention mismatch is the
most likely cause.

Ported from:

- ``report_boundary_positions``: ``bart26k-lecture/scripts/team4_4zones_submit.py``
  (``report_boundary_positions`` function, section ``@sec-team4-boundary-diagnostics``).
- ``boundary_report`` / ``suggest_bounds``: ``bart26k-lecture/14_team_4_submission.qmd``
  (cell ``team4-boundary-helpers``, section ``@sec-team4-boundary-management``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

_logger = logging.getLogger(__name__)


def _position_flag(pos: float, warn_frac: float) -> str:
    """Classify a normalized boundary position as near-upper, near-lower, or interior.

    Args:
        pos: Position of the tuned value inside its dimension, normalized to
            ``[0, 1]`` in the dimension's own scale.
        warn_frac: Fraction of the range defining the "near-boundary" zone at
            each end.

    Returns:
        ``"> upper"``, ``"< lower"``, or ``""`` (interior).
    """
    upper_zone = 1.0 - warn_frac
    if pos > upper_zone:
        return "> upper"
    if pos < warn_frac:
        return "< lower"
    return ""


def report_boundary_positions(
    params: Mapping[str, float | int],
    search_space: Mapping[str, Any],
    *,
    warn_frac: float = 0.10,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Log where each tuned value sits inside its numeric search-space interval.

    For each entry in `search_space` that is a 2- or 3-tuple numeric dimension
    ``(low, high)`` or ``(low, high, "log10")``, the function:

    - strips the ``"estimator__"`` prefix from the key when looking up the
      corresponding entry in `params`;
    - skips non-numeric or boolean values;
    - computes the position in the dimension's own scale (log10 for log dims,
      guarding against ``val <= 0`` or ``low <= 0``);
    - flags ``"> upper"`` when ``pos > 1 - warn_frac`` and ``"< lower"`` when
      ``pos < warn_frac``;
    - logs each dimension at INFO level in a columnar format;
    - returns the list of flagged strings (e.g. ``["reg_alpha > upper"]``).

    Categorical dimensions (list-valued entries) and unreadable entries are
    skipped. The function never raises — it is a diagnostic and returns an
    empty list on any unexpected error.

    Args:
        params: Flat dict of parameter names to numeric values, as returned by
            ``estimator.get_params()`` or equivalent. Keys should NOT carry the
            ``"estimator__"`` prefix (it is stripped from `search_space` keys,
            not from `params` keys).
        search_space: Dict mapping search-space keys (potentially with
            ``"estimator__"`` prefix) to dimension specs: ``(low, high)``,
            ``(low, high, "log10")``, or a list of categories.
        warn_frac: Fraction of the range (in the dimension's own scale) that
            defines the "near-boundary" zone at each end. Default is 0.10.
        logger: Logger to use for INFO/WARNING messages. Defaults to the
            module-level ``logging.getLogger(__name__)`` logger.

    Returns:
        List of flagged dimension strings, e.g. ``["reg_alpha > upper",
        "learning_rate < lower"]``. Empty if all dimensions are interior.

    Examples:
        Interior optimum — no flags returned:

        ```{python}
        from spotforecast2.model_selection.boundary import report_boundary_positions

        params = {"num_leaves": 300, "learning_rate": 0.05}
        space = {
            "estimator__num_leaves": (8, 1024),
            "estimator__learning_rate": (0.005, 0.3, "log10"),
        }
        flagged = report_boundary_positions(params, space)
        print("flagged:", flagged)
        assert flagged == []
        ```

        Near-upper-boundary — flag is returned:

        ```{python}
        from spotforecast2.model_selection.boundary import report_boundary_positions

        params = {"reg_alpha": 9.9}
        space = {"estimator__reg_alpha": (0.001, 10.0)}
        flagged = report_boundary_positions(params, space)
        print("flagged:", flagged)
        assert flagged == ["reg_alpha > upper"]
        ```

    """
    log = logger if logger is not None else _logger
    flagged: list[str] = []

    for name, spec in search_space.items():
        if not (isinstance(spec, tuple) and len(spec) in (2, 3)):
            continue  # categorical dim (e.g. "lags")
        key = name.split("estimator__", 1)[-1]
        try:
            val = params.get(key)  # type: ignore[arg-type]
            if val is None:
                continue
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            low, high = float(spec[0]), float(spec[1])
            is_log = len(spec) == 3 and spec[2] == "log10"
            if is_log and (val <= 0 or low <= 0):
                continue
            if is_log:
                pos = (math.log10(val) - math.log10(low)) / (
                    math.log10(high) - math.log10(low)
                )
            else:
                pos = (float(val) - low) / (high - low)
            flag = _position_flag(pos, warn_frac)
            log.info(
                "  bound %-18s = %-11.5g in [%g, %g]%s  pos=%.2f%s",
                key,
                val,
                low,
                high,
                " log10" if is_log else "",
                pos,
                f"  <-- {flag}" if flag else "",
            )
            if flag:
                flagged.append(f"{key} {flag}")
        except Exception as exc:  # noqa: BLE001
            log.warning("boundary check: skipping %r (unreadable entry: %s)", name, exc)

    if flagged:
        log.warning(
            "boundary check: %d tuned dim(s) near a bound (%s) -- consider "
            "widening that side and re-running.",
            len(flagged),
            ", ".join(flagged),
        )
    else:
        log.info("boundary check: all tuned dimensions sit interior.")

    return flagged


def boundary_report(
    best_params: Mapping[str, float | int],
    search_space: Mapping[str, Any],
    *,
    warn_frac: float = 0.10,
) -> pd.DataFrame:
    """Tabulate each tuned value's position inside its search-space bound.

    Returns a DataFrame sorted by descending position, with one row per
    numeric dimension. Categorical and boolean-valued dimensions are skipped.
    ``flag`` is one of ``"> upper"``, ``"< lower"``, or ``""`` (interior).

    This function uses the `search_space` keys as-is (including any
    ``"estimator__"`` prefix) to look up matching entries in `best_params`.
    The returned ``param`` column strips the ``"estimator__"`` prefix for
    readability.

    Ported from ``bart26k-lecture/14_team_4_submission.qmd``, cell
    ``team4-boundary-helpers``.

    Args:
        best_params: Flat dict of parameter names to values, keyed with the
            same names as `search_space` (including any ``"estimator__"``
            prefix).
        search_space: Dict mapping parameter names to dimension specs:
            ``(low, high)``, ``(low, high, "log10")``, or a list of
            categories (skipped).
        warn_frac: Fraction of the range (in the dimension's own scale)
            defining the "near-boundary" zone. Default is 0.10.

    Returns:
        DataFrame with columns ``param``, ``low``, ``high``, ``value``,
        ``scale``, ``position``, ``flag``, sorted by ``position`` descending.

    Examples:
        Report on a near-upper-boundary value:

        ```{python}
        from spotforecast2.model_selection.boundary import boundary_report

        best = {
            "estimator__reg_alpha": 9.89,
            "estimator__learning_rate": 0.069,
        }
        space = {
            "estimator__reg_alpha": (0.001, 10.0),
            "estimator__learning_rate": (0.005, 0.3, "log10"),
        }
        df = boundary_report(best, space)
        print(df.to_string(index=False))
        assert "reg_alpha" in df["param"].values
        flagged = df[df["flag"] == "> upper"]["param"].tolist()
        assert "reg_alpha" in flagged
        ```

    """
    rows = []
    for name, spec in search_space.items():
        if not (isinstance(spec, tuple) and len(spec) in (2, 3)):
            continue  # categorical dim
        low, high = float(spec[0]), float(spec[1])
        is_log = len(spec) == 3 and spec[2] == "log10"
        val = best_params.get(name)  # type: ignore[arg-type]
        if val is None:
            continue
        if isinstance(val, bool):
            continue  # bool is a subclass of int; skip to avoid false positions
        if is_log:
            if val <= 0 or low <= 0:
                continue
            pos = (math.log10(val) - math.log10(low)) / (
                math.log10(high) - math.log10(low)
            )
        else:
            pos = (float(val) - low) / (high - low)
        flag = _position_flag(pos, warn_frac)
        rows.append(
            {
                "param": name.replace("estimator__", ""),
                "low": low,
                "high": high,
                "value": val,
                "scale": "log10" if is_log else "linear",
                "position": round(pos, 3),
                "flag": flag,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        numeric_dims = sum(
            1
            for spec in search_space.values()
            if isinstance(spec, tuple) and len(spec) in (2, 3)
        )
        if numeric_dims > 0:
            _logger.warning(
                "boundary_report: result is empty but search_space has %d numeric "
                "dimension(s). Check key-convention: best_params must use the FULL "
                "search-space keys (e.g. 'estimator__reg_alpha'), not the unprefixed "
                "get_params() style ('reg_alpha'). "
                "Use report_boundary_positions() instead if you have unprefixed keys.",
                numeric_dims,
            )
        return pd.DataFrame(
            columns=["param", "low", "high", "value", "scale", "position", "flag"]
        )
    return df.sort_values("position", ascending=False).reset_index(drop=True)


def suggest_bounds(
    best_params: Mapping[str, float | int],
    search_space: Mapping[str, Any],
    *,
    warn_frac: float = 0.10,
    widen_factor: float = 10.0,
) -> dict[str, Any]:
    """Return a copy of `search_space` with flagged bounds widened.

    Upper-pinned dimensions grow upward (``high * widen_factor`` for float/log,
    ``high + (high - low)`` for integer); lower-pinned dimensions grow downward
    (``low / widen_factor`` for float/log, ``max(1, low - (high - low))`` for
    integer). Interior and categorical dimensions are copied unchanged.

    Pass the result straight back to
    ``run_task_spotoptim(search_space=suggest_bounds(...))``.

    Ported from ``bart26k-lecture/14_team_4_submission.qmd``, cell
    ``team4-boundary-helpers`` (parameter ``widen`` renamed to ``widen_factor``
    for clarity).

    Args:
        best_params: Flat dict of parameter names to values, keyed with the
            same names as `search_space` (including any ``"estimator__"``
            prefix).
        search_space: Dict mapping parameter names to dimension specs:
            ``(low, high)``, ``(low, high, "log10")``, or a list of
            categories (returned unchanged).
        warn_frac: Fraction of the range (in the dimension's own scale)
            defining the "near-boundary" zone. Default is 0.10.
        widen_factor: Multiplicative factor for widening float/log bounds.
            Integer bounds use an additive span instead. Default is 10.0.

    Returns:
        New search-space dict with the same keys as `search_space` but with
        boundary-pressed bounds extended on the pressed side.

    Examples:
        Upper-pinned float bound is multiplied by widen_factor:

        ```{python}
        from spotforecast2.model_selection.boundary import suggest_bounds

        best = {"estimator__reg_alpha": 9.89}
        space = {"estimator__reg_alpha": (0.001, 10.0)}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        print(new_space)
        assert new_space["estimator__reg_alpha"][1] == 100.0
        ```

        Log-scale upper-pinned bound is also multiplied:

        ```{python}
        from spotforecast2.model_selection.boundary import suggest_bounds

        best = {"estimator__reg_alpha": 9.89}
        space = {"estimator__reg_alpha": (0.001, 10.0, "log10")}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        print(new_space)
        assert new_space["estimator__reg_alpha"][1] == 100.0
        ```

        Integer bound grows additively:

        ```{python}
        from spotforecast2.model_selection.boundary import suggest_bounds

        best = {"estimator__n_estimators": 4950}
        space = {"estimator__n_estimators": (100, 5000)}
        new_space = suggest_bounds(best, space, widen_factor=10.0)
        print(new_space)
        assert new_space["estimator__n_estimators"][1] == 5000 + (5000 - 100)
        ```

        Interior dimension is returned unchanged:

        ```{python}
        from spotforecast2.model_selection.boundary import suggest_bounds

        best = {"estimator__num_leaves": 300}
        space = {"estimator__num_leaves": (8, 1024)}
        new_space = suggest_bounds(best, space)
        assert new_space["estimator__num_leaves"] == (8, 1024)
        ```

    """
    report = boundary_report(best_params, search_space, warn_frac=warn_frac)
    flagged = {row["param"]: row["flag"] for _, row in report.iterrows() if row["flag"]}
    out: dict[str, Any] = {}
    for name, spec in search_space.items():
        if not (isinstance(spec, tuple) and len(spec) in (2, 3)):
            out[name] = spec
            continue
        low, high, rest = spec[0], spec[1], spec[2:]
        is_int = isinstance(low, int) and isinstance(high, int) and not rest
        short_name = name.replace("estimator__", "")
        flag = flagged.get(short_name, "")
        if flag == "> upper":
            high = int(high + (high - low)) if is_int else high * widen_factor
        elif flag == "< lower":
            low = max(1, int(low - (high - low))) if is_int else low / widen_factor
        out[name] = (low, high, *rest)
    return out
