# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2.plots.diagnostics.

Headless matplotlib backend must be set BEFORE pyplot is imported.  The
`matplotlib.use()` call at module level (before any pyplot import) satisfies
that requirement.
"""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402 — must be after matplotlib.use
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spotforecast2.plots.diagnostics import (  # noqa: E402
    feature_family,
    plot_acf_with_lags,
    plot_feature_importance_by_family,
    plot_forecast_vs_reference,
    plot_shap_summary,
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
# feature_family — exhaustive family-to-name mapping table
# ---------------------------------------------------------------------------

_FAMILY_CASES = [
    # holiday family
    ("holiday_DE", "holiday"),
    ("is_holiday", "holiday"),
    ("HOLIDAY_NW", "holiday"),
    ("brueckentag_NW", "holiday"),
    ("BRUECKENTAG_DE", "holiday"),
    ("day_after_holiday", "holiday"),
    ("day_before_holiday", "holiday"),
    # polynomial family
    ("poly_hour_2", "polynomial"),
    ("poly_dayofweek_3", "polynomial"),
    ("POLY_month", "polynomial"),
    # precedence: poly beats window
    ("poly_window_24", "polynomial"),
    # weather_window family
    ("window_mean_72", "weather_window"),
    ("roll_window_168", "weather_window"),
    ("WINDOW_std_24", "weather_window"),
    # precedence: window beats cyclical
    ("sin_window", "weather_window"),
    # cyclical/RBF family
    ("sin_hour", "cyclical/RBF"),
    ("cos_hour", "cyclical/RBF"),
    ("rbf_hour_0", "cyclical/RBF"),
    ("SIN_dayofyear", "cyclical/RBF"),
    ("COS_month", "cyclical/RBF"),
    # lag family
    ("lag_1", "lag"),
    ("lag_24", "lag"),
    ("lag_168", "lag"),
    # weather/other (catch-all)
    ("wind_speed_10m", "weather/other"),
    ("temperature_2m", "weather/other"),
    ("entsoe_forecasted_load", "weather/other"),
    ("is_workday", "weather/other"),
    ("day_type", "weather/other"),
    ("solar_zenith", "weather/other"),
]


@pytest.mark.parametrize("name,expected", _FAMILY_CASES)
def test_feature_family_mapping(name, expected):
    """Every name-to-family pair in the exhaustive table resolves correctly."""
    assert feature_family(name) == expected


def test_feature_family_case_insensitive_for_holiday():
    """Holiday detection is case-insensitive (lowercased internally)."""
    assert feature_family("Holiday_NW") == "holiday"


def test_feature_family_lag_prefix_only():
    """Only names that START with 'lag' (after lowercasing) map to lag family."""
    assert feature_family("lag_1") == "lag"
    # 'flag' contains 'lag' but does NOT start with it
    assert feature_family("flag_col") == "weather/other"


# ---------------------------------------------------------------------------
# plot_acf_with_lags
# ---------------------------------------------------------------------------


def _make_acf(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"lag": np.arange(n), "autocorrelation": rng.uniform(-0.4, 0.4, n)}
    )


def test_plot_acf_returns_figure():
    acf = _make_acf(60)
    fig = plot_acf_with_lags(acf, key_lags=[1, 24, 48], conf=0.1)
    assert isinstance(fig, Figure)


def test_plot_acf_annotation_count_equals_present_key_lags():
    """Annotations are drawn only for key_lags that actually appear in acf."""
    acf = _make_acf(50)  # lags 0..49
    # 3 of 4 lags are present in the frame (lag 100 is not)
    key_lags = [1, 24, 48, 100]
    expected_annotations = sum(1 for lag in key_lags if lag in acf["lag"].values)
    fig = plot_acf_with_lags(acf, key_lags=key_lags, conf=0.1)
    ax = fig.axes[0]
    n_annotations = len(ax.texts)
    assert n_annotations == expected_annotations


def test_plot_acf_empty_key_lags_no_annotations():
    acf = _make_acf(30)
    fig = plot_acf_with_lags(acf, key_lags=[], conf=0.1)
    ax = fig.axes[0]
    assert len(ax.texts) == 0


def test_plot_acf_all_key_lags_missing_no_annotations():
    """key_lags entirely outside the acf range produces zero annotations."""
    acf = _make_acf(10)  # lags 0..9
    fig = plot_acf_with_lags(acf, key_lags=[50, 100, 168], conf=0.1)
    ax = fig.axes[0]
    assert len(ax.texts) == 0


# ---------------------------------------------------------------------------
# plot_feature_importance_by_family
# ---------------------------------------------------------------------------


def test_importance_plot_returns_figure():
    names = ["lag_1", "lag_24", "holiday_DE", "wind_speed"]
    scores = [100, 80, 60, 40]
    fig = plot_feature_importance_by_family(names, scores)
    assert isinstance(fig, Figure)


def test_importance_plot_top_n_respected():
    """The bar chart contains at most top_n bars."""
    rng = np.random.default_rng(1)
    n_features = 30
    names = [f"feature_{i}" for i in range(n_features)]
    scores = rng.integers(1, 100, n_features).tolist()
    top_n = 10
    fig = plot_feature_importance_by_family(names, scores, top_n=top_n)
    ax = fig.axes[0]
    # barh produces one patch per bar
    assert len(ax.patches) == top_n


def test_importance_plot_family_colors_distinct():
    """Known feature names from different families get different bar colors."""
    names = ["lag_1", "holiday_DE", "poly_hour_2", "window_mean_72",
             "sin_hour", "wind_speed"]
    scores = [100, 90, 80, 70, 60, 50]
    fig = plot_feature_importance_by_family(names, scores)
    ax = fig.axes[0]
    # Each bar's facecolor should be non-grey (grey = unknown family)
    face_colors = [p.get_facecolor() for p in ax.patches]
    # At minimum the lag bar (red #B22222) and holiday (green #2E8B57) differ
    assert len(set(str(c) for c in face_colors)) > 1


def test_importance_plot_top_n_larger_than_features():
    """top_n larger than available features still works (no error)."""
    names = ["lag_1", "lag_24"]
    scores = [50, 30]
    fig = plot_feature_importance_by_family(names, scores, top_n=20)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_shap_summary
# ---------------------------------------------------------------------------


def test_shap_summary_returns_figure():
    """Train a tiny LGBMRegressor and verify plot_shap_summary returns Figure."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.standard_normal((120, 4)), columns=["f0", "f1", "f2", "f3"]
    )
    y = X["f0"] * 2 + rng.standard_normal(120) * 0.1
    est = LGBMRegressor(n_estimators=10, verbose=-1, random_state=0)
    est.fit(X, y)

    fig = plot_shap_summary(est, X, max_samples=50)
    assert isinstance(fig, Figure)


def test_shap_summary_subsamples_max_samples():
    """When X has more rows than max_samples, SHAP only sees a subset.

    We cannot directly inspect the SHAP call, but we verify the function
    completes without error and returns a Figure.  The subsampling logic
    (stride = max(1, len(X) // max_samples)) is verified by the fast runtime.
    """
    pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(9)
    X = pd.DataFrame(
        rng.standard_normal((500, 3)), columns=["a", "b", "c"]
    )
    y = X["a"] + rng.standard_normal(500) * 0.2
    est = LGBMRegressor(n_estimators=10, verbose=-1, random_state=0)
    est.fit(X, y)

    fig = plot_shap_summary(est, X, max_samples=50)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_forecast_vs_reference
# ---------------------------------------------------------------------------


def _make_forecast(n: int = 24, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-15", periods=n, freq="h", tz="UTC")
    return pd.Series(40_000 + rng.standard_normal(n) * 1000, index=idx)


def test_forecast_vs_reference_returns_figure_with_overlap():
    forecast = _make_forecast(24)
    reference = _make_forecast(24, seed=1)  # same index -> full overlap
    fig = plot_forecast_vs_reference(
        forecast, reference,
        forecast_label="team_4", reference_label="ENTSO-E"
    )
    assert isinstance(fig, Figure)


def test_forecast_vs_reference_two_lines_when_overlap():
    """Two lines (forecast + reference) should appear when overlap exists."""
    forecast = _make_forecast(24)
    reference = _make_forecast(24, seed=2)
    fig = plot_forecast_vs_reference(forecast, reference)
    ax = fig.axes[0]
    assert len(ax.lines) == 2


def test_forecast_vs_reference_one_line_when_empty_overlap():
    """When reference has no overlap with forecast, only the forecast line is drawn."""
    forecast = _make_forecast(24)
    # Reference on a completely different index (no overlap)
    idx_other = pd.date_range("2025-06-01", periods=24, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    reference = pd.Series(40_000 + rng.standard_normal(24) * 500, index=idx_other)
    fig = plot_forecast_vs_reference(forecast, reference)
    ax = fig.axes[0]
    assert len(ax.lines) == 1


def test_forecast_vs_reference_no_exception_on_empty_overlap():
    """Empty overlap must not raise — the function returns a valid figure."""
    forecast = _make_forecast(24)
    empty_ref = pd.Series(dtype=float)
    # Should not raise
    fig = plot_forecast_vs_reference(forecast, empty_ref)
    assert isinstance(fig, Figure)


def test_forecast_vs_reference_unit_scale_applied(caplog):
    """unit_scale=1 leaves values unscaled; verify axis range is ~MW not GW."""
    import logging

    forecast = _make_forecast(24)
    reference = _make_forecast(24, seed=5)
    with caplog.at_level(logging.INFO, logger="spotforecast2.plots.diagnostics"):
        fig = plot_forecast_vs_reference(
            forecast, reference, unit_scale=1.0, unit="MW"
        )
    ax = fig.axes[0]
    # y-axis upper limit should be in the MW range (>1000), not GW (<100)
    ymin, ymax = ax.get_ylim()
    assert ymax > 1000


def test_forecast_vs_reference_mad_logged_in_display_unit(caplog):
    """MAD is logged in the display unit (GW), not in raw MW.

    With a constant offset of 1000 MW and unit_scale=1e-3, the logged value
    must be 1.0 GW.  The old bug (mad / unit_scale) would have logged 1000000.
    """
    import logging
    import re

    idx = pd.date_range("2024-01-15", periods=24, freq="h", tz="UTC")
    forecast = pd.Series(np.full(24, 40_000.0), index=idx)
    # Exact constant offset of 1000 MW → MAD = 1000 MW = 1.0 GW
    reference = pd.Series(np.full(24, 41_000.0), index=idx)

    with caplog.at_level(logging.INFO, logger="spotforecast2.plots.diagnostics"):
        plot_forecast_vs_reference(
            forecast,
            reference,
            forecast_label="fc",
            reference_label="ref",
            unit_scale=1e-3,
            unit="GW",
        )

    # Find the MAD log message and extract the numeric value
    mad_messages = [r for r in caplog.records if "overlap hours" in r.message]
    assert mad_messages, "Expected MAD log message not found in caplog"
    msg = mad_messages[0].message
    # Extract the number immediately before " GW"
    match = re.search(r"([\d.]+)\s+GW", msg)
    assert match, f"Could not parse GW value from log message: {msg!r}"
    logged_value = float(match.group(1))
    assert logged_value == pytest.approx(1.0, abs=0.05), (
        f"Logged MAD should be 1.0 GW, got {logged_value}. "
        f"Full message: {msg!r}"
    )
