# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for PredictionFigure.save_to_file and PredictionFigure.write_to_file."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2.plots.plotter import PredictionFigure


@pytest.fixture
def figure():
    rng = np.random.default_rng(0)
    train_idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    future_idx = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
    pkg = {
        "train_actual": pd.Series(rng.standard_normal(48) + 100, index=train_idx),
        "train_pred": pd.Series(rng.standard_normal(48) + 100, index=train_idx),
        "future_actual": pd.Series(rng.standard_normal(24) + 100, index=future_idx),
        "future_pred": pd.Series(rng.standard_normal(24) + 100, index=future_idx),
        "future_forecast": pd.Series(rng.standard_normal(24) + 100, index=future_idx),
        "metrics_train": {"mae": 4.0, "mape": 0.08},
        "metrics_future": {"mae": 6.0, "mape": 0.12},
        "metrics_future_one_day": {"mae": 5.0, "mape": 0.10},
        "metrics_forecast": {"mae": 7.0, "mape": 0.14},
        "metrics_forecast_one_day": {"mae": 5.5, "mape": 0.11},
    }
    fig = PredictionFigure(pkg, title="Smoke")
    fig.make_plot()
    return fig


def test_save_to_file_writes_png_to_tmp_path(figure, tmp_path):
    out = tmp_path / "out.png"
    result = figure.save_to_file(out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_to_file_raises_when_parent_missing(figure, tmp_path):
    bad = tmp_path / "does_not_exist" / "out.png"
    with pytest.raises(FileNotFoundError):
        figure.save_to_file(bad)


def test_write_to_file_emits_html_with_plotly_div(figure, tmp_path):
    out = tmp_path / "report.html"
    result = figure.write_to_file(out)
    assert result == out
    content = out.read_text(encoding="utf-8")
    assert "plotly-graph-div" in content
    assert "<title>Smoke</title>" in content


def test_write_to_file_uses_custom_template_when_given(figure, tmp_path):
    tpl = tmp_path / "tpl.html.j2"
    tpl.write_text(
        "<html><body><h2>CUSTOM-{{ title }}</h2>{{ fig | safe }}</body></html>",
        encoding="utf-8",
    )
    out = tmp_path / "custom.html"
    figure.write_to_file(out, template_path=tpl)
    content = out.read_text(encoding="utf-8")
    assert "CUSTOM-Smoke" in content
    assert "plotly-graph-div" in content


def test_write_to_file_raises_when_template_missing(figure, tmp_path):
    out = tmp_path / "x.html"
    with pytest.raises(FileNotFoundError):
        figure.write_to_file(out, template_path=tmp_path / "missing.html.j2")
