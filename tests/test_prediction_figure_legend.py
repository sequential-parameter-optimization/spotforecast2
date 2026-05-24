# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the inline MAE / MAPE legend strings on PredictionFigure."""

import numpy as np
import pandas as pd
import pytest

from spotforecast2.plots.plotter import PredictionFigure


@pytest.fixture
def package():
    """Minimal prediction package covering the documented keys."""
    rng = np.random.default_rng(0)
    train_idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    future_idx = pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC")
    return {
        "train_actual": pd.Series(rng.standard_normal(48) + 100, index=train_idx),
        "train_pred": pd.Series(rng.standard_normal(48) + 100, index=train_idx),
        "future_actual": pd.Series(rng.standard_normal(24) + 100, index=future_idx),
        "future_pred": pd.Series(rng.standard_normal(24) + 100, index=future_idx),
        "future_forecast": pd.Series(rng.standard_normal(24) + 100, index=future_idx),
        "metrics_train": {"mae": 4.2, "mape": 0.08},
        "metrics_future": {"mae": 6.7, "mape": 0.13},
        "metrics_future_one_day": {"mae": 5.1, "mape": 0.10},
        "metrics_forecast": {"mae": 7.0, "mape": 0.14},
        "metrics_forecast_one_day": {"mae": 5.3, "mape": 0.11},
    }


def _all_legend_text(fig) -> str:
    return "\n".join(trace.name or "" for trace in fig.data)


def test_prediction_figure_legend_contains_training_metrics(package):
    fig = PredictionFigure(package).make_plot()
    legend = _all_legend_text(fig)
    assert "Training: MAE=4" in legend
    assert "MAPE=0.08" in legend


def test_prediction_figure_legend_contains_24h_horizon_metrics(package):
    fig = PredictionFigure(package).make_plot()
    legend = _all_legend_text(fig)
    assert "Prediction (24h): MAE=5" in legend


def test_prediction_figure_legend_contains_full_horizon_metrics(package):
    fig = PredictionFigure(package).make_plot()
    legend = _all_legend_text(fig)
    # metrics_future.mae=6.7 rounds to 7 via the "{:.0f}" formatter.
    # Asserts the multi-row pred_legend block carries the full-horizon line.
    assert "MAE=7, MAPE=0.13" in legend


def test_prediction_figure_legend_contains_benchmark_metrics(package):
    fig = PredictionFigure(package).make_plot()
    legend = _all_legend_text(fig)
    assert "Benchmark (24h): MAE=5" in legend
    assert "Benchmark Forecast" in legend


def test_prediction_figure_does_not_crash_when_metrics_missing(package):
    """Missing metric dicts should not crash the figure builder."""
    minimal = {
        k: v
        for k, v in package.items()
        if not k.startswith("metrics_")
    }
    minimal["future_forecast"] = package["future_forecast"]
    fig = PredictionFigure(minimal).make_plot()
    assert len(fig.data) >= 1
