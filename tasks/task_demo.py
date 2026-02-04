"""
Task demo: compare baseline vs covariate forecasts against ground truth.

This script executes the baseline N-to-1 task and the covariate-enhanced N-to-1
pipeline, then loads the ground truth from ~/spotforecast2_data/data_test.csv
and plots Actual vs Predicted using Plotly.

The plot includes:
    - Actual combined values (ground truth)
    - Baseline combined prediction (n2n_predict)
    - Covariate combined prediction (n2n_predict_with_covariates)

Examples:
    Run the demo:

    >>> python tasks/task_demo.py
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from spotforecast2.processing.agg_predict import agg_predict
from spotforecast2.processing.n2n_predict import n2n_predict
from spotforecast2.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)

warnings.simplefilter("ignore")


def _load_actual_combined(
    data_path: str,
    columns: List[str],
    weights: List[float],
    forecast_horizon: int,
) -> pd.Series:
    """Load ground truth and compute combined actual series.

    Args:
        data_path: Path to the data_test.csv file.
        columns: Column names to use for aggregation.
        weights: Weight list aligned with columns.
        forecast_horizon: Number of steps to take from the start of test data.

    Returns:
        Combined actual values as a Series.
    """
    data_test = pd.read_csv(data_path, index_col=0, parse_dates=True)
    actual_df = data_test[columns].iloc[:forecast_horizon]
    return agg_predict(actual_df, weights=weights)


def _plot_actual_vs_predicted(
    actual_combined: pd.Series,
    baseline_combined: pd.Series,
    covariates_combined: pd.Series,
) -> None:
    """Plot actual vs predicted combined values.

    Args:
        actual_combined: Ground truth combined series.
        baseline_combined: Baseline combined prediction series.
        covariates_combined: Covariate-enhanced combined prediction series.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_combined.index,
            y=actual_combined.values,
            mode="lines+markers",
            name="Actual",
            line=dict(color="green", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_combined.index,
            y=baseline_combined.values,
            mode="lines+markers",
            name="Predicted (Baseline)",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=covariates_combined.index,
            y=covariates_combined.values,
            mode="lines+markers",
            name="Predicted (Covariates)",
            line=dict(color="blue", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Combined Values: Actual vs. Predicted",
        xaxis_title="Time",
        yaxis_title="Combined Value",
        width=1000,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )

    fig.show()


def main() -> None:
    """Run the demo, compute predictions, and plot actual vs predicted."""
    DATA_PATH = "~/spotforecast2_data/data_test.csv"
    FORECAST_HORIZON = 24
    CONTAMINATION = 0.01
    WINDOW_SIZE = 72
    LAGS = 24
    TRAIN_RATIO = 0.8
    VERBOSE = True
    SHOW_PROGRESS = True

    WEIGHTS = [
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
    ]

    print("--- Starting task_demo: baseline and covariates ---")

    # --- Baseline predictions ---
    baseline_predictions, _ = n2n_predict(
        columns=None,
        forecast_horizon=FORECAST_HORIZON,
        contamination=CONTAMINATION,
        window_size=WINDOW_SIZE,
        verbose=VERBOSE,
        show_progress=SHOW_PROGRESS,
    )

    baseline_combined = agg_predict(baseline_predictions, weights=WEIGHTS)

    # --- Covariate-enhanced predictions ---
    cov_predictions, _, _ = n2n_predict_with_covariates(
        forecast_horizon=FORECAST_HORIZON,
        contamination=CONTAMINATION,
        window_size=WINDOW_SIZE,
        lags=LAGS,
        train_ratio=TRAIN_RATIO,
        verbose=VERBOSE,
        show_progress=SHOW_PROGRESS,
    )

    covariates_combined = agg_predict(cov_predictions, weights=WEIGHTS)

    # --- Debug output ---
    print("\n=== DEBUG INFO ===")
    print(f"Baseline combined shape: {baseline_combined.shape}")
    print(f"Baseline index: {baseline_combined.index[0]} to {baseline_combined.index[-1]}")
    print(f"Baseline values (first 5): {baseline_combined.head().values}")
    print(f"\nCovariates combined shape: {covariates_combined.shape}")
    print(f"Covariates index: {covariates_combined.index[0]} to {covariates_combined.index[-1]}")
    print(f"Covariates values (first 5): {covariates_combined.head().values}")
    print(f"\nAre indices aligned? {(baseline_combined.index == covariates_combined.index).all()}")
    print(f"Are values identical? {(baseline_combined.values == covariates_combined.values).all()}")
    if not (baseline_combined.values == covariates_combined.values).all():
        diff = baseline_combined - covariates_combined
        print(f"Difference stats:\n{diff.describe()}")
    print("==================\n")

    # --- Ground truth ---
    columns = list(baseline_predictions.columns)
    actual_combined = _load_actual_combined(
        data_path=DATA_PATH,
        columns=columns,
        weights=WEIGHTS,
        forecast_horizon=FORECAST_HORIZON,
    )

    # Align indices to predictions for clean plotting
    actual_combined = actual_combined.reindex(baseline_combined.index)

    # --- Plot ---
    _plot_actual_vs_predicted(
        actual_combined=actual_combined,
        baseline_combined=baseline_combined,
        covariates_combined=covariates_combined,
    )


if __name__ == "__main__":
    main()
