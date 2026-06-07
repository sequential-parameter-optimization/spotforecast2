# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings

from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.processing.n2n_predict import n2n_predict

warnings.simplefilter("ignore")


def main():
    """Run the N-to-1 baseline forecasting pipeline with automatic data acquisition.

    Fetches time-series data from the default source (no explicit DataFrame
    supplied), applies outlier detection, imputation, and equivalent-date
    forecasting via `n2n_predict`, then aggregates the per-target predictions
    into a single combined series via `agg_predict`.

    This function is the CLI entry point registered as
    `spotforecast-n2o1` in `pyproject.toml`.  It requires the target CSV file
    to be present in the data home directory or a network connection to fetch
    it automatically.

    Examples:
        ```{python}
        # Demonstrate the n2n_predict + agg_predict pipeline that main() wires
        # together, using a small synthetic DataFrame instead of the live data
        # source that main() fetches automatically.
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.processing.agg_predict import agg_predict
        from spotforecast2_safe.processing.n2n_predict import n2n_predict

        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=500, freq="h", tz="UTC")
        data = pd.DataFrame(
            rng.standard_normal((500, 2)),
            index=dates,
            columns=["solar", "wind"],
        )

        predictions, forecasters = n2n_predict(
            data=data,
            columns=["solar", "wind"],
            forecast_horizon=3,
            contamination=0.01,
            window_size=24,
            verbose=False,
            show_progress=False,
        )
        print("Predictions shape:", predictions.shape)
        assert predictions.shape == (3, 2)
        assert set(predictions.columns) == {"solar", "wind"}

        combined = agg_predict(predictions, weights=[1.0, -1.0])
        print("Combined prediction:", combined.tolist())
        assert len(combined) == 3
        ```
    """
    FORECAST_HORIZON = 24
    CONTAMINATION = 0.01
    WINDOW_SIZE = 72
    VERBOSE = True
    SHOW_PROGRESS = True
    WEIGHTS = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]

    print("--- Starting n_to_1_task using modular functions ---")

    # --- Prediction ---
    # Fetch, Preprocess, Train, Evaluate, Predict
    predictions, _ = n2n_predict(
        columns=None,
        forecast_horizon=FORECAST_HORIZON,
        contamination=CONTAMINATION,
        window_size=WINDOW_SIZE,
        verbose=VERBOSE,
        show_progress=SHOW_PROGRESS,
    )

    print("\nMulti-output predictions head:")
    print(predictions)

    # --- Aggregation ---
    print("Calculating combined prediction...")
    combined_prediction = agg_predict(predictions, weights=WEIGHTS)

    print("Combined Prediction:")
    print(combined_prediction)


if __name__ == "__main__":
    main()
