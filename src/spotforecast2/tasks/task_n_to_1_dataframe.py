# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings

from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home
from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.processing.n2n_predict import n2n_predict

warnings.simplefilter("ignore")


def main() -> None:
    """Execute the complete N-to-1 baseline forecasting pipeline with default parameters.

    This is the entry point when running the script directly. It fetches data
    from the user's data home directory, runs the equivalent-date baseline
    forecasting pipeline via `n2n_predict`, and aggregates the multi-output
    predictions into a single combined series with `agg_predict`.

    The default configuration:
    - Reads ``data_in.csv`` from `get_data_home()` (user-specific path)
    - Forecasts 24 steps ahead
    - Applies 1% contamination for outlier detection
    - Uses a 72-step rolling window
    - Aggregates with predefined weights

    Returns:
        None. Results are printed to stdout.

    Examples:
        ```{python}
        #| eval: false
        # main() reads data_in.csv from the user's data home directory; not reproducible without that file.
        from spotforecast2.tasks.task_n_to_1_dataframe import main

        main()
        ```
    """
    FORECAST_HORIZON = 24
    CONTAMINATION = 0.01
    WINDOW_SIZE = 72
    VERBOSE = True
    SHOW_PROGRESS = True
    WEIGHTS = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]

    df = fetch_data(filename=get_data_home() / "data_in.csv")

    print("--- Starting n_to_1_task using modular functions ---")

    # --- Prediction ---
    # Fetch, Preprocess, Train, Evaluate, Predict
    predictions, _ = n2n_predict(
        data=df,
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
