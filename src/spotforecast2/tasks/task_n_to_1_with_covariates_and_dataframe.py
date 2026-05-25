# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
N-to-1 Forecasting with Exogenous Covariates and Prediction Aggregation.

This module implements a complete end-to-end pipeline for multi-step time series
forecasting with exogenous variables (weather, holidays, calendar features),
followed by prediction aggregation using configurable weights.

The pipeline:
    1. Performs multi-output recursive forecasting with exogenous covariates
    2. Aggregates predictions using weighted combinations
    3. Supports flexible model selection (string or object-based)
    4. Allows customization via kwargs for all underlying functions

Key Features:
    - Automatic weather, holiday, and calendar feature generation
    - Cyclical and polynomial feature engineering
    - Configurable recursive forecaster with LGBMRegressor default
    - Weighted prediction aggregation
    - Comprehensive parameter flexibility via **kwargs
    - Detailed logging and progress tracking

Examples:
    ```{python}
    import tempfile

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor

    from spotforecast2.tasks.task_n_to_1_with_covariates_and_dataframe import (
        n_to_1_with_covariates,
    )

    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "A": rng.normal(50, 5, n),
            "B": rng.normal(30, 3, n),
        },
        index=idx,
    )

    estimator = LGBMRegressor(n_estimators=50, verbose=-1)
    predictions, combined, metrics, feature_info = n_to_1_with_covariates(
        data=data,
        forecast_horizon=2,
        lags=3,
        window_size=6,
        train_ratio=0.8,
        estimator=estimator,
        weights=[1.0, -1.0],
        verbose=False,
        show_progress=False,
        on_weather_failure="skip",
        force_train=True,
        model_dir=tempfile.mkdtemp(),
    )
    print(f"Predictions shape: {predictions.shape}")
    print(f"Combined forecast length: {len(combined)}")
    assert predictions.shape[0] == 2
    assert len(combined) == 2
    ```

Available Parameters:

Forecasting Parameters:
    forecast_horizon (int): Number of steps ahead to forecast. Default: 24.
    contamination (float): Outlier detection threshold [0, 1]. Default: 0.01.
    window_size (int): Rolling window size for feature engineering. Default: 72.
    lags (int): Number of lag features to create. Default: 24.
    train_ratio (float): Train-test split ratio [0, 1]. Default: 0.8.
    verbose (bool): Enable detailed progress logging. Default: True.

Location & Time Parameters:
    latitude (float): Location latitude for sun features. Default: 51.5136 (Dortmund).
    longitude (float): Location longitude for sun features. Default: 7.4653 (Dortmund).
    timezone (str): Timezone for data processing. Default: "UTC".
    country_code (str): Country code for holidays (ISO 3166-1 alpha-2). Default: "DE".
    state (str): State/region code for holidays (depends on country). Default: "NW".

Feature Engineering Parameters:
    include_weather_windows (bool): Include rolling weather statistics. Default: False.
    include_holiday_features (bool): Include holiday indicator features. Default: False.
    include_poly_features (bool): Include polynomial interaction features. Default: False.

Model Parameters:
    estimator (Optional[Union[str, object]]): Forecaster estimator. Can be:
        - None: Uses default LGBMRegressor(n_estimators=100)
        - "ForecasterRecursive": String reference (uses default)
        - LGBMRegressor(...): Custom estimator object
        Default: None.

Aggregation Parameters:
    weights (Optional[Union[Dict[str, float], List[float], np.ndarray]]):
        Weights for prediction aggregation. Can be:
        - None: Defaults to uniform weights (1.0 for each column)
        - Dict: Column name -> weight mapping
        - List/Array: Weights in column order
        Default: [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0].
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home
from spotforecast2_safe.processing.agg_predict import agg_predict
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)

warnings.simplefilter("ignore")


def n_to_1_with_covariates(
    data: Optional[pd.DataFrame] = None,
    forecast_horizon: int = 24,
    contamination: float = 0.01,
    window_size: int = 72,
    lags: int = 24,
    train_ratio: float = 0.8,
    latitude: float = 51.5136,
    longitude: float = 7.4653,
    timezone: str = "UTC",
    country_code: str = "DE",
    state: str = "NW",
    estimator: Optional[Union[str, object]] = None,
    include_weather_windows: bool = False,
    include_holiday_features: bool = False,
    include_poly_features: bool = False,
    weights: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    verbose: bool = True,
    show_progress: bool = True,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.Series, Dict, Dict]:
    """Execute N-to-1 forecasting pipeline with exogenous covariates.

    This function performs a complete time series forecasting workflow:
    1. Fetches and preprocesses data
    2. Engineers features (calendar, weather, holidays, cyclical, polynomial)
    3. Trains recursive forecaster on multiple targets
    4. Aggregates predictions using weighted combination

    Args:
        data (Optional[pd.DataFrame]): Optional DataFrame with target time series data.
            If None, fetches data automatically. Default: None.

        forecast_horizon (int): Number of forecast steps ahead.
            Determines how many time steps to predict into the future.
            Typical values: 24 (1 day), 48 (2 days), 168 (1 week). Default: 24.

        contamination (float): Outlier contamination level for anomaly detection.
            Expected proportion of outliers in the training data [0, 1].
            Higher values detect fewer outliers. Default: 0.01 (1%).

        window_size (int): Rolling window size for feature engineering (hours).
            Size of the rolling window for computing statistics.
            Must be > lags. Typical range: 24-168. Default: 72.

        lags (int): Number of lagged features to create.
            Creates AR(p) features with p=lags.
            Typical values: 12, 24, 48. Default: 24.

        train_ratio (float): Proportion of data for training [0, 1].
            Remaining data (1 - train_ratio) used for validation/testing.
            Typical values: 0.7-0.9. Default: 0.8.

        latitude (float): Geographic latitude for solar features.
            Used to compute sunrise/sunset times for day/night features.
            Default: 51.5136 (Dortmund, Germany).

        longitude (float): Geographic longitude for solar features.
            Used to compute sunrise/sunset times for day/night features.
            Default: 7.4653 (Dortmund, Germany).

        timezone (str): Timezone for time-based features.
            Any timezone recognized by pytz. Default: "UTC".

        country_code (str): ISO 3166-1 alpha-2 country code for holidays.
            Examples: "DE" (Germany), "US" (USA), "GB" (UK). Default: "DE".

        state (str): State/region code for holidays.
            Country-dependent. For Germany: "BW", "BY", "NW", etc.
            Default: "NW" (Nordrhein-Westfalen).

        estimator (Optional[Union[str, object]]): Forecaster model.
            Can be:
            - None: Uses LGBMRegressor(n_estimators=100, verbose=-1).
            - "ForecasterRecursive": References default estimator (same as None).
            - LGBMRegressor(...): Custom pre-configured estimator.
            - Any sklearn-compatible regressor.
            Default: None.

        include_weather_windows (bool): Add rolling weather statistics.
            Creates moving averages, min, max of weather features over
            multiple windows (1D, 7D). Increases feature count significantly.
            Default: False.

        include_holiday_features (bool): Add holiday binary indicators.
            Creates features indicating holidays and special dates.
            Useful for capturing demand patterns around holidays.
            Default: False.

        include_poly_features (bool): Add polynomial interactions.
            Creates 2nd-order interaction terms between selected features.
            Useful for capturing non-linear relationships.
            Default: False.

        weights (Optional[Union[Dict[str, float], List[float], np.ndarray]]):
            Weights for combining multi-output predictions.
            Can be:
            - None: Default weights [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]
            - Dict: {"col_name": weight, ...} for specific columns
            - List: [w1, w2, ...] in column order
            - np.ndarray: Same as list
            Default: None (uses default weights).

        verbose (bool): Enable progress logging.
            Prints intermediate results and timestamps.
            Default: True.

        show_progress (bool): Show a progress bar for major pipeline steps.
            Default: True.

        **kwargs (Any): Additional parameters for underlying functions.
            These are passed to n2n_predict_with_covariates().
            Examples:
            - freq: Frequency for data resampling. Default: "h" (hourly).
            - columns: Specific columns to forecast. Default: None (all).
            Any parameter accepted by n2n_predict_with_covariates().

    Returns:
        Tuple[pd.DataFrame, pd.Series, Dict, Dict]: A tuple containing:
            - predictions (pd.DataFrame): Multi-output forecasts from recursive model.
                Each column represents a target variable.
                Index is datetime matching the forecast period.
            - combined_prediction (pd.Series): Aggregated forecast from weighted combination.
                Single column combining all output predictions.
                Index is datetime matching the forecast period.
            - model_metrics (Dict): Performance metrics from recursive forecaster.
                Keys may include: 'mae', 'rmse', 'mape', etc.
            - feature_info (Dict): Information about engineered features.
                Contains feature counts, types, and engineering details.

    Raises:
        ValueError: If forecast_horizon <= 0 or invalid parameter combinations.
        FileNotFoundError: If data source files cannot be accessed.
        RuntimeError: If model training fails or data processing errors occur.

    Examples:
        ```{python}
        import tempfile

        import numpy as np
        import pandas as pd
        from lightgbm import LGBMRegressor

        from spotforecast2.tasks.task_n_to_1_with_covariates_and_dataframe import (
            n_to_1_with_covariates,
        )

        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
        data = pd.DataFrame(
            {
                "target_a": rng.normal(100, 10, n),
                "target_b": rng.normal(60, 6, n),
            },
            index=idx,
        )

        custom_estimator = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            verbose=-1,
        )
        custom_weights = [1.0, -1.0]

        predictions, combined, metrics, feature_info = n_to_1_with_covariates(
            data=data,
            forecast_horizon=2,
            lags=3,
            window_size=6,
            train_ratio=0.8,
            estimator=custom_estimator,
            weights=custom_weights,
            verbose=False,
            show_progress=False,
            on_weather_failure="skip",
            force_train=True,
            model_dir=tempfile.mkdtemp(),
        )
        print(f"Predictions shape: {predictions.shape}")
        print("Combined forecast head:")
        print(combined.head())
        print(f"Feature info keys: {sorted(feature_info.keys())[:4]}")
        assert predictions.shape[0] == 2
        assert len(combined) == 2
        assert isinstance(metrics, dict)
        ```
    """
    # Default weights if not provided
    if weights is None:
        weights = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]

    if verbose:
        print("=" * 80)
        print("N-to-1 Forecasting with Exogenous Covariates")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Forecast Horizon: {forecast_horizon} steps")
        print(f"  Contamination Level: {contamination}")
        print(f"  Window Size: {window_size}")
        print(f"  Lags: {lags}")
        print(f"  Train Ratio: {train_ratio}")
        print("  Location: Lat=***, Lon=***")
        print("  Timezone: ***")
        print("  Country Code: ***, State: ***")
        print(f"  Estimator: {type(estimator).__name__ if estimator else 'Default'}")
        print("  Feature Engineering:")
        print(f"    - Weather Windows: {include_weather_windows}")
        print(f"    - Holiday Features: {include_holiday_features}")
        print(f"    - Polynomial Features: {include_poly_features}")
        print(f"  Weights Type: {type(weights).__name__}")
        print(f"\n{'=' * 80}\n")

    # --- Step 1: Multi-Output Recursive Forecasting with Covariates ---
    if verbose:
        print("Step 1: Executing multi-output recursive forecasting...")

    # Prepare kwargs for n2n_predict_with_covariates
    forecast_kwargs = {
        "data": data,
        "forecast_horizon": forecast_horizon,
        "contamination": contamination,
        "window_size": window_size,
        "lags": lags,
        "train_ratio": train_ratio,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "country_code": country_code,
        "state": state,
        "estimator": estimator,
        "include_weather_windows": include_weather_windows,
        "include_holiday_features": include_holiday_features,
        "include_poly_features": include_poly_features,
        "verbose": verbose,
        "show_progress": show_progress,
    }

    # Add any additional kwargs
    forecast_kwargs.update(kwargs)

    # Execute recursive forecasting
    predictions, model_metrics, feature_info = n2n_predict_with_covariates(
        **forecast_kwargs
    )

    if verbose:
        print(f"\nMulti-output predictions shape: {predictions.shape}")
        print(f"Output columns: {list(predictions.columns)}")
        print(f"Date range: {predictions.index[0]} to {predictions.index[-1]}")

    # --- Step 2: Prediction Aggregation ---
    if verbose:
        print("\nStep 2: Aggregating predictions using weighted combination...")

    combined_prediction = agg_predict(predictions, weights=weights)

    if verbose:
        print(f"Combined prediction shape: {combined_prediction.shape}")
        print("\nAggregation Summary:")
        print("  Combined Prediction Head:")
        print(combined_prediction.head())
        print("\n  Combined Prediction Statistics:")
        print(f"    Mean: {combined_prediction.mean():.4f}")
        print(f"    Std:  {combined_prediction.std():.4f}")
        print(f"    Min:  {combined_prediction.min():.4f}")
        print(f"    Max:  {combined_prediction.max():.4f}")
        print(f"\n{'=' * 80}\n")

    return predictions, combined_prediction, model_metrics, feature_info


def main() -> None:
    """Execute the complete N-to-1 forecasting pipeline with default parameters.

    This is the entry point when running the script directly. It executes the full
    forecasting pipeline with default settings and prints comprehensive results.

    The default configuration:
    - Forecasts 24 steps ahead
    - Uses Dortmund, Germany coordinates
    - Applies default contamination and window parameters
    - Aggregates with predefined weights
    - Provides verbose output

    Returns:
        None. Results are printed to stdout.

    Examples:
        ```{python}
        #| eval: false
        # main() reads data_in.csv from the user's data home directory; not reproducible without that file.
        from spotforecast2.tasks.task_n_to_1_with_covariates_and_dataframe import main

        main()
        ```
    """
    data = fetch_data(filename=get_data_home() / "data_in.csv")

    FORECAST_HORIZON = 24
    CONTAMINATION = 0.01
    WINDOW_SIZE = 72
    LAGS = 24
    TRAIN_RATIO = 0.8
    LATITUDE = 51.5136
    LONGITUDE = 7.4653
    TIMEZONE = "UTC"
    COUNTRY_CODE = "DE"
    STATE = "NW"
    INCLUDE_WEATHER_WINDOWS = False
    INCLUDE_HOLIDAY_FEATURES = False
    INCLUDE_POLY_FEATURES = False
    VERBOSE = False
    WEIGHTS = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]

    print("--- Starting n_to_1_with_covariates using modular functions ---")

    # Execute the forecasting pipeline
    predictions, combined_prediction, model_metrics, feature_info = (
        n_to_1_with_covariates(
            data=data,
            forecast_horizon=FORECAST_HORIZON,
            contamination=CONTAMINATION,
            window_size=WINDOW_SIZE,
            lags=LAGS,
            train_ratio=TRAIN_RATIO,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            timezone=TIMEZONE,
            country_code=COUNTRY_CODE,
            state=STATE,
            estimator=None,
            include_weather_windows=INCLUDE_WEATHER_WINDOWS,
            include_holiday_features=INCLUDE_HOLIDAY_FEATURES,
            include_poly_features=INCLUDE_POLY_FEATURES,
            weights=WEIGHTS,
            verbose=VERBOSE,
        )
    )

    # Print results (similar to n_to_1_task.py)
    print("\nMulti-output predictions head:")
    print(predictions)

    print("Calculating combined prediction...")
    print("Combined Prediction:")
    print(combined_prediction)


if __name__ == "__main__":
    main()
