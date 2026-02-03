import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from spotforecast2.forecaster.recursive import ForecasterEquivalentDate
from spotforecast2.data.fetch_data import fetch_data
from spotforecast2.preprocessing.curate_data import basic_ts_checks
from spotforecast2.preprocessing.curate_data import agg_and_resample_data
from spotforecast2.preprocessing.outlier import mark_outliers

from spotforecast2.preprocessing.split import split_rel_train_val_test
from spotforecast2.forecaster.utils import predict_multivariate
from spotforecast2.model_selection import TimeSeriesFold, backtesting_forecaster
from spotforecast2.preprocessing.curate_data import get_start_end


def n2n_predict(
    columns: Optional[List[str]] = None,
    forecast_horizon: int = 24,
    contamination: float = 0.01,
    window_size: int = 72,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    End-to-end prediction function replicating the workflow from 01_base_predictor combined with fetch_data.

    Args:
        columns: List of target columns to forecast. If None, uses a default set (defined internally or from data).
                 Note: fetch_data now supports None to return all columns.
        forecast_horizon: Number of steps to forecast.
        contamination: Contamination factor for outlier detection.
        window_size: Window size for weighting (not fully utilized in main flow but kept for consistency).
        verbose: Whether to print progress logs.

    Returns:
        Tuple containing:
            - predictions (pd.DataFrame): The multi-output predictions.
            - metrics (Optional[Dict]): Dictionary containing backtesting metrics if performed.
    """
    if columns is not None:
        TARGET = columns
    else:
        TARGET = None

    if verbose:
        print("--- Starting n2n_predict ---")
        print("Fetching data...")

    # Fetch data
    data = fetch_data(columns=TARGET)

    START, END, COV_START, COV_END = get_start_end(
        data=data,
        forecast_horizon=forecast_horizon,
        verbose=verbose,
    )

    basic_ts_checks(data, verbose=verbose)

    data = agg_and_resample_data(data, verbose=verbose)

    # --- Outlier Handling ---
    if verbose:
        print("Handling outliers...")

    # data_old = data.copy() # kept in notebook, maybe useful for debugging but not used logic-wise here
    data, outliers = mark_outliers(
        data, contamination=contamination, random_state=1234, verbose=verbose
    )

    # --- Missing Data (Imputation) ---
    if verbose:
        print("Imputing missing data...")

    missing_indices = data.index[data.isnull().any(axis=1)]
    if verbose:
        n_missing = len(missing_indices)
        pct_missing = (n_missing / len(data)) * 100
        print(f"Number of rows with missing values: {n_missing}")
        print(f"Percentage of rows with missing values: {pct_missing:.2f}%")

    data = data.ffill()
    data = data.bfill()

    # --- Train, Val, Test Split ---
    if verbose:
        print("Splitting data...")
    data_train, data_val, data_test = split_rel_train_val_test(
        data, perc_train=0.8, perc_val=0.2, verbose=verbose
    )

    # --- Model Fit ---
    if verbose:
        print("Fitting models...")

    end_validation = pd.concat([data_train, data_val]).index[-1]

    baseline_forecasters = {}

    for target in data.columns:
        forecaster = ForecasterEquivalentDate(offset=pd.DateOffset(days=1), n_offsets=1)

        forecaster.fit(y=data.loc[:end_validation, target])

        baseline_forecasters[target] = forecaster

    if verbose:
        print("✓ Multi-output baseline system trained")

    
    # --- Predict ---
    if verbose:
        print("Generating predictions...")

    predictions = predict_multivariate(
        baseline_forecasters, steps_ahead=forecast_horizon
    )

    return predictions
