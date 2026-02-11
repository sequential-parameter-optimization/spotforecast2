# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import pandas as pd


def custom_weights(index, weights_series: pd.Series) -> float:
    """
    Return 0 if index is in or near any gap.

    Args:
        index (pd.Index):
            The index to check.
        weights_series (pd.Series):
            Series containing weights.

    Returns:
        float: The weight corresponding to the index.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.imputation import custom_weights
        >>> data = fetch_data(filename="data_in.csv")
        >>> _, missing_weights = get_missing_weights(data, window_size=72, verbose=False)
        >>> for idx in data.index[:5]:
        ...     weight = custom_weights(idx, missing_weights)
        ...     print(f"Index: {idx}, Weight: {weight}")
    """
    # do plausibility check
    if isinstance(index, pd.Index):
        if not index.isin(weights_series.index).all():
            raise ValueError("Index not found in weights_series.")
        return weights_series.loc[index].values

    if index not in weights_series.index:
        raise ValueError("Index not found in weights_series.")
    return weights_series.loc[index]


def get_missing_weights(
    data: pd.DataFrame, window_size: int = 72, verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return imputed DataFrame and a series indicating missing weights.

    Args:
        data (pd.DataFrame):
            The input dataset.
        window_size (int):
            The size of the rolling window to consider for missing values.
        verbose (bool):
            Whether to print additional information.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            A tuple containing the forward and backward filled DataFrame and a boolean series where True indicates missing weights.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.imputation import get_missing_weights
        >>> data = fetch_data(filename="data_in.csv")
        >>> filled_data, missing_weights = get_missing_weights(data, window_size=72, verbose=True)

    """
    # first perform some checks if dataframe has enough data and if window_size is appropriate
    if data.shape[0] == 0:
        raise ValueError("Input data is empty.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if window_size >= data.shape[0]:
        raise ValueError("window_size must be smaller than the number of rows in data.")

    missing_indices = data.index[data.isnull().any(axis=1)]
    n_missing = len(missing_indices)
    if verbose:
        pct_missing = (n_missing / len(data)) * 100
        print(f"Number of rows with missing values: {n_missing}")
        print(f"Percentage of rows with missing values: {pct_missing:.2f}%")
        print(f"missing_indices: {missing_indices}")
    data = data.ffill()
    data = data.bfill()

    is_missing = pd.Series(0, index=data.index)
    is_missing.loc[missing_indices] = 1
    weights_series = 1 - is_missing.rolling(window=window_size + 1, min_periods=1).max()
    if verbose:
        n_missing_after = weights_series.isna().sum()
        pct_missing_after = (n_missing_after / len(data)) * 100
        print(
            f"Number of rows with missing weights after processing: {n_missing_after}"
        )
        print(
            f"Percentage of rows with missing weights after processing: {pct_missing_after:.2f}%"
        )
    return data, weights_series.isna()
