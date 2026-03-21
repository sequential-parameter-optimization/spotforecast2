# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import logging
from typing import Any

import pandas as pd
from spotforecast2_safe.preprocessing import LinearlyInterpolateTS, WeightFunction


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
        >>> from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home
        >>> from spotforecast2.preprocessing.imputation import custom_weights
        >>> data = fetch_data(filename=get_data_home() / "data_in.csv")
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
        >>> from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home
        >>> from spotforecast2.preprocessing.imputation import get_missing_weights
        >>> data = fetch_data(filename=get_data_home() / "data_in.csv")
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


def apply_imputation(
    df_pipeline: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, WeightFunction | None]:
    """Apply imputation to a DataFrame based on the method specified in config.

    Supports two strategies:

    - ``"weighted"``: forward-fill then backward-fill gaps, then build a
      :class:`~spotforecast2_safe.preprocessing.WeightFunction` that
      down-weights training rows near any gap.  Rows inside a gap receive
      weight 0; the rolling window ``config.window_size`` controls how far
      the penalty extends.
    - ``"linear"``: apply
      :class:`~spotforecast2_safe.preprocessing.LinearlyInterpolateTS`
      column-by-column.

    A diagnostic summary (NaN count before **and** after imputation) is
    always written to the logger.

    Args:
        df_pipeline (pd.DataFrame): DataFrame to impute.  Modified in-place
            for the ``"linear"`` method; a new DataFrame is returned for
            ``"weighted"`` (via :func:`get_missing_weights`).
        config: Configuration object that must expose:
            - ``imputation_method`` (``str``): ``"weighted"`` or ``"linear"``.
            - ``targets`` (``list[str]``): column names to interpolate
              (``"linear"`` method only).
            - ``window_size`` (``int``): rolling-window size passed to
              :func:`get_missing_weights` (``"weighted"`` method only).
        logger (logging.Logger): Standard-library logger used to emit
            ``INFO`` and ``WARNING`` messages.

    Returns:
        tuple[pd.DataFrame, WeightFunction | None]: A two-element tuple:

        - **df_pipeline** – imputed DataFrame with no NaN values (when the
            chosen method can fill all gaps).
        - **weight_func** – a :class:`~spotforecast2_safe.preprocessing.WeightFunction`
            instance ready to be passed to a forecaster's ``weight_func``
            parameter, or ``None`` when ``"linear"`` imputation is used.

    Raises:
        ValueError: If ``config.imputation_method`` is neither ``"weighted"``
            nor ``"linear"``.

    Examples:
        ```{python}
        import logging
        import pandas as pd
        import numpy as np
        from types import SimpleNamespace
        from spotforecast2.preprocessing.imputation import apply_imputation

        # Build a small DataFrame with deliberate gaps
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {"A": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]},
            index=idx,
        )

        # Minimal config and stdlib logger
        config = SimpleNamespace(
            imputation_method="linear",
            targets=["A"],
            window_size=3,
        )
        logger = logging.getLogger("demo")

        imputed, weight_func = apply_imputation(df, config, logger)
        print(imputed["A"].tolist())
        print(weight_func)  # None for linear method
        ```
    """
    nan_before = int(df_pipeline.isnull().sum().sum())
    logger.info(
        "apply_imputation: NaN cells before imputation: %d " "(method=%r, shape=%s)",
        nan_before,
        config.imputation_method,
        df_pipeline.shape,
    )

    weight_func = None  # default: no sample weighting

    if config.imputation_method == "weighted":
        logger.info("Applying weighted imputation (n2n style)...")
        df_pipeline, weights_series = get_missing_weights(
            df_pipeline,
            window_size=config.window_size,
            verbose=True,
        )
        weight_func = WeightFunction(weights_series)
        logger.info("Weight function created with %d entries.", len(weights_series))
    elif config.imputation_method == "linear":
        logger.info("Applying linear interpolation...")
        interpolator = LinearlyInterpolateTS()
        # LinearlyInterpolateTS expects a Series; apply per column
        for col in config.targets:
            series = df_pipeline[col]
            df_pipeline[col] = interpolator.fit_transform(series)
    else:
        raise ValueError(
            f"Unknown imputation_method: {config.imputation_method!r}. "
            "Expected one of: 'weighted', 'linear'."
        )

    nan_after = int(df_pipeline.isnull().sum().sum())
    logger.info("apply_imputation: NaN cells after imputation: %d", nan_after)
    if nan_after > 0:
        logger.warning(
            "apply_imputation: %d NaN cell(s) remain after imputation. "
            "Consider reviewing the data or adjusting the imputation method.",
            nan_after,
        )
    return df_pipeline, weight_func
