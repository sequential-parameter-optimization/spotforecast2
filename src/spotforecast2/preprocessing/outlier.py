from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


def mark_outliers(
    data: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 1234,
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Marks outliers as NaN in the dataset using Isolation Forest.

    Args:
        data (pd.DataFrame):
            The input dataset.
        contamination (float):
            The (estimated) proportion of outliers in the dataset.
        random_state (int):
            Random seed for reproducibility. Default is 1234.
        verbose (bool):
            Whether to print additional information.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the modified dataset with outliers marked as NaN and the outlier labels.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.outlier import mark_outliers
        >>> data = fetch_data()
        >>> cleaned_data, outlier_labels = mark_outliers(data, contamination=0.1, random_state=42, verbose=True)
    """
    for col in data.columns:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        # Fit and predict (-1 for outliers, 1 for inliers)
        outliers = iso.fit_predict(data[[col]])

        # Mark outliers as NaN
        data.loc[outliers == -1, col] = np.nan

        pct_outliers = (outliers == -1).mean() * 100
        if verbose:
            print(
                f"Column '{col}': Marked {pct_outliers:.4f}% of data points as outliers."
            )
    return data, outliers


def manual_outlier_removal(
    data: pd.DataFrame,
    column: str,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Manual outlier removal function.
    Args:
        data (pd.DataFrame):
            The input dataset.
        column (str):
            The column name in which to perform manual outlier removal.
        lower_threshold (float | None):
            The lower threshold below which values are considered outliers.
            If None, no lower threshold is applied.
        upper_threshold (float | None):
            The upper threshold above which values are considered outliers.
            If None, no upper threshold is applied.
        verbose (bool):
            Whether to print additional information.

    Returns:
        tuple[pd.DataFrame, int]: A tuple containing the modified dataset with outliers marked as NaN and the number of outliers marked.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_data
        >>> from spotforecast2.preprocessing.outlier import manual_outlier_removal
        >>> data = fetch_data()
        >>> data, n_manual_outliers = manual_outlier_removal(
        ...     data,
        ...     column='ABC',
        ...     lower_threshold=50,
        ...     upper_threshold=700,
        ...     verbose=True
    """
    if lower_threshold is None and upper_threshold is None:
        if verbose:
            print(f"No thresholds provided for {column}; no outliers marked.")
        return data, 0

    if lower_threshold is not None and upper_threshold is not None:
        mask = (data[column] > upper_threshold) | (data[column] < lower_threshold)
    elif lower_threshold is not None:
        mask = data[column] < lower_threshold
    else:
        mask = data[column] > upper_threshold

    n_manual_outliers = mask.sum()

    data.loc[mask, column] = np.nan

    if verbose:
        if lower_threshold is not None and upper_threshold is not None:
            print(
                f"Manually marked {n_manual_outliers} values > {upper_threshold} or < {lower_threshold} as outliers in {column}."
            )
        elif lower_threshold is not None:
            print(
                f"Manually marked {n_manual_outliers} values < {lower_threshold} as outliers in {column}."
            )
        else:
            print(
                f"Manually marked {n_manual_outliers} values > {upper_threshold} as outliers in {column}."
            )
    return data, n_manual_outliers
