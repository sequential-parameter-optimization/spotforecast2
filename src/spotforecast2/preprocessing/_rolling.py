import numpy as np
import pandas as pd
from typing import List, Any
from ._common import (
    _np_mean_jit,
    _np_std_jit,
    _np_min_jit,
    _np_max_jit,
    _np_sum_jit,
    _np_median_jit,
)


class RollingFeatures:
    """
    Compute rolling features (stats) over a window of the time series.
    Compatible with scikit-learn transformers API (fit, transform).

    Attributes:
        stats_funcs (list): List of rolling statistics functions.
        window_sizes (list): List of window sizes.
        features_names (list): List of feature names.
    """

    def __init__(
        self,
        stats: str | List[str] | List[Any],
        window_sizes: int | List[int],
        features_names: List[str] | None = None,
    ):
        """
        Initialize the rolling features transformer.

        Args:
            stats (str | List[str] | List[Any]): Rolling statistics to compute.
            window_sizes (int | List[int]): Window sizes for rolling statistics.
            features_names (List[str] | None, optional): Names of the features.
                Defaults to None.
        """
        self.stats = stats
        self.window_sizes = window_sizes
        self.features_names = features_names

        # Validation and processing logic...
        self._validate_params()

    def _validate_params(self):
        """
        Validate the parameters of the rolling features transformer.
        """
        if isinstance(self.window_sizes, int):
            self.window_sizes = [self.window_sizes]

        if isinstance(self.stats, str):
            self.stats = [self.stats]

        # Map strings to functions
        valid_stats = {
            "mean": _np_mean_jit,
            "std": _np_std_jit,
            "min": _np_min_jit,
            "max": _np_max_jit,
            "sum": _np_sum_jit,
            "median": _np_median_jit,
        }

        self.stats_funcs = []
        for s in self.stats:
            if isinstance(s, str):
                if s not in valid_stats:
                    raise ValueError(
                        f"Stat '{s}' not supported. Supported: {list(valid_stats.keys())}"
                    )
                self.stats_funcs.append(valid_stats[s])
            else:
                self.stats_funcs.append(s)

        if self.features_names is None:
            self.features_names = []
            for ws in self.window_sizes:
                for s in self.stats:
                    s_name = s if isinstance(s, str) else s.__name__
                    self.features_names.append(f"roll_{s_name}_{ws}")

    def fit(self, X, y=None):
        """
        Fit the rolling features transformer.

        Args:
            X (np.ndarray): Time series to transform.
            y (object, optional): Ignored.

        Returns:
            self: Fitted rolling features transformer.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute rolling features.

        Args:
            X (np.ndarray): Time series to transform.

        Returns:
            np.ndarray: Array with rolling features.
        """
        # Assume X is 1D array
        n_samples = len(X)
        output = np.full((n_samples, len(self.features_names)), np.nan)

        idx_feature = 0
        for ws in self.window_sizes:
            for func in self.stats_funcs:
                # Naive rolling window loop - can be optimized or use pandas rolling
                # Using pandas for simplicity and speed if X is convertible
                series = pd.Series(X)
                rolled = series.rolling(window=ws).apply(func, raw=True)
                output[:, idx_feature] = rolled.values
                idx_feature += 1

        return output

    def transform_batch(self, X: pd.Series) -> pd.DataFrame:
        """
        Transform a pandas Series to rolling features DataFrame.

        Args:
            X (pd.Series): Time series to transform.

        Returns:
            pd.DataFrame: DataFrame with rolling features.
        """
        values = X.to_numpy()
        transformed = self.transform(values)
        return pd.DataFrame(transformed, index=X.index, columns=self.features_names)
