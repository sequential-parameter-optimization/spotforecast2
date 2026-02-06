"""Outlier detection utilities (legacy wrapper for spotforecast2_safe)."""

from spotforecast2_safe.preprocessing.outlier import (
    get_outliers,
    mark_outliers,
    manual_outlier_removal,
    IsolationForest,
)

__all__ = [
    "get_outliers",
    "mark_outliers",
    "manual_outlier_removal",
    "IsolationForest",
]
