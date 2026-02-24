import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import (
    TimeSeriesFold,
    spotoptim_search_forecaster,
)
from spotforecast2.model_selection.spotoptim_search import (
    parse_lags_from_strings,
    convert_search_space,
)


def test_parse_lags_from_strings():
    """Test the internal lag string parser."""
    assert parse_lags_from_strings(24) == 24
    assert parse_lags_from_strings("[1, 2, 3]") == [1, 2, 3]
    assert parse_lags_from_strings(" [24, 48] ") == [24, 48]
    assert parse_lags_from_strings("lag_name") == "lag_name"


def test_convert_search_space_lags():
    """Test that lag search spaces are converted correctly."""
    # 1. Integer range
    ss_int = {"lags": (2, 24)}
    bounds, vt, vn, vtrans = convert_search_space(ss_int)
    assert vn == ["lags"]
    assert vt == ["int"]
    assert bounds == [(2, 24)]

    # 2. Factor (list of strings representing lags)
    ss_factor = {"lags": ["24", "48", "[1, 2, 24]"]}
    bounds, vt, vn, vtrans = convert_search_space(ss_factor)
    assert vn == ["lags"]
    assert vt == ["factor"]
    assert bounds == [["24", "48", "[1, 2, 24]"]]


def test_spotoptim_search_lags_numeric():
    """Smoke test for numeric lag search."""
    np.random.seed(42)
    y = pd.Series(
        np.random.randn(50).cumsum(),
        index=pd.date_range("2022-01-01", periods=50, freq="h"),
        name="load",
    )
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=5)
    cv = TimeSeriesFold(steps=2, initial_train_size=40, refit=False)

    # Search lags as an integer range
    search_space = {"lags": (2, 10), "alpha": (0.1, 1.0)}
    results, _ = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=3,
        n_initial=2,
        random_state=42,
        return_best=True,
        verbose=False,
    )

    assert "lags" in results.columns
    best_lags = results.loc[0, "lags"]
    # initialize_lags converts integer 5 to np.array([1, 2, 3, 4, 5])
    # The max value of the lags array should be within our search range [2, 10]
    max_lag = np.max(best_lags)
    assert 2 <= max_lag <= 10


def test_spotoptim_search_lags_categorical():
    """Smoke test for categorical lag search."""
    np.random.seed(42)
    y = pd.Series(
        np.random.randn(50).cumsum(),
        index=pd.date_range("2022-01-01", periods=50, freq="h"),
        name="load",
    )
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=5)
    cv = TimeSeriesFold(steps=2, initial_train_size=40, refit=False)

    # Search lags as a discrete set of configurations (must be strings in the list for factors)
    search_space = {"lags": ["2", "5", "[1, 2]"], "alpha": (0.1, 1.0)}
    results, _ = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=3,
        n_initial=2,
        random_state=42,
        return_best=True,
        verbose=False,
    )

    assert "lags" in results.columns
    best_lags = results.loc[0, "lags"]
    # Convert to list for easier comparison
    best_lags_list = list(best_lags)
    # Expected versions after initialize_lags:
    # "2" -> [1, 2]
    # "5" -> [1, 2, 3, 4, 5]
    # "[1, 2]" -> [1, 2]
    assert best_lags_list in [[1, 2], [1, 2, 3, 4, 5]]
