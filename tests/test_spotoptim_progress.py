import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import (
    TimeSeriesFold,
    spotoptim_search_forecaster,
)


def test_spotoptim_search_show_progress_arg():
    """
    Test that spotoptim_search_forecaster accepts and handles show_progress argument.
    """
    np.random.seed(42)
    y = pd.Series(
        np.random.randn(50).cumsum(),
        index=pd.date_range("2022-01-01", periods=50, freq="h"),
        name="load",
    )
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=5)
    cv = TimeSeriesFold(
        steps=2,
        initial_train_size=40,
        refit=False,
    )
    search_space = {"alpha": (0.1, 1.0)}

    # Test with show_progress=True
    results, _ = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=2,
        n_initial=1,
        random_state=42,
        return_best=False,
        verbose=False,
        show_progress=True,
    )
    assert isinstance(results, pd.DataFrame)

    # Test with show_progress=False
    results, _ = spotoptim_search_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        search_space=search_space,
        metric="mean_absolute_error",
        n_trials=2,
        n_initial=1,
        random_state=42,
        return_best=False,
        verbose=False,
        show_progress=False,
    )
    assert isinstance(results, pd.DataFrame)
