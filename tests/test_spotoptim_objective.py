import numpy as np
import pandas as pd
from unittest.mock import patch

from spotforecast2.model_selection.spotoptim_search import spotoptim_objective
from spotforecast2_safe.model_selection import TimeSeriesFold, OneStepAheadFold


class MockForecaster:
    def __init__(self):
        self.params = {}
        self.lags = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def set_lags(self, lags):
        self.lags = lags

    def _train_test_split_one_step_ahead(self, y, initial_train_size, exog=None):
        return [0], [0], [0], [0]


def dummy_metric(y_true, y_pred):
    return 0.42


def test_spotoptim_objective_timeseriesfold():
    forecaster = MockForecaster()
    X = np.array([[2.0, 5.0]])
    cv_name = "TimeSeriesFold"
    cv = TimeSeriesFold(initial_train_size=10, steps=2)
    metric = [dummy_metric]
    y = pd.Series(np.arange(20))
    exog = None
    n_jobs = 1
    verbose = False
    show_progress = False
    suppress_warnings = True
    var_name = ["alpha", "max_depth"]
    var_type = ["float", "int"]
    bounds = [(0.1, 10.0), (2, 8)]

    all_metric_values = []
    all_lags = []
    all_params = []

    with patch(
        "spotforecast2.model_selection.spotoptim_search._backtesting_forecaster"
    ) as mock_backtesting:
        # Mock returns: (metrics_df, dict_predictions)
        mock_backtesting.return_value = (
            pd.DataFrame({dummy_metric.__name__: [0.42]}),
            None,
        )

        result = spotoptim_objective(
            X=X,
            forecaster_search=forecaster,
            cv_name=cv_name,
            cv=cv,
            metric=metric,
            y=y,
            exog=exog,
            n_jobs=n_jobs,
            verbose=verbose,
            show_progress=show_progress,
            suppress_warnings=suppress_warnings,
            var_name=var_name,
            var_type=var_type,
            bounds=bounds,
            all_metric_values=all_metric_values,
            all_lags=all_lags,
            all_params=all_params,
        )

        assert mock_backtesting.called
        assert len(result) == 1
        assert result[0] == 0.42
        assert len(all_metric_values) == 1
        assert all_metric_values[0] == [0.42]
        assert len(all_params) == 1
        assert all_params[0] == {"alpha": 2.0, "max_depth": 5}


def test_spotoptim_objective_onestepahead():
    forecaster = MockForecaster()
    X = np.array([[0.1, 3.0]])
    cv_name = "OneStepAheadFold"
    cv = OneStepAheadFold(initial_train_size=10)
    metric = [dummy_metric]
    y = pd.Series(np.arange(20))
    exog = None
    n_jobs = 1
    verbose = False
    show_progress = False
    suppress_warnings = True
    var_name = ["lr", "n_estimators"]
    var_type = ["float", "int"]
    bounds = [(0.01, 1.0), (2, 10)]

    all_metric_values = []
    all_lags = []
    all_params = []

    with patch(
        "spotforecast2.model_selection.spotoptim_search._calculate_metrics_one_step_ahead"
    ) as mock_calc:
        mock_calc.return_value = [0.99]

        result = spotoptim_objective(
            X=X,
            forecaster_search=forecaster,
            cv_name=cv_name,
            cv=cv,
            metric=metric,
            y=y,
            exog=exog,
            n_jobs=n_jobs,
            verbose=verbose,
            show_progress=show_progress,
            suppress_warnings=suppress_warnings,
            var_name=var_name,
            var_type=var_type,
            bounds=bounds,
            all_metric_values=all_metric_values,
            all_lags=all_lags,
            all_params=all_params,
        )

        assert mock_calc.called
        assert len(result) == 1
        assert result[0] == 0.99
        assert len(all_metric_values) == 1
        assert all_metric_values[0] == [0.99]
        assert len(all_params) == 1
        assert all_params[0] == {"lr": 0.1, "n_estimators": 3}
