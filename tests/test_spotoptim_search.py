# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotoptim_search_forecaster — SpotOptim-based hyperparameter search."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import (
    TimeSeriesFold,
    spotoptim_search_forecaster,
)
from spotforecast2.model_selection.spotoptim_search import (
    _array_to_params,
    _convert_search_space,
    _parse_lags_from_string,
)
from spotoptim.hyperparameters import ParameterSet

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def y_series():
    """200-point hourly time series."""
    np.random.seed(42)
    idx = pd.date_range("2022-01-01", periods=200, freq="h")
    return pd.Series(np.random.randn(200).cumsum() + 100, index=idx, name="load")


@pytest.fixture
def forecaster():
    """Minimal ForecasterRecursive with Ridge estimator."""
    return ForecasterRecursive(estimator=Ridge(alpha=1.0), lags=5)


@pytest.fixture
def cv():
    """TimeSeriesFold with small training size."""
    return TimeSeriesFold(steps=5, initial_train_size=150, refit=False)


# ------------------------------------------------------------------
# _parse_lags_from_string
# ------------------------------------------------------------------


class TestParseLagsFromString:
    def test_integer_string(self):
        assert _parse_lags_from_string("24") == 24

    def test_list_string(self):
        assert _parse_lags_from_string("[1, 2, 3]") == [1, 2, 3]

    def test_whitespace_handling(self):
        assert _parse_lags_from_string("  12  ") == 12

    def test_list_with_whitespace(self):
        assert _parse_lags_from_string("  [4, 5, 6]  ") == [4, 5, 6]


# ------------------------------------------------------------------
# _convert_search_space
# ------------------------------------------------------------------


class TestConvertSearchSpace:
    def test_dict_float_bounds(self):
        bounds, vt, vn, vtrans = _convert_search_space({"alpha": (0.01, 10.0)})
        assert vn == ["alpha"]
        assert vt == ["float"]
        assert bounds == [(0.01, 10.0)]

    def test_dict_int_bounds(self):
        bounds, vt, vn, _ = _convert_search_space({"max_depth": (2, 8)})
        assert vt == ["int"]

    def test_dict_factor(self):
        bounds, vt, vn, _ = _convert_search_space(
            {"solver": ["svd", "cholesky", "lsqr"]}
        )
        assert vt == ["factor"]
        assert bounds == [["svd", "cholesky", "lsqr"]]

    def test_dict_raw_spotoptim_format(self):
        raw = {
            "bounds": [(1, 10)],
            "var_type": ["int"],
            "var_name": ["n"],
            "var_trans": [None],
        }
        bounds, vt, vn, vtrans = _convert_search_space(raw)
        assert vn == ["n"]

    def test_parameter_set(self):
        ps = ParameterSet()
        ps.add_float("lr", low=0.001, high=0.1)
        ps.add_int("depth", low=2, high=10)
        bounds, vt, vn, _ = _convert_search_space(ps)
        assert vn == ["lr", "depth"]
        assert len(bounds) == 2

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be ParameterSet or dict"):
            _convert_search_space("not_valid")

    def test_invalid_dict_value_raises(self):
        with pytest.raises(ValueError, match="Invalid search space"):
            _convert_search_space({"bad": 42})


# ------------------------------------------------------------------
# _array_to_params
# ------------------------------------------------------------------


class TestArrayToParams:
    def test_int_and_float(self):
        result = _array_to_params(
            np.array([100.0, 0.05]),
            var_name=["n_estimators", "lr"],
            var_type=["int", "float"],
            bounds=[(50, 200), (0.01, 0.3)],
        )
        assert result == {"n_estimators": 100, "lr": 0.05}

    def test_factor_by_index(self):
        result = _array_to_params(
            np.array([1.0]),
            var_name=["solver"],
            var_type=["factor"],
            bounds=[["svd", "cholesky", "lsqr"]],
        )
        assert result["solver"] == "cholesky"

    def test_factor_by_name(self):
        result = _array_to_params(
            np.array(["svd"]),
            var_name=["solver"],
            var_type=["factor"],
            bounds=[["svd", "cholesky"]],
        )
        assert result["solver"] == "svd"


# ------------------------------------------------------------------
# spotoptim_search_forecaster — integration
# ------------------------------------------------------------------


class TestSpotoptimSearchForecaster:
    def test_dict_search_space(self, y_series, forecaster, cv):
        """Run with a simple dict-based search space."""
        results, optimizer = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_series,
            cv=cv,
            search_space={"alpha": (0.01, 10.0)},
            metric="mean_absolute_error",
            n_trials=5,
            n_initial=3,
            random_state=42,
            return_best=False,
            verbose=False,
            show_progress=False,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5
        assert "alpha" in results.columns
        assert "mean_absolute_error" in results.columns
        assert "lags" in results.columns
        assert "params" in results.columns

    def test_parameter_set_search_space(self, y_series, forecaster, cv):
        """Run with a ParameterSet-based search space."""
        ps = ParameterSet()
        ps.add_float("alpha", low=0.01, high=10.0)

        results, optimizer = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_series,
            cv=cv,
            search_space=ps,
            metric="mean_absolute_error",
            n_trials=5,
            n_initial=3,
            return_best=False,
            verbose=False,
            show_progress=False,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5

    def test_results_sorted_ascending(self, y_series, forecaster, cv):
        """Results should be sorted by the first metric (ascending for regression)."""
        results, _ = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_series,
            cv=cv,
            search_space={"alpha": (0.01, 10.0)},
            metric="mean_absolute_error",
            n_trials=5,
            n_initial=3,
            return_best=False,
            verbose=False,
            show_progress=False,
        )
        mae = results["mean_absolute_error"].values
        assert (mae[:-1] <= mae[1:]).all()

    def test_return_best_refits(self, y_series, cv, capsys):
        """When return_best=True the forecaster should be refit."""
        forecaster = ForecasterRecursive(estimator=Ridge(alpha=1.0), lags=5)
        results, _ = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_series,
            cv=cv,
            search_space={"alpha": (0.01, 10.0)},
            metric="mean_absolute_error",
            n_trials=5,
            n_initial=3,
            return_best=True,
            verbose=False,
            show_progress=False,
        )
        captured = capsys.readouterr()
        assert "refitted" in captured.out.lower()
        # Forecaster should be fitted and able to predict
        preds = forecaster.predict(steps=3)
        assert len(preds) == 3

    def test_multiple_metrics(self, y_series, forecaster, cv):
        """Support for multiple metrics."""
        results, _ = spotoptim_search_forecaster(
            forecaster=forecaster,
            y=y_series,
            cv=cv,
            search_space={"alpha": (0.01, 10.0)},
            metric=["mean_absolute_error", "mean_squared_error"],
            n_trials=5,
            n_initial=3,
            return_best=False,
            verbose=False,
            show_progress=False,
        )
        assert "mean_absolute_error" in results.columns
        assert "mean_squared_error" in results.columns

    def test_output_file(self, y_series, forecaster, cv):
        """Results should be saved to file when output_file is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "results.tsv"
            spotoptim_search_forecaster(
                forecaster=forecaster,
                y=y_series,
                cv=cv,
                search_space={"alpha": (0.01, 10.0)},
                metric="mean_absolute_error",
                n_trials=5,
                n_initial=3,
                return_best=False,
                verbose=False,
                show_progress=False,
                output_file=str(outpath),
            )
            assert outpath.exists()
            saved = pd.read_csv(outpath, sep="\t")
            assert len(saved) == 5

    def test_exog_length_mismatch_raises(self, y_series, forecaster, cv):
        """ValueError if exog length != y length and return_best=True."""
        bad_exog = pd.Series(np.ones(10))
        with pytest.raises(ValueError, match="same number of samples"):
            spotoptim_search_forecaster(
                forecaster=forecaster,
                y=y_series,
                cv=cv,
                search_space={"alpha": (0.01, 5.0)},
                metric="mean_absolute_error",
                return_best=True,
                n_trials=3,
                n_initial=2,
                exog=bad_exog,
            )

    def test_invalid_cv_raises(self, y_series, forecaster):
        """TypeError for non-TimeSeriesFold / OneStepAheadFold cv."""
        with pytest.raises(TypeError, match="TimeSeriesFold"):
            spotoptim_search_forecaster(
                forecaster=forecaster,
                y=y_series,
                cv="invalid",
                search_space={"alpha": (0.01, 5.0)},
                metric="mean_absolute_error",
                n_trials=3,
                n_initial=2,
            )


# ------------------------------------------------------------------
# Parity with bayesian_search_forecaster
# ------------------------------------------------------------------


class TestParityWithBayesian:
    """Ensure the interface matches bayesian_search_forecaster."""

    def test_same_return_type(self, y_series, forecaster, cv):
        """Both should return (DataFrame, object)."""
        from spotforecast2.model_selection import bayesian_search_forecaster

        results_spot, opt = spotoptim_search_forecaster(
            forecaster=ForecasterRecursive(estimator=Ridge(), lags=5),
            y=y_series,
            cv=cv,
            search_space={"alpha": (0.01, 10.0)},
            metric="mean_absolute_error",
            n_trials=5,
            n_initial=3,
            return_best=False,
            verbose=False,
            show_progress=False,
        )

        def optuna_search_space(trial):
            return {"alpha": trial.suggest_float("alpha", 0.01, 10.0)}

        results_bay, trial = bayesian_search_forecaster(
            forecaster=ForecasterRecursive(estimator=Ridge(), lags=5),
            y=y_series,
            cv=cv,
            search_space=optuna_search_space,
            metric="mean_absolute_error",
            n_trials=5,
            return_best=False,
            verbose=False,
            show_progress=False,
        )

        # Both should be DataFrames with the same columns
        assert set(results_spot.columns) == set(results_bay.columns)
        assert isinstance(results_spot, pd.DataFrame)
        assert isinstance(results_bay, pd.DataFrame)


# ------------------------------------------------------------------
# Docstring examples
# ------------------------------------------------------------------


class TestDocstringExamples:
    """Run all docstring examples via doctest."""

    def test_doctest(self):
        import doctest
        from spotforecast2.model_selection import spotoptim_search

        results = doctest.testmod(
            spotoptim_search, verbose=False, optionflags=doctest.ELLIPSIS
        )
        assert results.failed == 0, f"{results.failed} doctest(s) failed"
