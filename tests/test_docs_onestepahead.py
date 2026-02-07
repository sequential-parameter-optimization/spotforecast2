"""
Test OneStepAheadFold documentation examples.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2.model_selection import backtesting_forecaster, OneStepAheadFold
import time


def test_rapid_model_screening():
    """Test rapid model screening example."""
    # Simulate critical infrastructure monitoring (water pressure)
    rng = np.random.default_rng(321)
    dates = pd.date_range("2024-01-01", periods=365 * 24, freq="h")

    hour_of_day = dates.hour
    daily_cycle = 5 * np.sin(2 * np.pi * hour_of_day / 24)
    baseline_pressure = 50
    degradation = -0.01 * np.arange(len(dates))
    noise = rng.normal(0, 1, len(dates))

    y = pd.Series(
        baseline_pressure + daily_cycle + degradation + noise,
        index=dates,
        name="pressure_psi",
    )

    # Define candidate models for rapid screening
    model_candidates = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=321),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=50, random_state=321),
    }

    cv = OneStepAheadFold(
        initial_train_size=180 * 24,
        verbose=False,
    )

    # Inspect fold structure
    folds_df = cv.split(y, as_pandas=True)
    assert len(folds_df) == 1, "OneStepAheadFold should create exactly one fold"
    assert "train_start" in folds_df.columns
    assert "test_end" in folds_df.columns

    # Rapid screening
    results = {}
    for name, estimator in model_candidates.items():
        forecaster = ForecasterRecursive(
            estimator=estimator,
            lags=24 * 7,
        )

        metric_values, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=y,
            cv=cv,
            metric="mean_absolute_error",
            verbose=False,
            show_progress=False,
        )

        mae = metric_values["mean_absolute_error"].iloc[0]
        results[name] = mae

    # Verify all models were evaluated
    assert len(results) == len(model_candidates)
    assert all(mae > 0 for mae in results.values())

    best_model = min(results, key=results.get)
    print(f"✓ Rapid screening example test passed")
    print(f"  Best model: {best_model} (MAE: {results[best_model]:.3f} PSI)")
    print(f"  Models screened: {len(results)}")


def test_static_deployment_validation():
    """Test static model deployment validation example."""
    # Simulate embedded sensor system
    rng = np.random.default_rng(654)
    dates = pd.date_range("2024-01-01", periods=730, freq="D")

    day_of_year = np.arange(len(dates)) % 365
    seasonal = 15 * np.sin(2 * np.pi * day_of_year / 365)
    baseline_temp = 20
    noise = rng.normal(0, 2, len(dates))

    y = pd.Series(baseline_temp + seasonal + noise, index=dates, name="temperature_c")

    forecaster = ForecasterRecursive(
        estimator=GradientBoostingRegressor(n_estimators=100, random_state=654),
        lags=30,
    )

    cv = OneStepAheadFold(
        initial_train_size=365,
        verbose=False,
    )

    metric_values, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric=["mean_absolute_error", "mean_squared_error"],
        interval=0.90,
        interval_method="conformal",
        use_in_sample_residuals=True,
        verbose=False,
        show_progress=False,
    )

    # Verify results
    assert isinstance(metric_values, pd.DataFrame)
    assert "mean_absolute_error" in metric_values.columns
    assert "mean_squared_error" in metric_values.columns
    assert len(predictions) == 365  # Second year

    # Temporal degradation analysis
    n_test = len(predictions)
    quarter_size = n_test // 4
    quarterly_mae = []

    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else n_test
        quarter_preds = predictions.iloc[start_idx:end_idx]
        quarter_actual = y.loc[quarter_preds.index]
        quarter_mae = (quarter_actual - quarter_preds["pred"]).abs().mean()
        quarterly_mae.append(quarter_mae)

    mae_trend = np.polyfit(range(4), quarterly_mae, 1)[0]

    print(f"✓ Static deployment example test passed")
    print(f"  MAE: {metric_values['mean_absolute_error'].iloc[0]:.2f} °C")
    print(f"  Performance trend: {mae_trend:+.3f} °C/quarter")


def test_emergency_validation():
    """Test emergency production validation example."""
    # Simulate production scenario
    rng = np.random.default_rng(987)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")

    baseline = 100
    trend = 0.1 * np.arange(len(dates))
    shift = np.where(np.arange(len(dates)) > 60, 10, 0)
    noise = rng.normal(0, 5, len(dates))

    y = pd.Series(baseline + trend + shift + noise, index=dates, name="metric")

    rollback_model = ForecasterRecursive(
        estimator=RandomForestRegressor(n_estimators=30, random_state=987),
        lags=7,
    )

    cv = OneStepAheadFold(
        initial_train_size=60,
        verbose=False,
    )

    # Time the validation
    start_time = time.time()

    metric_values, predictions = backtesting_forecaster(
        forecaster=rollback_model,
        y=y,
        cv=cv,
        metric="mean_absolute_error",
        verbose=False,
        show_progress=False,
    )

    validation_time = time.time() - start_time

    # Verify results
    assert isinstance(metric_values, pd.DataFrame)
    assert len(predictions) == 30  # Last 30 days
    assert validation_time < 10.0, "Emergency validation should be fast"

    rollback_mae = metric_values["mean_absolute_error"].iloc[0]

    print(f"✓ Emergency validation example test passed")
    print(f"  Validation time: {validation_time:.2f} seconds")
    print(f"  Rollback MAE: {rollback_mae:.2f}")


if __name__ == "__main__":
    test_rapid_model_screening()
    test_static_deployment_validation()
    test_emergency_validation()
    print("\n✓ All OneStepAheadFold documentation examples passed")
