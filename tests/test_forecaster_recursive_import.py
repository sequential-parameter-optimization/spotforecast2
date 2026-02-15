def test_forecaster_recursive_import_direct():
    """Verify that models can be imported from preprocessing."""
    from spotforecast2_safe.manager.models import ForecasterRecursiveLGBM

    model = ForecasterRecursiveLGBM(iteration=0)
    assert model.name == "lgbm"


def test_forecaster_recursive_import_toplevel():
    """Verify that models can be imported from top-level spotforecast2_safe."""
    from spotforecast2_safe import ForecasterRecursiveLGBM, ForecasterRecursiveXGB

    model_l = ForecasterRecursiveLGBM(iteration=0)
    assert model_l.name == "lgbm"
    model_x = ForecasterRecursiveXGB(iteration=0)
    assert model_x.name == "xgb"


def test_forecaster_recursive_instantiation_params():
    """Verify that instantiation with parameters works."""
    from spotforecast2_safe import ForecasterRecursiveLGBM

    model = ForecasterRecursiveLGBM(
        iteration=1, lags=24, country_code="AT", random_state=42
    )
    assert model.iteration == 1
    assert model.forecaster.max_lag == 24
    assert model.preprocessor.country_code == "AT"
    assert model.random_state == 42
