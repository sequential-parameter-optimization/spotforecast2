from unittest.mock import patch
import pandas as pd
from spotforecast2.manager.trainer_full import train_new_model


def test_model_training_intro_simple():
    """Validates the Simple Training Example from model_training_intro.qmd."""

    # 1. Define a Mock Model Class meeting the API requirements
    class MockForecaster:
        def __init__(self, iteration, end_dev, train_size, **kwargs):
            self.iteration = iteration
            self.end_dev = end_dev
            self.train_size = train_size
            self.config = kwargs

        def tune(self):
            # In actual usage, this acts as the gateway to spotoptim_search
            print(f"Executing tune() for iteration {self.iteration}")
            print(f"Focus window cuts off at: {self.end_dev}")

        def get_params(self):
            return {"stub": "mock"}

    # Mock fetch_data to prevent actual file I/O operations
    dummy_data = pd.DataFrame(
        {"load": [1, 2, 3]}, index=pd.date_range("2022-12-01", periods=3, freq="D")
    )
    with patch(
        "spotforecast2.manager.trainer_full.fetch_data", return_value=dummy_data
    ):
        from spotforecast2_safe.data.fetch_data import get_package_data_home
        demo_file = get_package_data_home() / "demo01.csv"

        # 2. Start a basic training run explicitly overriding the cutoff
        model_basic = train_new_model(
            model_class=MockForecaster,
            n_iteration=1,
            model_name="baseline_mock",
            end_dev="2023-01-01 00:00+00:00",
            train_size=None,  # Use the entire history
            save_to_file=False,
            data_filename=str(demo_file)
        )

        assert isinstance(model_basic, MockForecaster)
        assert model_basic.end_dev == pd.to_datetime("2023-01-01 00:00+00:00", utc=True)
        assert model_basic.train_size is None
        assert model_basic.iteration == 1


def test_model_training_intro_advanced():
    """Validates the Advanced Training Example from model_training_intro.qmd."""

    class MockForecaster:
        def __init__(self, iteration, end_dev, train_size, **kwargs):
            self.iteration = iteration
            self.end_dev = end_dev
            self.train_size = train_size
            self.config = kwargs

        def tune(self):
            pass

        def get_params(self):
            return {}

    # Mock fetch_data to prevent actual file I/O operations
    dummy_data = pd.DataFrame(
        {"load": [1, 2, 3]}, index=pd.date_range("2023-12-01", periods=3, freq="D")
    )
    with patch(
        "spotforecast2.manager.trainer_full.fetch_data", return_value=dummy_data
    ):
        from spotforecast2_safe.data.fetch_data import get_package_data_home
        demo_file = get_package_data_home() / "demo01.csv"

        # 1. Start an advanced tuning workflow
        model_advanced = train_new_model(
            model_class=MockForecaster,
            n_iteration=3,
            model_name="production_mock",
            train_size=pd.Timedelta(days=365),  # Force exactly 1 year backward logic
            end_dev="2024-03-15 00:00+00:00",
            save_to_file=False,
            data_filename=str(demo_file),
            # Inject specific kwargs dynamically
            lags=48,
            advanced_regularization=True,
            surrogate_seed=1214,
        )

        assert isinstance(model_advanced, MockForecaster)
        assert model_advanced.train_size == pd.Timedelta(days=365)
        assert model_advanced.end_dev == pd.to_datetime(
            "2024-03-15 00:00+00:00", utc=True
        )
        assert model_advanced.config.get("lags") == 48
        assert model_advanced.config.get("advanced_regularization") is True
        assert model_advanced.config.get("surrogate_seed") == 1214


def test_model_training_intro_functional():
    """Validates the Fully Functional End-to-End Example from model_training_intro.qmd."""
    from sklearn.linear_model import Ridge
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home

    class FunctionalForecaster:
        def __init__(self, iteration, end_dev, train_size, dataset_path=None, **kwargs):
            self.iteration = iteration
            self.end_dev = end_dev
            self.train_size = train_size
            self.dataset_path = dataset_path

            self.forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            self.name = "demo01_model"

        def tune(self):
            df = fetch_data(filename=self.dataset_path)
            y = df["Actual Load"]

            if self.train_size is not None:
                start_date = self.end_dev - self.train_size
                y_train = y.loc[start_date : self.end_dev]
            else:
                y_train = y.loc[: self.end_dev]

            print(f"Fitting model strictly on data until {self.end_dev}")
            print(f"Training window length: {len(y_train)} hours")
            self.forecaster.fit(y=y_train)

        def get_params(self):
            return {}

    demo_file = get_package_data_home() / "demo01.csv"

    model_functional = train_new_model(
        model_class=FunctionalForecaster,
        n_iteration=1,
        train_size=pd.Timedelta(days=7),
        end_dev=None,
        data_filename=str(demo_file),
        save_to_file=False,
        dataset_path=str(demo_file),
    )

    assert model_functional.forecaster.is_fitted is True


def test_model_training_intro_visualization():
    """Validates the Visualizing Prediction Quality Example from model_training_intro.qmd."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
    import plotly.graph_objects as go

    class VisualizingForecaster:
        def __init__(self, iteration, end_dev, train_size, dataset_path=None, **kwargs):
            self.iteration = iteration
            self.end_dev = end_dev
            self.train_size = train_size
            self.dataset_path = dataset_path

            self.forecaster = ForecasterRecursive(estimator=Ridge(), lags=24)
            self.name = "demo02_model"

        def tune(self):
            df = fetch_data(filename=self.dataset_path)
            y = (
                df["A"].groupby(level=0).mean().asfreq("h").ffill()
            )  # safely manage duplicates and gaps
            y_train = y.loc[: self.end_dev]

            if self.train_size is not None:
                start_date = pd.to_datetime(self.end_dev, utc=True) - self.train_size
                y_train = y_train.loc[start_date:]

            self.forecaster.fit(y=y_train)

        def get_params(self):
            return {}

    demo_file = get_package_data_home() / "demo02.csv"
    df_full = fetch_data(filename=str(demo_file))
    y_full = df_full["A"].groupby(level=0).mean().asfreq("h").ffill()

    test_duration = pd.Timedelta(days=7)
    cutoff_date = y_full.index.max() - test_duration

    model_vis = train_new_model(
        model_class=VisualizingForecaster,
        n_iteration=1,
        train_size=pd.Timedelta(days=60),
        end_dev=cutoff_date,
        data_filename=str(demo_file),
        save_to_file=False,
        dataset_path=str(demo_file),
    )

    y_test = y_full.loc[cutoff_date + pd.Timedelta(hours=1) :]
    preds = model_vis.forecaster.predict(steps=len(y_test))
    preds.index = y_test.index

    mae = mean_absolute_error(y_test, preds)

    assert mae > 0.0
    assert len(preds) == len(y_test)
    assert model_vis.forecaster.is_fitted is True

    # Check that plotly layout is instantiated properly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Actual Truth")
    )
    fig.add_trace(
        go.Scatter(x=preds.index, y=preds, mode="lines", name="Forecaster Projection")
    )

    assert len(fig.data) == 2


def test_model_training_intro_visualization_lgbm():
    """Validates the LightGBM Visualizing Prediction Example from model_training_intro.qmd."""
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error
    from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
    from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
    import plotly.graph_objects as go

    class LGBMVisualizingForecaster:
        def __init__(self, iteration, end_dev, train_size, dataset_path=None, **kwargs):
            self.iteration = iteration
            self.end_dev = end_dev
            self.train_size = train_size
            self.dataset_path = dataset_path
            
            self.forecaster = ForecasterRecursive(
                estimator=LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1), 
                lags=24
            )
            self.name = "demo02_lgbm_model"

        def tune(self):
            df = fetch_data(filename=self.dataset_path)
            y = (
                df["A"].groupby(level=0).mean().asfreq("h").ffill()
            ) 
            y_train = y.loc[: self.end_dev]

            if self.train_size is not None:
                start_date = pd.to_datetime(self.end_dev, utc=True) - self.train_size
                y_train = y_train.loc[start_date:]

            self.forecaster.fit(y=y_train)

        def get_params(self):
            return {}

    demo_file = get_package_data_home() / "demo02.csv"
    df_full = fetch_data(filename=str(demo_file))
    y_full = df_full["A"].groupby(level=0).mean().asfreq("h").ffill()

    test_duration = pd.Timedelta(days=7)
    cutoff_date = y_full.index.max() - test_duration

    model_lgbm = train_new_model(
        model_class=LGBMVisualizingForecaster,
        n_iteration=1,
        train_size=pd.Timedelta(days=60),
        end_dev=cutoff_date,
        data_filename=str(demo_file),
        save_to_file=False,
        dataset_path=str(demo_file),
    )

    y_test = y_full.loc[cutoff_date + pd.Timedelta(hours=1) :]
    preds = model_lgbm.forecaster.predict(steps=len(y_test))
    preds.index = y_test.index

    mae = mean_absolute_error(y_test, preds)

    assert mae > 0.0
    assert len(preds) == len(y_test)
    assert model_lgbm.forecaster.is_fitted is True

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Actual Truth")
    )
    fig.add_trace(
        go.Scatter(x=preds.index, y=preds, mode="lines", name="LGBM Projection")
    )

    assert len(fig.data) == 2
