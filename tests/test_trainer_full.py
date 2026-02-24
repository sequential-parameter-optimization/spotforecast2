import pytest
import pandas as pd
from unittest.mock import patch

from spotforecast2.manager.trainer_full import train_new_model


class DummyModel:
    def __init__(self, iteration, end_dev, train_size, **kwargs):
        self.iteration = iteration
        self.end_dev = end_dev
        self.train_size = train_size
        self.kwargs = kwargs
        self.tune_called = False
        self.name = kwargs.get("name", "dummymodel")

    def tune(self):
        self.tune_called = True

    def get_params(self):
        return {}


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame({"load": range(10)}, index=dates)


@patch("spotforecast2.manager.trainer_full.fetch_data")
@patch("spotforecast2.manager.trainer_full.dump")
def test_train_new_model_basic(mock_dump, mock_fetch_data, dummy_data, tmp_path):
    """Test basic model training and saving behavior."""
    mock_fetch_data.return_value = dummy_data

    model = train_new_model(
        model_class=DummyModel,
        n_iteration=5,
        model_name="test_basic",
        train_size=pd.Timedelta(days=5),
        save_to_file=True,
        model_dir=tmp_path,
        end_dev="2023-01-08",
        data_filename="dummy.csv",
        extra_param="foo",
    )

    # Validate fetch_data arguments
    mock_fetch_data.assert_called_once_with(filename="dummy.csv")

    # Validate model initialization
    assert isinstance(model, DummyModel)
    assert model.iteration == 5
    assert model.end_dev == pd.to_datetime("2023-01-08", utc=True)
    assert model.train_size == pd.Timedelta(days=5)
    assert model.kwargs["extra_param"] == "foo"

    # Validate method calls
    assert model.tune_called is True

    # Validate saving
    expected_path = tmp_path / "test_basic_forecaster_5.joblib"
    mock_dump.assert_called_once_with(model, expected_path, compress=3)


@patch("spotforecast2.manager.trainer_full.fetch_data")
def test_train_new_model_no_end_dev(mock_fetch_data, dummy_data, tmp_path):
    """Test automatic calculation of end_dev based on data."""
    mock_fetch_data.return_value = dummy_data

    model = train_new_model(
        model_class=DummyModel,
        n_iteration=1,
        save_to_file=False,
    )

    # When end_dev is None, it should be the last index minus 1 day
    expected_cutoff = dummy_data.index[-1] - pd.Timedelta(days=1)
    assert model.end_dev == expected_cutoff


@patch("spotforecast2.manager.trainer_full.fetch_data")
def test_train_new_model_empty_data(mock_fetch_data):
    """Test behavior when no data is returned."""
    mock_fetch_data.return_value = pd.DataFrame()

    model = train_new_model(
        model_class=DummyModel,
        n_iteration=1,
    )
    assert model is None


@patch("spotforecast2.manager.trainer_full.fetch_data")
@patch("spotforecast2.manager.trainer_full.dump")
def test_train_new_model_default_naming(
    mock_dump, mock_fetch_data, dummy_data, tmp_path
):
    """Test behavior when no model_name is provided."""
    mock_fetch_data.return_value = dummy_data

    # DummyModel sets self.name = "dummymodel"
    model = train_new_model(
        model_class=DummyModel,
        n_iteration=2,
        save_to_file=True,
        model_dir=tmp_path,
    )

    expected_path = tmp_path / "dummymodel_forecaster_2.joblib"
    mock_dump.assert_called_once_with(model, expected_path, compress=3)
