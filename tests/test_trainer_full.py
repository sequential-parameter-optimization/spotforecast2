# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import pandas as pd
from unittest.mock import patch

from spotforecast2.manager.trainer_full import train_new_model, handle_training


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


# ---------------------------------------------------------------------------
# Fixtures and helpers for handle_training tests
# ---------------------------------------------------------------------------


class StubForecaster:
    """Minimal model stub: no real ML dependencies required."""

    name = "stub"

    def __init__(self, iteration, end_dev, train_size=None, **kw):
        self.iteration = iteration
        self.end_dev = end_dev
        self.train_size = train_size

    def tune(self):
        pass

    def get_params(self):
        return {}


def _recent_model(hours_ago: int = 24) -> StubForecaster:
    return StubForecaster(
        0, pd.Timestamp.now("UTC") - pd.Timedelta(hours=hours_ago)
    )


def _stale_model(days_ago: int = 10) -> StubForecaster:
    return StubForecaster(
        2, pd.Timestamp.now("UTC") - pd.Timedelta(days=days_ago)
    )


# ---------------------------------------------------------------------------
# handle_training tests
# ---------------------------------------------------------------------------


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_no_model_trains_iteration_0(mock_get, mock_train, tmp_path):
    """Empty cache → triggers train_new_model at iteration 0."""
    mock_get.return_value = (-1, None)

    handle_training(StubForecaster, model_name="stub", model_dir=tmp_path)

    mock_train.assert_called_once()
    assert mock_train.call_args[0][1] == 0  # n_iteration positional arg


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_recent_model_skips(mock_get, mock_train, tmp_path):
    """Model trained 24 h ago (< 168 h threshold) → no retraining."""
    mock_get.return_value = (1, _recent_model(hours_ago=24))

    handle_training(StubForecaster, model_name="stub", model_dir=tmp_path)

    mock_train.assert_not_called()


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_old_model_retrains(mock_get, mock_train, tmp_path):
    """Model trained 10 days ago (> 168 h) → retrains at n_iteration + 1."""
    mock_get.return_value = (2, _stale_model(days_ago=10))

    handle_training(StubForecaster, model_name="stub", model_dir=tmp_path)

    mock_train.assert_called_once()
    assert mock_train.call_args[0][1] == 3  # iteration 2 + 1


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_force_retrains_recent(mock_get, mock_train, tmp_path):
    """force=True with a recent model still triggers retraining."""
    mock_get.return_value = (0, _recent_model(hours_ago=6))

    handle_training(
        StubForecaster, model_name="stub", model_dir=tmp_path, force=True
    )

    mock_train.assert_called_once()
    assert mock_train.call_args[0][1] == 1  # iteration 0 + 1


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_missing_end_dev_forces_retrain(mock_get, mock_train, tmp_path):
    """Model without end_dev attribute → warns and forces retraining."""

    class ModelWithoutEndDev:
        name = "nodev"

    mock_get.return_value = (3, ModelWithoutEndDev())

    handle_training(StubForecaster, model_name="stub", model_dir=tmp_path)

    mock_train.assert_called_once()
    assert mock_train.call_args[0][1] == 4  # iteration 3 + 1


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_default_name_from_class(mock_get, mock_train, tmp_path):
    """model_name=None → infers lower-cased class name."""
    mock_get.return_value = (-1, None)

    handle_training(StubForecaster, model_dir=tmp_path)

    mock_get.assert_called_once_with("stubforecaster", tmp_path)
    assert mock_train.call_args[1]["model_name"] == "stubforecaster"


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_passes_kwargs(mock_get, mock_train, tmp_path):
    """Extra kwargs are forwarded to train_new_model."""
    mock_get.return_value = (-1, None)

    handle_training(
        StubForecaster,
        model_name="stub",
        model_dir=tmp_path,
        custom_param=42,
        another="hello",
    )

    _, kwargs = mock_train.call_args
    assert kwargs["custom_param"] == 42
    assert kwargs["another"] == "hello"


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_passes_end_dev_and_train_size(mock_get, mock_train, tmp_path):
    """end_dev and train_size are forwarded to train_new_model."""
    mock_get.return_value = (-1, None)
    end = "2025-01-01 00:00+00:00"
    size = pd.Timedelta(days=365)

    handle_training(
        StubForecaster,
        model_name="stub",
        model_dir=tmp_path,
        end_dev=end,
        train_size=size,
    )

    _, kwargs = mock_train.call_args
    assert kwargs["end_dev"] == end
    assert kwargs["train_size"] == size


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_at_exactly_168_hours(mock_get, mock_train, tmp_path):
    """Model trained exactly 168 h ago meets the >= threshold → retrains."""
    mock_get.return_value = (1, _recent_model(hours_ago=168))

    handle_training(StubForecaster, model_name="stub", model_dir=tmp_path)

    mock_train.assert_called_once()


@patch("spotforecast2.manager.trainer_full.train_new_model")
@patch("spotforecast2.manager.trainer_full.get_last_model")
def test_handle_training_naive_end_dev_localised(mock_get, mock_train, tmp_path):
    """Model with timezone-naive end_dev is handled without error."""
    naive_model = StubForecaster(0, pd.Timestamp.now() - pd.Timedelta(hours=24))
    naive_model.end_dev = pd.Timestamp.now() - pd.Timedelta(hours=24)  # naive
    mock_get.return_value = (0, naive_model)

    # Should not raise; naive timestamp is localised internally
    handle_training(StubForecaster, model_name="stub", model_dir=tmp_path)
