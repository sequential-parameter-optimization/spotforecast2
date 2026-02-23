import pandas as pd
import numpy as np
from unittest.mock import patch
from spotforecast2.tasks import task_n_to_1_dataframe, task_demo

def mock_fetch_data(*args, **kwargs):
    dates = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    data = pd.DataFrame(
        np.random.rand(200, 11),
        index=dates,
        columns=[f"col{i}" for i in range(11)]
    )
    return data

@patch("spotforecast2.tasks.task_n_to_1_dataframe.fetch_data", side_effect=mock_fetch_data)
def test_task_n_to_1_dataframe_execution(mock_fetch):
    """
    Smoke test to ensure task_n_to_1_dataframe.main runs without crashing.
    This also implicitly tests n2n_predict from spotforecast2_safe.
    """
    # Overwrite the default config simply by patching or relying on defaults
    # Since it fetches data directly, our mock will supply the 200 rows.
    task_n_to_1_dataframe.main()

@patch("spotforecast2.tasks.task_demo.n2n_predict")
@patch("spotforecast2.tasks.task_demo.n2n_predict_with_covariates")
@patch("spotforecast2.tasks.task_demo.load_actual_combined")
@patch("spotforecast2.tasks.task_demo.plot_actual_vs_predicted")
def test_task_demo_execution(mock_plot, mock_load, mock_n2n_cov, mock_n2n):
    """
    Smoke test for task_demo.py to ensure it wires up the components correctly.
    We mock the heavy prediction functions since they are individually tested elsewhere,
    and we just want to ensure the task script itself doesn't crash on orchestration.
    """
    dates = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
    mock_df = pd.DataFrame(np.random.rand(24, 11), index=dates)
    mock_series = pd.Series(np.random.rand(24), index=dates)
    
    mock_n2n.return_value = (mock_df, {})
    mock_n2n_cov.return_value = (mock_df, {}, {})
    mock_load.return_value = mock_series
    
    task_demo.main(force_train=False)
    
    assert mock_n2n.called
    assert mock_n2n_cov.called
    assert mock_plot.called
