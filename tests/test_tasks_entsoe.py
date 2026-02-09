# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pytest

from spotforecast2.manager.plotter import make_plot
from spotforecast2.tasks.task_entsoe import (
    main,
    Period,
    ExogBuilder,
    RepeatingBasisFunction,
    ForecasterRecursiveLGBM,
    ForecasterRecursiveXGB
)

class TestTaskEntsoe(unittest.TestCase):
    """Tests for the task_entsoe script."""

    def test_rbf_transformer(self):
        """Test RepeatingBasisFunction simplified implementation."""
        rbf = RepeatingBasisFunction(n_periods=12, column="hour", input_range=(1, 24))
        df = pd.DataFrame({"hour": range(1, 25)})
        
        # Test transform
        out = rbf.transform(df)
        self.assertEqual(out.shape, (24, 12))
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 1))

    def test_exog_builder(self):
        """Test ExogBuilder creates expected columns."""
        periods = [Period(name="daily", n_periods=12, column="hour", input_range=(1, 24))]
        builder = ExogBuilder(periods=periods)
        
        start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2026-01-02 00:00", tz="UTC")
        
        X = builder.build(start, end)
        
        expected_cols = ["is_weekend"]
        # Plus 12 RBF columns: daily_0 ... daily_11
        for i in range(12):
             expected_cols.append(f"daily_{i}")
        for col in expected_cols:
            self.assertIn(col, X.columns)
        
        self.assertEqual(X.shape[0], 25) # 24 hours + 1 endpoint? freq="h" includes end?
        # pd.date_range includes end by default if matches freq. 00:00 to 00:00 next day is 25 points.

    @patch("spotforecast2.tasks.task_entsoe.download_new_data")
    def test_cli_download(self, mock_download):
        """Test download subcommand."""
        test_args = ["task_entsoe.py", "download", "--api-key", "test_key", "202601010000"]
        with patch.object(sys, "argv", test_args):
            main()
            
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        self.assertEqual(call_args.kwargs["api_key"], "test_key")
        self.assertEqual(call_args.kwargs["start"], "202601010000")

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_cli_train(self, mock_handle_training):
        """Test train subcommand."""
        test_args = ["task_entsoe.py", "train", "lgbm", "--force"]
        with patch.object(sys, "argv", test_args):
            main()
            
        mock_handle_training.assert_called_once()
        self.assertEqual(mock_handle_training.call_args.kwargs["model_name"], "lgbm")
        self.assertTrue(mock_handle_training.call_args.kwargs["force"])
        # Check if model class passed is correct
        model_class = mock_handle_training.call_args.kwargs["model_class"]
        self.assertEqual(model_class, ForecasterRecursiveLGBM)

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    @patch("spotforecast2.tasks.task_entsoe.make_plot")
    def test_cli_predict(self, mock_make_plot, mock_get_pred):
        """Test predict subcommand."""
        mock_get_pred.return_value = {"some": "data"}
        
        test_args = ["task_entsoe.py", "predict", "--plot"]
        with patch.object(sys, "argv", test_args):
            main()
            
        mock_get_pred.assert_called_once()
        # Verify model_name parameter is passed (defaults to lgbm)
        call_kwargs = mock_get_pred.call_args.kwargs
        self.assertEqual(call_kwargs["model_name"], "lgbm")
        mock_make_plot.assert_called_once_with({"some": "data"})


# ==============================================================================
# Safety-Critical Pytest Suite for MLOps
# ==============================================================================

import pytest
from unittest.mock import call


class TestSafetyCriticalEntsoe:
    """
    Safety-critical tests for ENTSO-E task following MLOps best practices.
    
    These tests ensure robustness for production deployment in safety-critical
    energy forecasting systems.
    """

    # -------------------------------------------------------------------------
    # Parameter Validation Tests
    # -------------------------------------------------------------------------

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_train_lgbm_model_parameter_correctness(self, mock_handle_training):
        """
        SAFETY: Verify correct model_name parameter is passed for LightGBM.
        
        Regression test for model_type → model_name bug fix.
        Critical for model registry and persistence systems.
        """
        test_args = ["task_entsoe.py", "train", "lgbm", "--force"]
        with patch.object(sys, "argv", test_args):
            main()
        
        # Assert called exactly once with correct parameters
        assert mock_handle_training.call_count == 1
        call_kwargs = mock_handle_training.call_args.kwargs
        
        # CRITICAL: Verify model_name (not model_type) is used
        assert "model_name" in call_kwargs, "model_name parameter missing"
        assert "model_type" not in call_kwargs, "Deprecated model_type parameter found"
        assert call_kwargs["model_name"] == "lgbm"
        assert call_kwargs["force"] is True
        assert call_kwargs["model_class"] == ForecasterRecursiveLGBM

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_train_xgb_model_parameter_correctness(self, mock_handle_training):
        """
        SAFETY: Verify correct model_name parameter is passed for XGBoost.
        
        Ensures consistent parameter naming across different model types.
        """
        test_args = ["task_entsoe.py", "train", "xgb"]
        with patch.object(sys, "argv", test_args):
            main()
        
        call_kwargs = mock_handle_training.call_args.kwargs
        assert call_kwargs["model_name"] == "xgb"
        assert call_kwargs["force"] is False
        assert call_kwargs["model_class"] == ForecasterRecursiveXGB

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_train_without_force_flag_defaults_to_false(self, mock_handle_training):
        """
        SAFETY: Verify force parameter defaults to False.
        
        Critical to prevent accidental model overwriting in production.
        """
        test_args = ["task_entsoe.py", "train", "lgbm"]
        with patch.object(sys, "argv", test_args):
            main()
        
        assert mock_handle_training.call_args.kwargs["force"] is False

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    def test_train_invalid_model_type_raises_error(self):
        """
        SAFETY: Verify invalid model type raises meaningful error.
        
        Prevents deployment of unvalidated models.
        """
        test_args = ["task_entsoe.py", "train", "invalid_model"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    @patch("spotforecast2.tasks.task_entsoe.download_new_data")
    def test_download_without_api_key_uses_environment_variable(self, mock_download):
        """
        SAFETY: Verify API key fallback to environment variable.
        
        Critical for secure credential handling in production.
        """
        with patch.dict(os.environ, {"ENTSOE_API_KEY": "env_key"}):
            test_args = ["task_entsoe.py", "download"]
            with patch.object(sys, "argv", test_args):
                main()
        
        # Verify environment key was used
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["api_key"] == "env_key"

    @patch("spotforecast2.tasks.task_entsoe.download_new_data")
    def test_download_cli_api_key_overrides_environment(self, mock_download):
        """
        SAFETY: CLI API key takes precedence over environment variable.
        
        Ensures explicit configuration overrides for testing/staging.
        """
        with patch.dict(os.environ, {"ENTSOE_API_KEY": "env_key"}):
            test_args = ["task_entsoe.py", "download", "--api-key", "cli_key"]
            with patch.object(sys, "argv", test_args):
                main()
        
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["api_key"] == "cli_key"

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    def test_predict_handles_none_prediction_gracefully(self, mock_get_pred):
        """
        SAFETY: Verify graceful handling of failed predictions.
        
        Critical for system stability when data is unavailable.
        """
        mock_get_pred.return_value = None
        
        test_args = ["task_entsoe.py", "predict"]
        with patch.object(sys, "argv", test_args):
            # Should not raise, just log and exit gracefully
            main()
        
        mock_get_pred.assert_called_once()

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    @patch("spotforecast2.tasks.task_entsoe.make_plot")
    def test_predict_plot_not_called_if_prediction_fails(self, mock_make_plot, mock_get_pred):
        """
        SAFETY: Ensure plotting is skipped if prediction fails.
        
        Prevents crash on visualization of invalid data.
        """
        mock_get_pred.return_value = None
        
        test_args = ["task_entsoe.py", "predict", "--plot"]
        with patch.object(sys, "argv", test_args):
            main()
        
        mock_make_plot.assert_not_called()

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    def test_predict_default_model_is_lgbm(self, mock_get_pred):
        """
        SAFETY: Verify predict defaults to LGBM model.
        
        Ensures backward compatibility and predictable behavior.
        """
        mock_get_pred.return_value = {"predictions": [1, 2, 3]}
        
        test_args = ["task_entsoe.py", "predict"]
        with patch.object(sys, "argv", test_args):
            main()
        
        call_kwargs = mock_get_pred.call_args.kwargs
        assert call_kwargs["model_name"] == "lgbm"

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    def test_predict_explicit_lgbm_model(self, mock_get_pred):
        """
        SAFETY: Verify explicit LGBM model selection in predict.
        
        Critical for model traceability in production.
        """
        mock_get_pred.return_value = {"predictions": [1, 2, 3]}
        
        test_args = ["task_entsoe.py", "predict", "lgbm"]
        with patch.object(sys, "argv", test_args):
            main()
        
        call_kwargs = mock_get_pred.call_args.kwargs
        assert call_kwargs["model_name"] == "lgbm"

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    def test_predict_explicit_xgb_model(self, mock_get_pred):
        """
        SAFETY: Verify explicit XGB model selection in predict.
        
        Critical for model traceability in production.
        """
        mock_get_pred.return_value = {"predictions": [1, 2, 3]}
        
        test_args = ["task_entsoe.py", "predict", "xgb"]
        with patch.object(sys, "argv", test_args):
            main()
        
        call_kwargs = mock_get_pred.call_args.kwargs
        assert call_kwargs["model_name"] == "xgb"

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    @patch("spotforecast2.tasks.task_entsoe.make_plot")
    def test_predict_with_plot_passes_correct_data(self, mock_make_plot, mock_get_pred):
        """
        SAFETY: Verify plotting receives correct prediction data.
        
        Ensures data integrity through visualization pipeline.
        """
        test_data = {"predictions": [1, 2, 3], "timestamps": ["2026-01-01"]}
        mock_get_pred.return_value = test_data
        
        test_args = ["task_entsoe.py", "predict", "lgbm", "--plot"]
        with patch.object(sys, "argv", test_args):
            main()
        
        # Verify exact data passed to plotting
        mock_make_plot.assert_called_once_with(test_data)

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    @patch("spotforecast2.tasks.task_entsoe.make_plot")
    def test_predict_without_plot_flag_skips_plotting(self, mock_make_plot, mock_get_pred):
        """
        SAFETY: Verify plotting is skipped when --plot flag is absent.
        
        Ensures resource efficiency and predictable behavior.
        """
        mock_get_pred.return_value = {"predictions": [1, 2, 3]}
        
        test_args = ["task_entsoe.py", "predict", "lgbm"]
        with patch.object(sys, "argv", test_args):
            main()
        
        mock_get_pred.assert_called_once()
        mock_make_plot.assert_not_called()

    def test_plot_saved_to_output_path(self, tmp_path):
        """
        SAFETY: Verify plot generation persists an HTML file.

        Ensures plotting outputs are saved for auditability.
        """
        index = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="UTC")
        prediction_package = {
            "train_actual": pd.Series([100.0, 105.0], index=index[:2]),
            "future_actual": pd.Series([110.0, 108.0], index=index[2:]),
            "train_pred": pd.Series([101.0, 104.0], index=index[:2]),
            "future_pred": pd.Series([109.0, 107.0], index=index[2:]),
            "metrics_train": {"mae": 1.0, "mape": 0.01},
            "metrics_future_one_day": {"mae": 2.0, "mape": 0.02},
            "metrics_future": {"mae": 2.5, "mape": 0.03},
        }
        output_path = tmp_path / "prediction_plot.html"

        fig = make_plot(prediction_package, output_path=output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert fig.data

    def test_plot_saved_log_includes_expected_path(self, tmp_path, caplog):
        """
        SAFETY: Verify plot save log line includes the output path.

        Ensures audit logs are traceable to concrete artifacts.
        """
        index = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="UTC")
        prediction_package = {
            "train_actual": pd.Series([100.0, 105.0], index=index[:2]),
            "future_actual": pd.Series([110.0, 108.0], index=index[2:]),
            "train_pred": pd.Series([101.0, 104.0], index=index[:2]),
            "future_pred": pd.Series([109.0, 107.0], index=index[2:]),
            "metrics_train": {"mae": 1.0, "mape": 0.01},
            "metrics_future_one_day": {"mae": 2.0, "mape": 0.02},
            "metrics_future": {"mae": 2.5, "mape": 0.03},
        }
        output_path = tmp_path / "prediction_plot.html"

        with caplog.at_level(logging.INFO, logger="spotforecast2.manager.plotter"):
            make_plot(prediction_package, output_path=output_path)

        log_text = "\n".join([record.getMessage() for record in caplog.records])
        assert "Plot saved to" in log_text
        assert str(output_path) in log_text

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    def test_cli_predict_plot_saves_to_default_data_home(self, mock_get_pred, tmp_path):
        """
        SAFETY: Verify CLI predict --plot saves to default data home.

        Ensures output location is deterministic and auditable.
        """
        index = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="UTC")
        mock_get_pred.return_value = {
            "train_actual": pd.Series([100.0, 105.0], index=index[:2]),
            "future_actual": pd.Series([110.0, 108.0], index=index[2:]),
            "train_pred": pd.Series([101.0, 104.0], index=index[:2]),
            "future_pred": pd.Series([109.0, 107.0], index=index[2:]),
            "metrics_train": {"mae": 1.0, "mape": 0.01},
            "metrics_future_one_day": {"mae": 2.0, "mape": 0.02},
            "metrics_future": {"mae": 2.5, "mape": 0.03},
        }

        with patch("spotforecast2.manager.plotter.get_data_home", return_value=tmp_path):
            test_args = ["task_entsoe.py", "predict", "--plot"]
            with patch.object(sys, "argv", test_args):
                main()

        expected_path = tmp_path / "index.html"
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

    # -------------------------------------------------------------------------
    # Data Validation Tests
    # -------------------------------------------------------------------------

    def test_rbf_transformer_boundary_conditions(self):
        """
        SAFETY: Test RBF transformer with edge cases.
        
        Ensures numerical stability at boundaries.
        """
        rbf = RepeatingBasisFunction(n_periods=24, column="hour", input_range=(0, 23))
        
        # Test boundary values
        df_min = pd.DataFrame({"hour": [0]})
        df_max = pd.DataFrame({"hour": [23]})
        df_middle = pd.DataFrame({"hour": [12]})
        
        out_min = rbf.transform(df_min)
        out_max = rbf.transform(df_max)
        out_middle = rbf.transform(df_middle)
        
        # All values should be in [0, 1]
        assert np.all(out_min >= 0) and np.all(out_min <= 1)
        assert np.all(out_max >= 0) and np.all(out_max <= 1)
        assert np.all(out_middle >= 0) and np.all(out_middle <= 1)

    def test_rbf_transformer_handles_out_of_range_gracefully(self):
        """
        SAFETY: Test RBF with out-of-range inputs.
        
        Critical for handling unexpected data in production.
        """
        rbf = RepeatingBasisFunction(n_periods=12, column="hour", input_range=(1, 24))
        
        # Test with values outside expected range
        df = pd.DataFrame({"hour": [0, 25, 100, -5]})
        out = rbf.transform(df)
        
        # Should still produce valid output without crashing
        assert out.shape == (4, 12)
        assert np.all(np.isfinite(out)), "Output contains NaN or Inf"

    def test_exog_builder_consistency_across_calls(self):
        """
        SAFETY: Ensure ExogBuilder produces consistent results.
        
        Critical for reproducibility in model training.
        """
        periods = [Period(name="daily", n_periods=12, column="hour", input_range=(1, 24))]
        builder1 = ExogBuilder(periods=periods)
        builder2 = ExogBuilder(periods=periods)
        
        start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2026-01-01 23:00", tz="UTC")
        
        X1 = builder1.build(start, end)
        X2 = builder2.build(start, end)
        
        # Results should be identical
        pd.testing.assert_frame_equal(X1, X2)

    def test_exog_builder_timezone_awareness(self):
        """
        SAFETY: Verify timezone handling in feature engineering.
        
        Critical for international deployment and DST transitions.
        """
        periods = [Period(name="daily", n_periods=12, column="hour", input_range=(1, 24))]
        builder = ExogBuilder(periods=periods)
        
        # Test with timezone-aware timestamps
        start_utc = pd.Timestamp("2026-01-01 00:00", tz="UTC")
        end_utc = pd.Timestamp("2026-01-01 23:00", tz="UTC")
        
        X = builder.build(start_utc, end_utc)
        
        # Verify index is timezone-aware
        assert X.index.tz is not None

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    @patch("spotforecast2.tasks.task_entsoe.merge_build_manual")
    def test_merge_subcommand(self, mock_merge):
        """
        SAFETY: Test merge subcommand functionality.
        
        Ensures data consolidation works correctly.
        """
        test_args = ["task_entsoe.py", "merge"]
        with patch.object(sys, "argv", test_args):
            main()
        
        mock_merge.assert_called_once()

    @patch("spotforecast2.tasks.task_entsoe.download_new_data")
    def test_download_with_date_range(self, mock_download):
        """
        SAFETY: Test download with explicit date range.
        
        Verifies correct parameter passing for historical data retrieval.
        """
        test_args = [
            "task_entsoe.py", "download",
            "--api-key", "test_key",
            "202301010000", "202312312300"
        ]
        with patch.object(sys, "argv", test_args):
            main()
        
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["start"] == "202301010000"
        assert call_kwargs["end"] == "202312312300"

    @patch("spotforecast2.tasks.task_entsoe.download_new_data")
    def test_download_force_flag(self, mock_download):
        """
        SAFETY: Test download with force flag.
        
        Ensures data can be re-downloaded when needed.
        """
        test_args = ["task_entsoe.py", "download", "--api-key", "test_key", "--force"]
        with patch.object(sys, "argv", test_args):
            main()
        
        assert mock_download.call_args.kwargs["force"] is True

    # -------------------------------------------------------------------------
    # Model Selection Safety Tests
    # -------------------------------------------------------------------------

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_model_class_mapping_lgbm(self, mock_handle_training):
        """
        SAFETY: Verify correct model class instantiation for LGBM.
        
        Prevents model mismatch in production pipelines.
        """
        test_args = ["task_entsoe.py", "train", "lgbm"]
        with patch.object(sys, "argv", test_args):
            main()
        
        model_class = mock_handle_training.call_args.kwargs["model_class"]
        assert model_class == ForecasterRecursiveLGBM
        assert model_class != ForecasterRecursiveXGB

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_model_class_mapping_xgb(self, mock_handle_training):
        """
        SAFETY: Verify correct model class instantiation for XGB.
        
        Prevents model mismatch in production pipelines.
        """
        test_args = ["task_entsoe.py", "train", "xgb"]
        with patch.object(sys, "argv", test_args):
            main()
        
        model_class = mock_handle_training.call_args.kwargs["model_class"]
        assert model_class == ForecasterRecursiveXGB
        assert model_class != ForecasterRecursiveLGBM

    # -------------------------------------------------------------------------
    # Regression Tests for Critical Bugs
    # -------------------------------------------------------------------------

    @patch("spotforecast2.tasks.task_entsoe.handle_training_safe")
    def test_regression_model_type_vs_model_name(self, mock_handle_training):
        """
        REGRESSION TEST: Ensure model_type bug does not reoccur.
        
        Historical bug: handle_training_safe was called with model_type
        instead of model_name, causing TypeError.
        
        This test ensures the parameter name remains correct.
        """
        test_args = ["task_entsoe.py", "train", "lgbm"]
        with patch.object(sys, "argv", test_args):
            main()
        
        # Get all keyword arguments passed
        call_kwargs = mock_handle_training.call_args.kwargs
        
        # CRITICAL: These assertions protect against regression
        assert "model_name" in call_kwargs, \
            "REGRESSION: model_name parameter missing (model_type bug may have returned)"
        assert "model_type" not in call_kwargs, \
            "REGRESSION: model_type parameter found (should be model_name)"
        
        # Ensure backward incompatible change is permanent
        with pytest.raises(KeyError):
            _ = call_kwargs["model_type"]

    @patch("spotforecast2.tasks.task_entsoe.get_model_prediction_safe")
    def test_regression_predict_missing_model_name(self, mock_get_pred):
        """
        REGRESSION TEST: Ensure predict passes model_name parameter.
        
        Historical bug: get_model_prediction_safe was called without
        required model_name parameter, causing TypeError.
        
        This test ensures the parameter is always passed.
        """
        mock_get_pred.return_value = {"predictions": [1, 2, 3]}
        
        test_args = ["task_entsoe.py", "predict"]
        with patch.object(sys, "argv", test_args):
            main()
        
        # Get all keyword arguments passed
        call_kwargs = mock_get_pred.call_args.kwargs
        
        # CRITICAL: These assertions protect against regression
        assert "model_name" in call_kwargs, \
            "REGRESSION: model_name parameter missing in predict call"
        assert call_kwargs["model_name"] in ["lgbm", "xgb"], \
            f"REGRESSION: Invalid model_name '{call_kwargs['model_name']}'"
        
        # Verify it's called with arguments, not positionally
        assert len(mock_get_pred.call_args.args) == 0, \
            "REGRESSION: model_name should be keyword argument, not positional"


if __name__ == "__main__":
    unittest.main()
