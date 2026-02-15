# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unified CLI task script for ENTSO-E data downloading, model training, and prediction.

This script acts as the main entry point for the forecasting pipeline, integrating
downloading, training, and plotting functionalities.

Usage:
    uv run python src/spotforecast2/tasks/task_entsoe.py download --api-key <KEY>
    uv run python src/spotforecast2/tasks/task_entsoe.py train lgbm
    uv run python src/spotforecast2/tasks/task_entsoe.py predict --plot

Environment Variables:
    ENTSOE_API_KEY: The API key for ENTSO-E Transparency Platform.
"""

import argparse
import logging
import os
import sys

from spotforecast2_safe.downloader.entsoe import download_new_data, merge_build_manual
from spotforecast2_safe.manager.predictor import (
    get_model_prediction as get_model_prediction_safe,
)
from spotforecast2_safe.manager.trainer import handle_training as handle_training_safe

from spotforecast2.manager.plotter import make_plot
from spotforecast2 import ConfigEntsoe
from spotforecast2_safe.manager.models import (
    ForecasterRecursiveLGBM as _ForecasterRecursiveLGBMBase,
    ForecasterRecursiveXGB as _ForecasterRecursiveXGBBase,
)

# Create default configuration instance
config = ConfigEntsoe()


# Wrapper classes that inject config values for safety-critical consistency
class ForecasterRecursiveLGBM(_ForecasterRecursiveLGBMBase):
    """
    LGBM forecaster with config-injected defaults.

    Ensures all model instances use consistent configuration values
    from ConfigEntsoe for periods, country_code, and random_state.
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs):
        # Inject config values if not explicitly provided
        kwargs.setdefault("periods", config.periods)
        kwargs.setdefault("country_code", config.API_COUNTRY_CODE)
        kwargs.setdefault("random_state", config.random_state)
        kwargs.setdefault("end_dev", config.end_train_default)
        kwargs.setdefault("train_size", config.train_size)
        kwargs.setdefault("delta_val", config.delta_val)
        kwargs.setdefault("predict_size", config.predict_size)
        kwargs.setdefault("refit_size", config.refit_size)
        super().__init__(iteration=iteration, lags=lags, **kwargs)


class ForecasterRecursiveXGB(_ForecasterRecursiveXGBBase):
    """
    XGBoost forecaster with config-injected defaults.

    Ensures all model instances use consistent configuration values
    from ConfigEntsoe for periods, country_code, and random_state.
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs):
        # Inject config values if not explicitly provided
        kwargs.setdefault("periods", config.periods)
        kwargs.setdefault("country_code", config.API_COUNTRY_CODE)
        kwargs.setdefault("random_state", config.random_state)
        kwargs.setdefault("end_dev", config.end_train_default)
        kwargs.setdefault("train_size", config.train_size)
        kwargs.setdefault("delta_val", config.delta_val)
        kwargs.setdefault("predict_size", config.predict_size)
        kwargs.setdefault("refit_size", config.refit_size)
        super().__init__(iteration=iteration, lags=lags, **kwargs)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --- Main CLI ---


def main():
    parser = argparse.ArgumentParser(description="spotforecast2 unified task")
    subparsers = parser.add_subparsers(dest="subcommand")

    # Download
    parser_dl = subparsers.add_parser("download")
    parser_dl.add_argument("--api-key", help="ENTSO-E API Key")
    parser_dl.add_argument("--force", action="store_true")
    parser_dl.add_argument("dates", nargs="*", help="Start [End]")

    # Train
    parser_tr = subparsers.add_parser("train")
    parser_tr.add_argument("model", choices=["lgbm", "xgb"], default="lgbm", nargs="?")
    parser_tr.add_argument("--force", action="store_true")

    # Predict
    parser_pr = subparsers.add_parser("predict")
    parser_pr.add_argument(
        "model",
        choices=["lgbm", "xgb"],
        default="lgbm",
        nargs="?",
        help="Model to use for prediction (default: lgbm)",
    )
    parser_pr.add_argument("--plot", action="store_true")

    # Merge
    subparsers.add_parser("merge")

    args = parser.parse_args()

    if args.subcommand == "download":
        api_key = args.api_key or os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            logger.error(
                "API Key not provided. Set ENTSOE_API_KEY env var or use --api-key."
            )
            sys.exit(1)

        start = args.dates[0] if args.dates else None
        end = args.dates[1] if args.dates and len(args.dates) > 1 else None
        download_new_data(api_key=api_key, start=start, end=end, force=args.force)

    elif args.subcommand == "train":
        # Register models dynamically if needed, or pass classes
        model_map = {"lgbm": ForecasterRecursiveLGBM, "xgb": ForecasterRecursiveXGB}
        handle_training_safe(
            model_class=model_map[args.model],
            model_name=args.model,
            force=args.force,
            data_filename=config.data_filename,
        )

    elif args.subcommand == "predict":
        model_name = getattr(
            args, "model", "lgbm"
        )  # Default to lgbm for backward compatibility
        out = get_model_prediction_safe(model_name=model_name)
        if out:
            logger.info("Prediction successful.")
            if args.plot:
                make_plot(out)

    elif args.subcommand == "merge":
        merge_build_manual()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
