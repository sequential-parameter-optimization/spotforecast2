# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unified CLI for the ENTSO-E single-target forecasting pipeline.

Thin argparse wrapper over `spotforecast2.multitask.runner.run` with
`ConfigEntsoe` plugged in via the `data_loader` and `forecaster_factory`
hooks introduced in ADR-001.  The CLI exposes four subcommands:

- ``download`` — fetch raw data from the ENTSO-E Transparency Platform.
- ``merge`` — concatenate raw CSVs into the interim merged file.
- ``train`` — fit a model and save it to the cache.
- ``predict`` — load the saved model and produce a forecast.

Usage::

    spotforecast2-entsoe download --api-key <KEY>
    spotforecast2-entsoe train lgbm
    spotforecast2-entsoe predict lgbm --plot

Environment Variables:
    ENTSOE_API_KEY: API key for the ENTSO-E Transparency Platform.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from spotforecast2_safe.configurator import ConfigEntsoe
from spotforecast2_safe.data.fetch_data import get_cache_home, get_data_home
from spotforecast2_safe.downloader.entsoe import download_new_data, merge_build_manual
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.trainer import should_retrain
from spotforecast2_safe.preprocessing import RollingFeatures

from spotforecast2.multitask.runner import run


_PROJECT_BY_MODEL = {"lgbm": "entsoe-lgbm", "xgb": "entsoe-xgb"}

# Default single-target list for ENTSO-E load forecasting.  ``download_new_data``
# / ``merge_build_manual`` produce a CSV whose load column is named
# ``Actual Load``; ``Forecasted Load`` is the official day-ahead benchmark
# kept around as a covariate but not predicted.
_DEFAULT_TARGETS = ["Actual Load"]

# Single-target ``bounds`` and ``agg_weights`` lists.  Passed explicitly so
# the runner does NOT substitute its 11-element demo10 defaults.  The wide
# bounds make the manual outlier-removal step a no-op; IsolationForest still
# runs through ``config.use_outlier_detection``.
_DEFAULT_BOUNDS = [(-1e9, 1e9)]
_DEFAULT_AGG_WEIGHTS = [1.0]


def entsoe_data_loader(config: ConfigEntsoe) -> pd.DataFrame:
    """Read the merged interim ENTSO-E CSV that ``config.data_filename`` points at.

    Args:
        config: A ``ConfigEntsoe`` with ``data_filename`` set.  Relative paths
            are resolved against ``spotforecast2_safe.data.fetch_data.get_data_home``.

    Returns:
        DataFrame indexed by the ENTSO-E timestamp column (``Time (UTC)``)
        with the load columns as data columns.

    Raises:
        FileNotFoundError: If the merged CSV does not exist.  Run
            ``spotforecast2-entsoe download`` and ``merge`` first.
    """
    path = Path(config.data_filename)
    if not path.is_absolute():
        path = get_data_home() / path
    if not path.exists():
        raise FileNotFoundError(
            f"ENTSO-E merged CSV not found at {path}.  Run "
            "`spotforecast2-entsoe download` and `merge` first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def entsoe_test_data_loader(config: ConfigEntsoe) -> pd.DataFrame:
    """Return the merged ENTSO-E CSV sliced to "yesterday in UTC".

    Mirrors ``entsoe_data_loader`` but returns only the 24 hourly rows for the
    UTC day immediately preceding today.  Plug into
    ``config.test_data_loader`` so ``BaseTask.prepare_data`` consumes the slice
    as ground truth, populating ``test_actual`` and ``metrics_future`` in the
    prediction package.

    Args:
        config: A ``ConfigEntsoe`` with ``data_filename`` set; the merged
            interim CSV must already contain data covering yesterday (run
            ``spotforecast2-entsoe download`` first).

    Returns:
        DataFrame indexed by ``Time (UTC)`` with the rows spanning
        ``[yesterday 00:00 UTC, today 00:00 UTC)``.
    """
    df = entsoe_data_loader(config)
    end = pd.Timestamp.now(tz="UTC").floor("D")
    start = end - pd.Timedelta(days=1)
    if df.index.tz is None:
        start = start.tz_localize(None)
        end = end.tz_localize(None)
    return df.loc[(df.index >= start) & (df.index < end)]


def entsoe_lgbm_factory(
    config: ConfigEntsoe,
    *,
    weight_func: Optional[Any] = None,
    target: Optional[str] = None,
) -> ForecasterRecursive:
    """LightGBM ForecasterRecursive for the ENTSO-E pipeline.

    Identical to ``spotforecast2.multitask.factories.default_lgbm_forecaster_factory``;
    kept as a named helper so the CLI's intent ("use LightGBM here") is
    visible at the configuration site.

    Args:
        config: Any object exposing ``random_state``, ``lags_consider``, and
            ``window_size`` (typically ``ConfigEntsoe``).
        weight_func: Per-sample weight function from the imputation step.
        target: Ignored; accepted for factory-signature compatibility.
    """
    del target
    return ForecasterRecursive(
        estimator=LGBMRegressor(random_state=config.random_state, verbose=-1),
        lags=config.lags_consider[-1],
        window_features=RollingFeatures(
            stats=["mean"], window_sizes=config.window_size
        ),
        weight_func=weight_func,
    )


def entsoe_xgb_factory(
    config: ConfigEntsoe,
    *,
    weight_func: Optional[Any] = None,
    target: Optional[str] = None,
) -> ForecasterRecursive:
    """XGBoost ForecasterRecursive for the ENTSO-E pipeline."""
    del target
    return ForecasterRecursive(
        estimator=XGBRegressor(random_state=config.random_state, verbosity=0),
        lags=config.lags_consider[-1],
        window_features=RollingFeatures(
            stats=["mean"], window_sizes=config.window_size
        ),
        weight_func=weight_func,
    )


_FACTORY_BY_MODEL = {"lgbm": entsoe_lgbm_factory, "xgb": entsoe_xgb_factory}


def _build_entsoe_config(model: str) -> ConfigEntsoe:
    """Build a ConfigEntsoe wired up with the ENTSO-E loader and factory."""
    config = ConfigEntsoe()
    config.targets = list(_DEFAULT_TARGETS)
    config.agg_weights = list(_DEFAULT_AGG_WEIGHTS)
    config.bounds = list(_DEFAULT_BOUNDS)
    config.data_loader = entsoe_data_loader
    config.forecaster_factory = _FACTORY_BY_MODEL[model]
    return config


def _latest_saved_model_timestamp(
    config: ConfigEntsoe, project_name: str
) -> Optional[pd.Timestamp]:
    """Return the mtime of the most recent saved forecaster for the project.

    Forecasters are persisted by ``BaseTask._run_strategy`` into
    ``<cache_home>/models/<project_name>/``.  Their mtime is the most
    reliable "last trained at" signal available without changing the
    persistence scheme.

    Args:
        config: A ``ConfigEntsoe`` (its ``cache_home`` resolves the model
            directory).
        project_name: Project / dataset identifier — same value the
            runner passes for ``data_frame_name``.

    Returns:
        UTC ``pd.Timestamp`` of the newest ``.joblib`` file, or ``None``
        if the directory is empty / missing.
    """
    model_dir = Path(get_cache_home(config.cache_home)) / "models" / project_name
    if not model_dir.exists():
        return None
    candidates = list(model_dir.glob("*.joblib"))
    if not candidates:
        return None
    latest_mtime = max(c.stat().st_mtime for c in candidates)
    return pd.Timestamp(latest_mtime, unit="s", tz="UTC")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the ``spotforecast2-entsoe`` console script."""
    parser = argparse.ArgumentParser(description="spotforecast2 ENTSO-E pipeline")
    subparsers = parser.add_subparsers(dest="subcommand")

    parser_dl = subparsers.add_parser("download", help="Download raw ENTSO-E data")
    parser_dl.add_argument("--api-key", help="ENTSO-E API key")
    parser_dl.add_argument("--force", action="store_true")
    parser_dl.add_argument("dates", nargs="*", help="Start [End]")

    parser_tr = subparsers.add_parser("train", help="Train a forecaster")
    parser_tr.add_argument("model", choices=["lgbm", "xgb"], default="lgbm", nargs="?")
    parser_tr.add_argument("--show", action="store_true")
    parser_tr.add_argument(
        "--force",
        action="store_true",
        help="Bypass the retraining cadence gate (config.retrain_max_age).",
    )

    parser_pr = subparsers.add_parser("predict", help="Predict with a saved forecaster")
    parser_pr.add_argument("model", choices=["lgbm", "xgb"], default="lgbm", nargs="?")
    parser_pr.add_argument("--show", action="store_true")

    subparsers.add_parser("merge", help="Merge raw CSVs into the interim file")

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
        config = _build_entsoe_config(args.model)
        project_name = _PROJECT_BY_MODEL[args.model]
        last_trained_at = _latest_saved_model_timestamp(config, project_name)
        if not should_retrain(
            last_trained_at,
            max_age=config.retrain_max_age,
            force=args.force,
        ):
            return
        run(
            config,
            task="defaults",
            project_name=project_name,
            show=args.show,
        )

    elif args.subcommand == "predict":
        config = _build_entsoe_config(args.model)
        run(
            config,
            task="predict",
            project_name=_PROJECT_BY_MODEL[args.model],
            show=args.show,
        )

    elif args.subcommand == "merge":
        merge_build_manual()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
