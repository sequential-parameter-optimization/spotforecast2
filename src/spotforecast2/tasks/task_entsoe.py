# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unified CLI for the ENTSO-E single-target forecasting pipeline.

Drives the pipeline directly through `spotforecast2.multitask.multi.MultiTask`
with `ConfigEntsoe` plugged in via the `data_loader` and `forecaster_factory`
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
from spotforecast2_safe.configurator import ConfigEntsoe
from spotforecast2_safe.data.fetch_data import get_cache_home, get_data_home
from spotforecast2_safe.downloader.entsoe import download_new_data, merge_build_manual
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.trainer import should_retrain
from spotforecast2_safe.preprocessing import RollingFeatures
from xgboost import XGBRegressor

from spotforecast2.multitask.multi import MultiTask

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
        config: A `ConfigEntsoe` with ``data_filename`` set.  Relative paths
            are resolved against `spotforecast2_safe.data.fetch_data.get_data_home`.

    Returns:
        DataFrame indexed by the ENTSO-E timestamp column (``Time (UTC)``)
        with the load columns as data columns.

    Raises:
        FileNotFoundError: If the merged CSV does not exist.  Run
            ``spotforecast2-entsoe download`` and ``merge`` first.

    Examples:
        ```{python}
        import os
        import tempfile

        import pandas as pd
        from spotforecast2_safe.configurator import ConfigEntsoe

        from spotforecast2.tasks.task_entsoe import entsoe_data_loader

        # Build a tiny synthetic interim CSV in a temp directory.
        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "energy_load.csv")
        idx = pd.date_range(
            "2025-01-01", periods=48, freq="h", tz="UTC", name="Time (UTC)"
        )
        pd.DataFrame({"Actual Load": range(48)}, index=idx).to_csv(csv_path)

        # Absolute path bypasses get_data_home; loader returns the full frame.
        config = ConfigEntsoe()
        config.data_filename = csv_path
        df = entsoe_data_loader(config)

        print(df.shape)
        assert df.shape == (48, 1)
        assert df.index.name == "Time (UTC)"
        ```
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
    """Return the merged ENTSO-E CSV sliced to the forecast horizon.

    The slice spans ``(end_train, end_train + predict_size * 1 h]`` so that
    ``build_prediction_package``'s ``test_actual = ts.reindex(future_pred.index)``
    matches the hourly forecast row-for-row.  ``end_train`` is taken from
    ``config.end_train_default`` (treated as the *inclusive* last training
    timestamp, the same convention the forecaster uses), and the step is
    assumed to be 1 h after the pipeline's hourly resampling.

    For the live ENTSO-E exemplar with ``end_train_default = D-2 23:00 UTC``
    and ``predict_size = 24``, this returns the rows for
    ``[D-1 00:00, D 00:00)`` — i.e., ``y_{-1}``.  For backtests at an arbitrary
    ``end_train_default``, it returns the post-cutoff window the model is
    actually predicting, rather than always "yesterday in wall-clock UTC".

    Args:
        config: A `ConfigEntsoe` with ``data_filename``, ``end_train_default``,
            and ``predict_size`` set; the merged interim CSV must already
            contain data covering the forecast horizon (run
            ``spotforecast2-entsoe download`` first).

    Returns:
        DataFrame indexed by ``Time (UTC)`` with the rows the forecast will be
        scored against.

    Examples:
        ```{python}
        import os
        import tempfile

        import pandas as pd
        from spotforecast2_safe.configurator import ConfigEntsoe

        from spotforecast2.tasks.task_entsoe import entsoe_test_data_loader

        # Synthetic interim CSV spanning the forecast window.
        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "energy_load.csv")
        idx = pd.date_range(
            "2025-12-29 00:00", periods=120, freq="h", tz="UTC", name="Time (UTC)"
        )
        pd.DataFrame({"Actual Load": range(120)}, index=idx).to_csv(csv_path)

        config = ConfigEntsoe()
        config.data_filename = csv_path
        config.end_train_default = "2025-12-31 00:00+00:00"
        config.predict_size = 24

        test_df = entsoe_test_data_loader(config)

        # The slice covers exactly predict_size hourly steps after end_train.
        print(test_df.shape)
        assert test_df.shape == (24, 1)
        assert test_df.index[0] == pd.Timestamp("2025-12-31 01:00", tz="UTC")
        ```
    """
    df = entsoe_data_loader(config)
    end_train = pd.Timestamp(config.end_train_default)
    if end_train.tzinfo is None:
        end_train = end_train.tz_localize("UTC")
    step = pd.Timedelta(hours=1)  # post-resample assumption
    start = end_train + step  # first forecast step
    end = start + config.predict_size * step  # exclusive upper bound
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
            ``window_size`` (typically `ConfigEntsoe`).
        weight_func: Per-sample weight function from the imputation step.
        target: Ignored; accepted for factory-signature compatibility.

    Examples:
        ```{python}
        from spotforecast2_safe.configurator import ConfigEntsoe
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

        from spotforecast2.tasks.task_entsoe import entsoe_lgbm_factory

        config = ConfigEntsoe()
        forecaster = entsoe_lgbm_factory(config, weight_func=None, target="Actual Load")

        print(type(forecaster).__name__)
        assert isinstance(forecaster, ForecasterRecursive)
        # The lags array is derived from lags_consider[-1] = 23.
        assert len(forecaster.lags) == config.lags_consider[-1]
        print("lags:", forecaster.lags)
        ```
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
    """XGBoost ForecasterRecursive for the ENTSO-E pipeline.

    Mirrors `entsoe_lgbm_factory()` but uses an `XGBRegressor` estimator.
    Kept as a named helper so the XGBoost variant is explicit at the
    configuration site.

    Args:
        config: Any object exposing ``random_state``, ``lags_consider``, and
            ``window_size`` (typically `ConfigEntsoe`).
        weight_func: Per-sample weight function from the imputation step.
        target: Ignored; accepted for factory-signature compatibility.

    Examples:
        ```{python}
        from spotforecast2_safe.configurator import ConfigEntsoe
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from xgboost import XGBRegressor

        from spotforecast2.tasks.task_entsoe import entsoe_xgb_factory

        config = ConfigEntsoe()
        forecaster = entsoe_xgb_factory(config, weight_func=None, target="Actual Load")

        print(type(forecaster).__name__)
        assert isinstance(forecaster, ForecasterRecursive)
        assert isinstance(forecaster.estimator, XGBRegressor)
        print("lags:", forecaster.lags)
        ```
    """
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
            pipeline passes for ``data_frame_name``.

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


def _run_entsoe_pipeline(
    config: ConfigEntsoe,
    *,
    task: str,
    project_name: str,
    show: bool,
) -> None:
    """Run the ENTSO-E single-target forecasting pipeline.

    Wires ``config`` into a ``MultiTask`` instance, sets
    ``config.data_frame_name`` to ``project_name``, then executes the
    standard five-step pipeline sequence and calls ``MultiTask.run(show=show)``.

    Args:
        config: A ``ConfigEntsoe`` already wired with ``data_loader`` and
            ``forecaster_factory``.
        task: Pipeline task mode — ``"defaults"``, ``"predict"``, etc.
        project_name: Active-dataset identifier; drives the cache-subdirectory
            and model-file naming.
        show: Whether to display prediction figures after the run.
    """
    config.data_frame_name = project_name
    mt = MultiTask(
        config,
        task=task,
        cache_home=get_cache_home(config.cache_home),
        log_level=logging.ERROR,
    )
    mt.prepare_data()
    mt.detect_outliers()
    mt.impute()
    mt.build_exogenous_features()
    mt.run(show=show)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the ``spotforecast2-entsoe`` console script.

    Parses ``sys.argv`` and dispatches to one of four subcommands:
    ``download``, ``merge``, ``train``, or ``predict``.  Calling with no
    subcommand prints the top-level help and returns.

    Examples:
        ```{python}
        import sys

        from spotforecast2.tasks.task_entsoe import main

        # With no subcommand, main() prints the usage summary and returns
        # without error — useful for verifying the CLI is wired correctly.
        sys.argv = ["spotforecast2-entsoe"]
        main()  # prints usage and returns normally
        ```
    """
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
        _run_entsoe_pipeline(
            config,
            task="defaults",
            project_name=project_name,
            show=args.show,
        )

    elif args.subcommand == "predict":
        config = _build_entsoe_config(args.model)
        _run_entsoe_pipeline(
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
