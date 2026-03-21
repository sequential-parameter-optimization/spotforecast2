# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Module for managing full model training.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from joblib import dump

from spotforecast2_safe.data.fetch_data import fetch_data, get_cache_home, get_data_home
from spotforecast2_safe.manager.trainer import get_last_model
from spotforecast2_safe.preprocessing import RollingFeatures

logger = logging.getLogger(__name__)


#: Candidate lag values for hyperparameter search.
LAGS_CONSIDER: list[int] = list(range(1, 24))

#: Default rolling window features matching the original chag25a configuration.
#: Each entry is a separate RollingFeatures instance to avoid duplicate-name
#: collisions in spotforecast2-safe's ``initialize_window_features``.
window_features = [
    RollingFeatures(stats="mean", window_sizes=24),
    RollingFeatures(stats="mean", window_sizes=24 * 7),
    RollingFeatures(stats="mean", window_sizes=24 * 30),
    RollingFeatures(stats="min", window_sizes=24),
    RollingFeatures(stats="max", window_sizes=24),
]


def search_space_lgbm(trial: Any) -> dict:
    """Optuna search space for LightGBM hyperparameters.

    Args:
        trial: An :class:`optuna.trial.Trial` instance.

    Returns:
        dict: Suggested hyperparameters for the current trial.

    Examples:
        >>> from spotforecast2.manager.trainer_full import search_space_lgbm
        >>> # Without Optuna, verify the function signature exists
        >>> callable(search_space_lgbm)
        True
    """
    search_space = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100),
        "lags": trial.suggest_categorical("lags", LAGS_CONSIDER),
    }
    return search_space


def search_space_xgb(trial: Any) -> dict:
    """Optuna search space for XGBoost hyperparameters.

    Args:
        trial: An :class:`optuna.trial.Trial` instance.

    Returns:
        dict: Suggested hyperparameters for the current trial.

    Examples:
        >>> from spotforecast2.manager.trainer_full import search_space_xgb
        >>> callable(search_space_xgb)
        True
    """
    search_space = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "n_estimators": trial.suggest_int("n_estimators", 50, 600, step=50),
        "alpha": trial.suggest_float("alpha", 0.0, 0.5),
        "lambda": trial.suggest_float("lambda", 0.0, 0.5),
        "lags": trial.suggest_categorical("lags", LAGS_CONSIDER),
    }
    return search_space


#: Registry mapping model names to their search space functions.
SEARCH_SPACES: dict[str, Any] = {
    "lgbm": search_space_lgbm,
    "xgb": search_space_xgb,
}


def train_new_model(
    model_class: type,
    n_iteration: int,
    model_name: Optional[str] = None,
    train_size: Optional[pd.Timedelta] = None,
    save_to_file: bool = True,
    model_dir: Optional[Union[str, Path]] = None,
    end_dev: Optional[Union[str, pd.Timestamp]] = None,
    data_filename: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Train a new forecaster model and optionally save it to disk.

    This function fetches the latest data, calculates the training cutoff,
    initializes a model of the given class, triggers the tuning process,
    and saves the model following the naming convention:
    `{model_name}_forecaster_{n_iteration}.joblib`.

    Args:
        model_class (type):
            The class of the forecaster model to train.
            The class should accept `iteration`, `end_dev`, and `train_size`
            in its constructor and provide a `tune()` method.
        n_iteration (int):
            The iteration number for this training run.
            This acts as an incrementing version number for the model.
            When using `handle_training`, the first model starts at iteration 0.
            Upon subsequent forced or scheduled retrainings, it is incremented
            by 1 (`get_last_model_iteration + 1`). It is primarily used to
            determine the filename when saving the model to disk
            (e.g., `lgbm_forecaster_0.joblib`, `lgbm_forecaster_1.joblib`).
        model_name (Optional[str]):
            Optional name of the model to train.
            If None, the name is inferred from the model class.
            Defaults to None.
        train_size (Optional[pd.Timedelta]):
            Optional size of the training set as a pandas Timedelta.
            Determines the lookback window length from `end_dev`. If provided, the training data
            will start at `end_dev - train_size`. If None, all available data up to `end_dev` is used.
            Defaults to None.
        save_to_file (bool):
            If True, saves the model to disk after training.
            Defaults to True.
        model_dir (Optional[Union[str, Path]]):
            Directory where the model should be saved. If None, defaults to
            the library's cache home.
        end_dev (Optional[Union[str, pd.Timestamp]]):
            Optional cutoff date for training.
            This represents the absolute point in time separating training/development data
            from unseen future data. If None, it is calculated automatically to be one day
            before the latest available index in the data.
        data_filename (Optional[str]):
            Absolute path to the CSV file used for training (e.g.,
            ``str(get_data_home() / 'interim/energy_load.csv')``).
            Relative paths are resolved against :func:`~spotforecast2_safe.data.fetch_data.get_data_home`.
            If None, a ``ValueError`` is raised by :func:`~spotforecast2_safe.data.fetch_data.fetch_data`.
            Defaults to None.
        **kwargs (Any):
            Additional keyword arguments to be passed to the model constructor.

    Notes:
        Relationship between ``train_size`` and ``end_dev``:
        The actual training data spans from ``max(dataset_start, end_dev - train_size)`` to ``end_dev``.
        - If ``train_size`` is larger than the available history before ``end_dev``, the framework
          gracefully clips the start date to the beginning of the dataset without throwing an error.
        - If ``end_dev`` is set to a time before the start of the dataset, the training subset will
          be empty and the forecaster will fail to fit.

    Examples:
        ```{python}
        import pandas as pd
        from spotforecast2.manager.trainer_full import train_new_model

        # Define a mock model class for demonstration
        class MyModel:
            def __init__(self, iteration, end_dev, train_size, **kwargs):
                self.iteration = iteration
                self.end_dev = end_dev
                self.train_size = train_size
            def tune(self): print(f"Tuning model {self.iteration} up to {self.end_dev}!")
            def get_params(self): return {}
            @property
            def name(self): return "mymodel"

        # Train using exactly 3 years of data leading up to the end of 2025:
        # Note: In a real scenario, this fetches data and saves a joblib file.
        # We pass save_to_file=False to avoid writing disk artifacts in the doc example.
        from spotforecast2_safe.data.fetch_data import get_package_data_home
        demo_file = get_package_data_home() / "demo01.csv"

        model = train_new_model(
            model_class=MyModel,
            n_iteration=0,
            train_size=pd.Timedelta(days=3*365),
            end_dev="2025-12-31 00:00+00:00",
            save_to_file=False,
            data_filename=str(demo_file)
        )
        ```

    Returns:
        The trained model instance.

    """
    logger.info("Training new model (iteration %d)...", n_iteration)

    # Resolve data path: require absolute path for fetch_data
    if data_filename is None:
        raise ValueError(
            "data_filename must be provided. "
            "Pass an absolute path, e.g. str(get_data_home() / 'my_data.csv')."
        )
    data_path = Path(data_filename)
    if not data_path.is_absolute():
        data_path = get_data_home() / data_filename

    current_data = fetch_data(filename=data_path)
    if current_data.empty:
        logger.error("No data fetched. Aborting training.")
        return None

    if end_dev is None:
        latest_idx = current_data.index[-1]
        # Calculate training cutoff. In this implementation, we use data up to one day
        # before the latest recorded index to ensure we have a full day's data for
        # validation or the last training window.
        end_train_cutoff = latest_idx - pd.Timedelta(days=1)
        logger.debug("Latest data index: %s", latest_idx)
        logger.debug("Calculated training cutoff: %s", end_train_cutoff)
    else:
        end_train_cutoff = pd.to_datetime(end_dev, utc=True)
        logger.debug("Using provided training cutoff: %s", end_train_cutoff)

    # Initialize the model instance
    model = model_class(
        iteration=n_iteration,
        end_dev=end_train_cutoff,
        train_size=train_size,
        **kwargs,
    )
    logger.debug("Model initialized: %s", model)
    logger.debug("Model parameters: %s", model.get_params())

    # Perform hyperparameter tuning and fitting as implemented in model_class
    logger.info("Starting model tuning...")
    model.tune()
    logger.info("Training and tuning completed for iteration %d.", n_iteration)

    if save_to_file:
        if model_dir is None:
            model_dir = get_cache_home()
        else:
            model_dir = Path(model_dir)

        model_dir.mkdir(parents=True, exist_ok=True)

        # Use provided model_name, or model's 'name' attribute,
        # otherwise use lowercase class name
        if model_name is None:
            model_name = getattr(model, "name", model_class.__name__.lower())
        else:
            # Update model's internal name for consistency
            model.name = model_name

        file_path = model_dir / f"{model_name}_forecaster_{n_iteration}.joblib"

        try:
            dump(model, file_path, compress=3)
            logger.info("Saved model to %s", file_path)
        except Exception as e:
            logger.error("Failed to save model to %s: %s", file_path, e)

    return model


def handle_training(
    model_class: type,
    model_name: Optional[str] = None,
    model_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    train_size: Optional[pd.Timedelta] = None,
    end_dev: Optional[Union[str, pd.Timestamp]] = None,
    data_filename: Optional[str] = None,
    hours_until_retrain: int = 168,
    **kwargs: Any,
) -> None:
    """Check if a new model needs to be trained and trigger training if necessary.

    Inspects the most recently saved model (if any) and trains a new one when
    the model cache is empty, the existing model's ``end_dev`` is older than
    ``hours_until_retrain`` hours, or ``force=True`` is passed.  All training parameters
    are forwarded verbatim to :func:`train_new_model`.

    Args:
        model_class:
            The class of the forecaster model to train, for example
            ``spotforecast2_safe.forecaster.ForecasterLGBM``.  The class must
            accept ``iteration``, ``end_dev``, and ``train_size`` in its
            constructor and expose a ``tune()`` method.
        model_name:
            Short identifier for the model (e.g. ``'lgbm'``).  Used
            to locate existing model files and to name the new one.  If
            ``None``, the lower-cased class name is used.
        model_dir:
            Directory where model files are stored.  Forwarded to
            :func:`~spotforecast2_safe.manager.trainer.get_last_model` and
            :func:`train_new_model`.  Defaults to
            :func:`~spotforecast2_safe.data.fetch_data.get_cache_home`.
        force:
            If ``True``, retrain unconditionally regardless of the
            existing model's age.  Default is ``False``.
        train_size:
            Length of the training window forwarded to the model
            constructor.  ``None`` means all available data up to ``end_dev``.
        end_dev:
            Hard cutoff timestamp passed to the model constructor.  When
            ``None``, :func:`train_new_model` calculates it automatically as
            one day before the latest index in the dataset.
        data_filename:
            Path to the CSV training file forwarded to
            :func:`train_new_model`.  When ``None``, the library default is
            used.
        hours_until_retrain:
            Number of hours after which the existing model is  considered stale and retraining is triggered.  Default is 168 hours (7 days).
        **kwargs:
            Extra keyword arguments forwarded to the model constructor.

    Returns:
        None

    Examples:
        ```{python}
        import tempfile
        import pandas as pd
        from unittest.mock import patch
        from spotforecast2.manager.trainer_full import handle_training

        # Minimal model stub — no real ML libraries required
        class StubForecaster:
            name = "stub"
            def __init__(self, iteration, end_dev, train_size=None, **kw):
                self.iteration = iteration
                self.end_dev = end_dev
            def tune(self): pass
            def get_params(self): return {}

        # Scenario 1: empty cache → trains at iteration 0
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("spotforecast2.manager.trainer_full.get_last_model",
                       return_value=(-1, None)):
                with patch("spotforecast2.manager.trainer_full.train_new_model") as m:
                    handle_training(StubForecaster, model_name="stub", model_dir=tmpdir)
                    print(f"Scenario 1 — first training at iteration {m.call_args[0][1]}")

        # Scenario 2: recent model (24 h old) → skipped
        recent = StubForecaster(0, pd.Timestamp.now("UTC") - pd.Timedelta(hours=24))
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("spotforecast2.manager.trainer_full.get_last_model",
                       return_value=(0, recent)):
                with patch("spotforecast2.manager.trainer_full.train_new_model") as m:
                    handle_training(StubForecaster, model_name="stub", model_dir=tmpdir)
                    print(f"Scenario 2 — recent model, retraining called: {m.called}")

        # Scenario 3: stale model (10 days old) → retrains at iteration n+1
        stale = StubForecaster(2, pd.Timestamp.now("UTC") - pd.Timedelta(days=10))
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("spotforecast2.manager.trainer_full.get_last_model",
                       return_value=(2, stale)):
                with patch("spotforecast2.manager.trainer_full.train_new_model") as m:
                    handle_training(StubForecaster, model_name="stub", model_dir=tmpdir)
                    print(f"Scenario 3 — stale model retrained at iteration {m.call_args[0][1]}")

        # Scenario 4: force=True with recent model → retrains unconditionally
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("spotforecast2.manager.trainer_full.get_last_model",
                       return_value=(0, recent)):
                with patch("spotforecast2.manager.trainer_full.train_new_model") as m:
                    handle_training(
                        StubForecaster, model_name="stub", model_dir=tmpdir, force=True
                    )
                    print(f"Scenario 4 — forced retraining called: {m.called}")
        ```
    """
    if model_name is None:
        model_name = model_class.__name__.lower()

    n_iteration, current_model = get_last_model(model_name, model_dir)

    if current_model is None:
        logger.info("No model found for %s. Training iteration 0...", model_name)
        train_new_model(
            model_class,
            0,
            model_name=model_name,
            train_size=train_size,
            model_dir=model_dir,
            end_dev=end_dev,
            data_filename=data_filename,
            **kwargs,
        )
        return

    # Check how long since the model has been trained
    # Note: We expect the model instance to have an 'end_dev' attribute
    last_training_date = getattr(current_model, "end_dev", None)
    if last_training_date is None:
        logger.warning(
            "Current model has no 'end_dev' attribute. Cannot determine age. Forcing retraining."
        )
        train_new_model(
            model_class,
            n_iteration + 1,
            model_name=model_name,
            train_size=train_size,
            model_dir=model_dir,
            end_dev=end_dev,
            data_filename=data_filename,
            **kwargs,
        )
        return

    # Ensure last_training_date is a pandas Timestamp and timezone aware
    last_training_date = pd.to_datetime(last_training_date)
    if last_training_date.tzinfo is None:
        last_training_date = last_training_date.tz_localize("UTC")

    today = pd.Timestamp.now("UTC")
    hours_since_last_training = (today - last_training_date).total_seconds() // 3600

    # Train a new model  after `hours_until_retrain` hours or if forced. All training parameters
    if hours_since_last_training >= hours_until_retrain or force:
        logger.info(
            "Model for %s is old enough (%.0f hours) or retraining forced. "
            "Training iteration %d...",
            model_name,
            hours_since_last_training,
            n_iteration + 1,
        )
        train_new_model(
            model_class,
            n_iteration + 1,
            model_name=model_name,
            train_size=train_size,
            model_dir=model_dir,
            end_dev=end_dev,
            data_filename=data_filename,
            **kwargs,
        )
    else:
        logger.info(
            "The current %s model was trained up to %s (%.0f hours ago). "
            "No retraining necessary.",
            model_name,
            last_training_date,
            hours_since_last_training,
        )
