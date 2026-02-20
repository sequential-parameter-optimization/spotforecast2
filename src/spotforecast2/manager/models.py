# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Full-featured forecasting model classes with Bayesian tuning and SHAP.

This module extends the stub implementations in ``spotforecast2-safe``
with real Bayesian hyperparameter search (Optuna) and SHAP-based
feature importance (``shap.TreeExplainer``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from spotforecast2.model_selection import bayesian_search_forecaster
from spotforecast2_safe.data.fetch_data import load_timeseries
from spotforecast2_safe.manager.models.forecaster_recursive_lgbm import (
    ForecasterRecursiveLGBM,
)
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)
from spotforecast2_safe.manager.models.forecaster_recursive_xgb import (
    ForecasterRecursiveXGB,
)
from spotforecast2.manager.trainer_full import SEARCH_SPACES
from spotforecast2_safe.preprocessing import LinearlyInterpolateTS

logger = logging.getLogger(__name__)

# Default number of Optuna trials when none is specified.
_DEFAULT_N_TRIALS: int = 10


class ForecasterRecursiveModelFull(ForecasterRecursiveModel):
    """ForecasterRecursiveModel with real Bayesian tuning and SHAP.

    This class overrides the two stubs in ``spotforecast2-safe``:

    * :meth:`tune` — performs a full Bayesian hyperparameter search
      using ``bayesian_search_forecaster`` (Optuna).
    * :meth:`get_global_shap_feature_importance` — computes global
      SHAP values using ``shap.TreeExplainer``.

    Examples:
        >>> from spotforecast2.manager.models import ForecasterRecursiveModelFull
        >>> model = ForecasterRecursiveModelFull(iteration=0)
        >>> hasattr(model, 'tune')
        True
        >>> hasattr(model, 'get_global_shap_feature_importance')
        True
    """

    def __init__(
        self,
        iteration: int,
        n_trials: int = _DEFAULT_N_TRIALS,
        **kwargs: Any,
    ):
        super().__init__(iteration, **kwargs)
        self.n_trials = n_trials

    # ------------------------------------------------------------------
    # Bayesian hyperparameter tuning
    # ------------------------------------------------------------------

    def tune(self) -> None:
        """Tune the forecaster via Bayesian search (Optuna).

        Loads time-series data, builds exogenous features, and runs
        ``bayesian_search_forecaster`` over the search space registered
        for ``self.name`` in ``SEARCH_SPACES``.

        After tuning the model is fitted with the best parameters and,
        if ``self.save_model_to_file`` is *True*, persisted to disk.

        Raises:
            KeyError: If ``self.name`` is not in ``SEARCH_SPACES``.
        """
        logger.info("Tuning %s Forecaster %d", self.name.upper(), self.iteration)

        # Load and preprocess data
        y = load_timeseries()
        y = LinearlyInterpolateTS().fit_transform(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=self.end_dev)

        # Training window boundaries
        end_train = self.end_dev - pd.Timedelta(
            hours=self.predict_size * self.refit_size
        )
        start_train = self._get_init_train(y.index.min(), end_train)
        fixed_train_size = self.train_size is not None
        length_training = len(y.loc[start_train:end_train])

        # Bayesian search
        results, _ = bayesian_search_forecaster(
            forecaster=self.forecaster,
            y=y.loc[start_train : self.end_dev],
            cv=self._build_cv(
                train_size=length_training,
                fixed_train_size=fixed_train_size,
                refit=False,
            ),
            search_space=SEARCH_SPACES[self.name],
            metric=self.metrics,
            exog=X.loc[start_train : self.end_dev],
            return_best=False,
            random_state=self.random_state,
            verbose=False,
            n_trials=self.n_trials,
        )

        # Record results
        results["name"] = self.name
        self.results_tuning = results
        logger.info("Best parameters:\n%s", results.iloc[0])
        self.best_params = results.iloc[0].params
        self.best_lags = results.iloc[0].lags

        # Fit with best and persist
        self.fit_with_best()
        self.is_tuned = True
        logger.info(
            "Model trained with data from %s until %s!",
            start_train,
            self.end_dev,
        )

        if self.save_model_to_file:
            self.save_to_file()

    # ------------------------------------------------------------------
    # SHAP-based feature importance
    # ------------------------------------------------------------------

    def get_global_shap_feature_importance(self, frac: float = 0.1) -> pd.Series:
        """Return global SHAP-based feature importances.

        Uses ``shap.TreeExplainer`` on the underlying estimator to
        compute mean absolute SHAP values across a random sample of
        the training data.

        Args:
            frac: Fraction of training data to sample (0 < frac <= 1).

        Returns:
            pd.Series: Feature importances sorted descending.  Empty
            if the model has not been tuned.

        Raises:
            ValueError: If the forecaster has not been initialized.
        """
        X_train, y_train = self._get_training_data()
        X_train_sample = X_train.sample(frac=frac, random_state=self.random_state)

        if self.best_params is None or self.best_lags is None:
            logger.warning("Model is not tuned — returning empty Series.")
            return pd.Series(dtype=float)

        shap.initjs()
        explainer = shap.TreeExplainer(self.forecaster.estimator)
        shap_values = explainer.shap_values(X_train_sample)
        average_shap_values = np.abs(shap_values).mean(axis=0)
        shap_importance = (
            pd.Series(average_shap_values, index=X_train_sample.columns)
            .abs()
            .sort_values(ascending=False)
        )
        return shap_importance


# ------------------------------------------------------------------
# Convenience subclasses
# ------------------------------------------------------------------


class ForecasterRecursiveLGBMFull(
    ForecasterRecursiveModelFull, ForecasterRecursiveLGBM
):
    """LGBM forecaster with real Bayesian tuning and SHAP.

    Inherits the LGBM forecaster initialisation from
    ``ForecasterRecursiveLGBM`` (``spotforecast2-safe``) and adds
    the real ``tune()`` and ``get_global_shap_feature_importance()``
    from ``ForecasterRecursiveModelFull``.

    Examples:
        >>> from spotforecast2.manager.models import ForecasterRecursiveLGBMFull
        >>> model = ForecasterRecursiveLGBMFull(iteration=0)
        >>> model.name
        'lgbm'
        >>> model.forecaster is not None
        True
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        super().__init__(iteration=iteration, lags=lags, **kwargs)


class ForecasterRecursiveXGBFull(ForecasterRecursiveModelFull, ForecasterRecursiveXGB):
    """XGBoost forecaster with real Bayesian tuning and SHAP.

    Inherits the XGBoost forecaster initialisation from
    ``ForecasterRecursiveXGB`` (``spotforecast2-safe``) and adds
    the real ``tune()`` and ``get_global_shap_feature_importance()``
    from ``ForecasterRecursiveModelFull``.

    Examples:
        >>> from spotforecast2.manager.models import ForecasterRecursiveXGBFull
        >>> model = ForecasterRecursiveXGBFull(iteration=0)
        >>> model.name
        'xgb'
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        super().__init__(iteration=iteration, lags=lags, **kwargs)
