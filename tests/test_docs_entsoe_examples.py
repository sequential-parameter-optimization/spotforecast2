# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for ENTSOE documentation examples.

Each test corresponds to a code example in docs/entsoe.md to ensure
documentation remains accurate and executable.
"""

import pandas as pd
import pytest
from pathlib import Path


class TestConfigExamples:
    """Tests for ConfigEntsoe examples in documentation."""

    def test_default_config(self):
        """
        Example: Creating default configuration.

        ```python
        from spotforecast2 import Config

        config = Config()
        print(config.API_COUNTRY_CODE)  # 'DE'
        print(config.predict_size)      # 24
        print(config.random_state)      # 314159
        ```
        """
        from spotforecast2 import Config

        config = Config()

        assert config.API_COUNTRY_CODE == "DE"
        assert config.predict_size == 24
        assert config.random_state == 314159

    def test_custom_config(self):
        """
        Example: Creating custom configuration.

        ```python
        from spotforecast2 import Config
        import pandas as pd

        config = Config(
            api_country_code='FR',
            predict_size=48,
            refit_size=14,
            random_state=42
        )
        print(config.API_COUNTRY_CODE)  # 'FR'
        print(config.predict_size)      # 48
        ```
        """
        from spotforecast2 import Config

        config = Config(
            api_country_code='FR',
            predict_size=48,
            refit_size=14,
            random_state=42
        )

        assert config.API_COUNTRY_CODE == "FR"
        assert config.predict_size == 48
        assert config.refit_size == 14
        assert config.random_state == 42

    def test_config_periods(self):
        """
        Example: Accessing period configurations.

        ```python
        from spotforecast2 import Config

        config = Config()
        for period in config.periods:
            print(f"{period.name}: {period.n_periods} basis functions")
        ```
        """
        from spotforecast2 import Config

        config = Config()

        assert len(config.periods) == 5
        period_names = [p.name for p in config.periods]
        assert "daily" in period_names
        assert "weekly" in period_names
        assert "monthly" in period_names


class TestPeriodExamples:
    """Tests for Period dataclass examples in documentation."""

    def test_period_creation(self):
        """
        Example: Creating a Period for cyclical encoding.

        ```python
        from spotforecast2_safe.data import Period

        daily = Period(
            name='daily',
            n_periods=12,
            column='hour',
            input_range=(1, 24)
        )
        print(daily.name)        # 'daily'
        print(daily.n_periods)   # 12
        ```
        """
        from spotforecast2_safe.data import Period

        daily = Period(
            name='daily',
            n_periods=12,
            column='hour',
            input_range=(1, 24)
        )

        assert daily.name == "daily"
        assert daily.n_periods == 12
        assert daily.column == "hour"
        assert daily.input_range == (1, 24)


class TestExogBuilderExamples:
    """Tests for ExogBuilder examples in documentation."""

    def test_exog_builder_basic(self):
        """
        Example: Building exogenous features.

        ```python
        from spotforecast2_safe.preprocessing import ExogBuilder
        from spotforecast2_safe.data import Period
        import pandas as pd

        periods = [
            Period(name='daily', n_periods=12, column='hour', input_range=(1, 24)),
            Period(name='weekly', n_periods=7, column='dayofweek', input_range=(0, 6)),
        ]

        builder = ExogBuilder(periods=periods, country_code='DE')
        X = builder.build(
            start=pd.Timestamp('2025-01-01', tz='UTC'),
            end=pd.Timestamp('2025-01-02', tz='UTC')
        )
        print(X.shape)  # (24, 21) - 12 + 7 + 2 (holiday, weekend)
        ```
        """
        from spotforecast2_safe.preprocessing import ExogBuilder
        from spotforecast2_safe.data import Period
        import pandas as pd

        periods = [
            Period(name='daily', n_periods=12, column='hour', input_range=(1, 24)),
            Period(name='weekly', n_periods=7, column='dayofweek', input_range=(0, 6)),
        ]

        builder = ExogBuilder(periods=periods, country_code='DE')
        X = builder.build(
            pd.Timestamp('2025-01-01', tz='UTC'),
            pd.Timestamp('2025-01-02', tz='UTC')
        )

        assert X.shape[0] == 25  # 25 hours (inclusive end date)
        assert X.shape[1] == 21  # 12 + 7 + 2 (holiday, weekend)
        assert 'holidays' in X.columns
        assert 'is_weekend' in X.columns

    def test_exog_builder_with_config(self):
        """
        Example: Building features using config periods.

        ```python
        from spotforecast2 import Config
        from spotforecast2_safe.preprocessing import ExogBuilder
        import pandas as pd

        config = Config()
        builder = ExogBuilder(
            periods=config.periods,
            country_code=config.API_COUNTRY_CODE
        )
        X = builder.build(
            start=pd.Timestamp('2025-12-31', tz='UTC'),
            end=pd.Timestamp('2026-01-01', tz='UTC')
        )
        print(f"Generated {X.shape[1]} features for {X.shape[0]} hours")
        ```
        """
        from spotforecast2 import Config
        from spotforecast2_safe.preprocessing import ExogBuilder
        import pandas as pd

        config = Config()
        builder = ExogBuilder(
            periods=config.periods,
            country_code=config.API_COUNTRY_CODE
        )
        X = builder.build(
            pd.Timestamp('2025-12-31', tz='UTC'),
            pd.Timestamp('2026-01-01', tz='UTC')
        )

        # 5 periods with different n_periods + 2 (holiday, weekend)
        expected_features = sum(p.n_periods for p in config.periods) + 2
        assert X.shape[0] == 25  # 25 hours (inclusive end date)
        assert X.shape[1] == expected_features


class TestRepeatingBasisFunctionExamples:
    """Tests for RepeatingBasisFunction examples in documentation."""

    def test_rbf_basic(self):
        """
        Example: Creating cyclical features with RBF.

        ```python
        from spotforecast2_safe.preprocessing import RepeatingBasisFunction
        import pandas as pd

        rbf = RepeatingBasisFunction(
            n_periods=12,
            column='hour',
            input_range=(1, 24)
        )

        df = pd.DataFrame({'hour': range(1, 25)})
        features = rbf.transform(df)
        print(features.shape)  # (24, 12)
        ```
        """
        from spotforecast2_safe.preprocessing import RepeatingBasisFunction
        import pandas as pd

        rbf = RepeatingBasisFunction(
            n_periods=12,
            column='hour',
            input_range=(1, 24)
        )

        df = pd.DataFrame({'hour': range(1, 25)})
        features = rbf.transform(df)

        assert features.shape == (24, 12)


class TestForecasterModelExamples:
    """Tests for ForecasterRecursiveLGBM/XGB examples in documentation."""

    def test_forecaster_lgbm_creation(self):
        """
        Example: Creating a LightGBM forecaster.

        ```python
        from spotforecast2.tasks.task_entsoe import ForecasterRecursiveLGBM, config

        model = ForecasterRecursiveLGBM(iteration=1)

        print(model.name)             # 'lgbm'
        print(model.random_state)     # 314159 (from config)
        print(len(model.preprocessor.periods))  # 5 (from config)
        ```
        """
        from spotforecast2.tasks.task_entsoe import ForecasterRecursiveLGBM, config

        model = ForecasterRecursiveLGBM(iteration=1)

        assert model.name == "lgbm"
        assert model.random_state == config.random_state
        assert len(model.preprocessor.periods) == len(config.periods)

    def test_forecaster_xgb_creation(self):
        """
        Example: Creating an XGBoost forecaster.

        ```python
        from spotforecast2.tasks.task_entsoe import ForecasterRecursiveXGB, config

        model = ForecasterRecursiveXGB(iteration=1, lags=24)

        print(model.name)  # 'xgb'
        ```
        """
        from spotforecast2.tasks.task_entsoe import ForecasterRecursiveXGB, config

        model = ForecasterRecursiveXGB(iteration=1, lags=24)

        assert model.name == "xgb"
        assert model.random_state == config.random_state

    def test_forecaster_custom_config(self):
        """
        Example: Creating a forecaster with custom configuration.

        ```python
        from spotforecast2.tasks.task_entsoe import ForecasterRecursiveLGBM
        from spotforecast2_safe.data import Period

        custom_periods = [
            Period(name='hourly', n_periods=24, column='hour', input_range=(1, 24)),
        ]

        model = ForecasterRecursiveLGBM(
            iteration=1,
            lags=48,
            periods=custom_periods,
            country_code='FR',
            random_state=42
        )

        print(len(model.preprocessor.periods))  # 1
        print(model.preprocessor.country_code)  # 'FR'
        ```
        """
        from spotforecast2.tasks.task_entsoe import ForecasterRecursiveLGBM
        from spotforecast2_safe.data import Period

        custom_periods = [
            Period(name='hourly', n_periods=24, column='hour', input_range=(1, 24)),
        ]

        model = ForecasterRecursiveLGBM(
            iteration=1,
            lags=48,
            periods=custom_periods,
            country_code='FR',
            random_state=42
        )

        assert len(model.preprocessor.periods) == 1
        assert model.preprocessor.country_code == 'FR'
        assert model.random_state == 42


class TestLinearlyInterpolateTSExamples:
    """Tests for LinearlyInterpolateTS examples in documentation."""

    def test_interpolation_basic(self):
        """
        Example: Interpolating missing values in time series.

        ```python
        from spotforecast2_safe.preprocessing import LinearlyInterpolateTS
        import pandas as pd
        import numpy as np

        ts = pd.Series(
            [1.0, np.nan, 3.0, np.nan, 5.0],
            index=pd.date_range('2025-01-01', periods=5, freq='h')
        )

        interpolator = LinearlyInterpolateTS()
        ts_clean = interpolator.fit_transform(ts)

        print(ts_clean.values)  # [1.0, 2.0, 3.0, 4.0, 5.0]
        ```
        """
        from spotforecast2_safe.preprocessing import LinearlyInterpolateTS
        import pandas as pd
        import numpy as np

        ts = pd.Series(
            [1.0, np.nan, 3.0, np.nan, 5.0],
            index=pd.date_range('2025-01-01', periods=5, freq='h')
        )

        interpolator = LinearlyInterpolateTS()
        ts_clean = interpolator.fit_transform(ts)

        assert not ts_clean.isna().any()
        assert list(ts_clean.values) == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestDataHomeExamples:
    """Tests for data home path examples in documentation."""

    def test_get_data_home(self):
        """
        Example: Getting the data home directory.

        ```python
        from spotforecast2_safe.data import get_data_home

        data_home = get_data_home()
        print(data_home)  # ~/spotforecast2_data or SPOTFORECAST2_DATA
        ```
        """
        from spotforecast2_safe.data import get_data_home

        data_home = get_data_home()

        assert data_home is not None
        assert isinstance(data_home, (str, Path))
