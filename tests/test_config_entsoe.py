# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ConfigEntsoe configuration class.

This test suite validates the ConfigEntsoe configuration class functionality,
including default values, custom initialization, docstring examples, and
import aliases.
"""

import pandas as pd
import pytest

from spotforecast2_safe import Config, ConfigEntsoe
from spotforecast2_safe.data import Period


class TestConfigEntsoeImports:
    """Test imports and aliases."""

    def test_config_alias_is_config_entsoe(self):
        """Verify that Config is an alias for ConfigEntsoe."""
        assert Config is ConfigEntsoe

    def test_config_can_be_imported_from_spotforecast2(self):
        """Verify Config can be imported from main package."""
        from spotforecast2 import Config as ImportedConfig

        config = ImportedConfig()
        assert isinstance(config, ConfigEntsoe)

    def test_config_entsoe_can_be_imported_from_manager(self):
        """Verify ConfigEntsoe can be imported from manager module."""
        from spotforecast2.manager import ConfigEntsoe as ManagerConfig

        config = ManagerConfig()
        assert isinstance(config, ConfigEntsoe)


class TestConfigEntsoeDefaults:
    """Test default configuration values."""

    def test_default_instance_has_expected_values(self):
        """Verify default ConfigEntsoe instance has expected values."""
        config = Config()

        assert config.API_COUNTRY_CODE == "DE"
        assert config.predict_size == 24
        assert config.refit_size == 7
        assert config.random_state == 314159
        assert config.n_hyperparameters_trials == 20
        assert config.end_train_default == "2025-12-31 00:00+00:00"

    def test_default_periods_are_valid(self):
        """Verify default periods list is properly structured."""
        config = Config()

        assert len(config.periods) == 5
        assert config.periods[0].name == "daily"
        assert config.periods[0].n_periods == 12
        assert config.periods[0].column == "hour"
        assert config.periods[0].input_range == (1, 24)

    def test_default_lags_consider(self):
        """Verify default lags_consider range."""
        config = Config()

        assert config.lags_consider == list(range(1, 24))
        assert len(config.lags_consider) == 23

    def test_default_timedeltas(self):
        """Verify default Timedelta values."""
        config = Config()

        assert config.train_size == pd.Timedelta(days=3 * 365)
        assert config.delta_val == pd.Timedelta(hours=24 * 7 * 10)


class TestConfigEntsoeCustomization:
    """Test custom configuration values."""

    def test_custom_api_country_code(self):
        """Verify custom API country code."""
        config = Config(api_country_code="FR")

        assert config.API_COUNTRY_CODE == "FR"

    def test_custom_predict_size(self):
        """Verify custom predict size."""
        config = Config(predict_size=48)

        assert config.predict_size == 48

    def test_custom_random_state(self):
        """Verify custom random state."""
        config = Config(random_state=42)

        assert config.random_state == 42

    def test_custom_periods(self):
        """Verify custom periods list."""
        custom_periods = [
            Period(name="hourly", n_periods=24, column="hour", input_range=(0, 23))
        ]
        config = Config(periods=custom_periods)

        assert len(config.periods) == 1
        assert config.periods[0].name == "hourly"

    def test_custom_lags_consider(self):
        """Verify custom lags_consider list."""
        custom_lags = [1, 2, 3, 24, 48]
        config = Config(lags_consider=custom_lags)

        assert config.lags_consider == custom_lags

    def test_custom_train_size(self):
        """Verify custom train_size."""
        custom_train_size = pd.Timedelta(days=365)
        config = Config(train_size=custom_train_size)

        assert config.train_size == custom_train_size

    def test_multiple_custom_parameters(self):
        """Verify multiple custom parameters at once."""
        config = Config(
            api_country_code="FR",
            predict_size=48,
            refit_size=14,
            random_state=42,
            n_hyperparameters_trials=50,
        )

        assert config.API_COUNTRY_CODE == "FR"
        assert config.predict_size == 48
        assert config.refit_size == 14
        assert config.random_state == 42
        assert config.n_hyperparameters_trials == 50


class TestConfigEntsoeDocstringExamples:
    """Test examples from the docstring to ensure they work correctly."""

    def test_example_default_configuration(self):
        """Test: Use default configuration."""
        from spotforecast2 import Config

        # Use default configuration
        config = Config()
        assert config.API_COUNTRY_CODE == "DE"
        assert config.predict_size == 24
        assert config.random_state == 314159

    def test_example_custom_configuration(self):
        """Test: Create custom configuration."""
        from spotforecast2 import Config

        # Create custom configuration
        custom_config = Config(api_country_code="FR", predict_size=48, random_state=42)
        assert custom_config.API_COUNTRY_CODE == "FR"
        assert custom_config.predict_size == 48

    def test_example_verify_training_window(self):
        """Test: Verify training window."""
        from spotforecast2 import Config

        config = Config()
        # Verify training window
        assert config.train_size == pd.Timedelta(days=3 * 365)

    def test_example_check_default_periods(self):
        """Test: Check default periods."""
        from spotforecast2 import Config

        config = Config()
        # Check default periods
        assert len(config.periods) == 5
        assert config.periods[0].name == "daily"


class TestConfigEntsoePeriod:
    """Test Period dataclass."""

    def test_period_is_frozen(self):
        """Verify Period dataclass is immutable."""
        period = Period(name="test", n_periods=10, column="hour", input_range=(0, 23))

        with pytest.raises(Exception):  # FrozenInstanceError in dataclasses
            period.name = "modified"

    def test_period_creation(self):
        """Verify Period can be created with all required fields."""
        period = Period(name="daily", n_periods=24, column="hour", input_range=(0, 23))

        assert period.name == "daily"
        assert period.n_periods == 24
        assert period.column == "hour"
        assert period.input_range == (0, 23)
