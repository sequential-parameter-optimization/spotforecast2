# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify sf2 picks up sf2-safe's exogenous-provider registry.

sf2 reuses ``spotforecast2_safe`` for all configuration and exogenous-feature
code (it has no own copies). These tests confirm that the provider registry
added in sf2-safe is reachable through sf2 and that sf2's
``ForecasterRecursiveModelFull`` forwards the provider flags into its inherited
``ExogBuilder`` via ``from_config``.

The whole module is skipped when the installed ``spotforecast2-safe`` predates
the provider registry, so sf2's CI stays green until the dependency floor is
raised to a release that ships it (see the rc-marker note in ``pyproject.toml``).
"""

import pytest

providers = pytest.importorskip(
    "spotforecast2_safe.preprocessing.exog_providers",
    reason="installed spotforecast2-safe has no exog-provider registry yet",
)

from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe  # noqa: E402
from spotforecast2_safe.configurator.config_multi import ConfigMulti  # noqa: E402

EXPECTED_FLAGS = {
    "include_covid_infection_rate",
    "include_entsoe_forecast_load",
    "include_entsoe_renewable_forecast",
    "include_entsoe_net_load",
    "include_entsoe_day_ahead_price",
    "include_football_match_window",
    "include_energy_saving_window",
}


def test_registry_reexported_from_preprocessing():
    from spotforecast2_safe.preprocessing import (
        EXOG_PROVIDER_REGISTRY,
        CovidInfectionRateProvider,
        EntsoeDayAheadPriceProvider,
        EntsoeForecastLoadProvider,
        EntsoeNetLoadProvider,
        EntsoeRenewableForecastProvider,
        ExogFeatureProvider,
        build_providers,
        build_providers_from_config,
    )

    # Superset, not equality: additive provider releases in sf2-safe must not
    # break sf2's suite; removals of expected flags still fail.
    assert EXPECTED_FLAGS <= set(EXOG_PROVIDER_REGISTRY)
    assert issubclass(CovidInfectionRateProvider, ExogFeatureProvider)
    assert callable(build_providers) and callable(build_providers_from_config)
    # silence unused-import linters for the re-export assertions
    assert all(
        cls is not None
        for cls in (
            EntsoeForecastLoadProvider,
            EntsoeRenewableForecastProvider,
            EntsoeNetLoadProvider,
            EntsoeDayAheadPriceProvider,
        )
    )


@pytest.mark.parametrize("config_cls", [ConfigEntsoe, ConfigMulti])
def test_provider_flags_present_on_configs(config_cls):
    cfg = config_cls()
    for flag in EXPECTED_FLAGS:
        assert getattr(cfg, flag) is False, flag
    assert cfg.on_exog_provider_failure == "raise"


def test_full_model_forwards_flags_into_preprocessor():
    from spotforecast2.models.forecaster_recursive_model_full import (
        ForecasterRecursiveModelFull,
    )

    cfg = ConfigEntsoe(
        include_covid_infection_rate=True,
        include_entsoe_forecast_load=True,
    )
    model = ForecasterRecursiveModelFull.from_config(iteration=0, config=cfg)
    names = [p.name for p in model.preprocessor.providers]
    assert "covid_infection_rate" in names
    assert "entsoe_forecasted_load" in names
