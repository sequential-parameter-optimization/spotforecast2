# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke tests verifying that weather classes are consumed from spotforecast2_safe.

Since the spotforecast2.weather subpackage has been removed, all weather
functionality is provided directly by spotforecast2_safe.weather.
"""


class TestWeatherNotInSpotforecast2:
    """The spotforecast2.weather subpackage no longer exists."""

    def test_no_weather_subpackage(self):
        """spotforecast2.weather must not be importable as a local subpackage."""
        import importlib.util

        spec = importlib.util.find_spec("spotforecast2.weather")
        assert (
            spec is None
        ), "spotforecast2.weather still exists — the subpackage was not removed."

    def test_spotforecast2_package_imports_cleanly(self):
        """spotforecast2 top-level package imports without error."""
        import spotforecast2  # noqa: F401


class TestWeatherAvailableFromSafe:
    """WeatherClient and WeatherService are available via spotforecast2_safe.weather."""

    def test_import_weather_client_from_safe(self):
        """WeatherClient is importable from spotforecast2_safe.weather."""
        from spotforecast2_safe.weather import WeatherClient  # noqa: F401

    def test_import_weather_service_from_safe(self):
        """WeatherService is importable from spotforecast2_safe.weather."""
        from spotforecast2_safe.weather import WeatherService  # noqa: F401

    def test_weather_service_is_subclass(self):
        """WeatherService is a subclass of WeatherClient in spotforecast2_safe."""
        from spotforecast2_safe.weather import WeatherClient, WeatherService

        assert issubclass(WeatherService, WeatherClient)
