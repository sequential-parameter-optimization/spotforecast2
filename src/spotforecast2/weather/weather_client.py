# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weather data fetching and processing using Open-Meteo API."""

from spotforecast2_safe.weather.weather_client import (
    WeatherClient,
    WeatherService,
)

__all__ = ["WeatherClient", "WeatherService"]
