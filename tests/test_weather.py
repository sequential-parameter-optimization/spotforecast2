import pytest
import pandas as pd
from unittest.mock import patch
from spotforecast2.weather.weather_client import WeatherClient, WeatherService


@pytest.fixture
def mock_weather_response():
    """Mock API response."""
    return {
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "temperature_2m": [10.0, 11.0],
            "relative_humidity_2m": [50, 55],
            # Add other fields as needed for strict check,
            # but client checks specific columns existence
        }
    }


def test_weather_client_fetch(mock_weather_response):
    """Test standard fetch mechanic."""
    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_weather_response

        client = WeatherClient(latitude=50.0, longitude=10.0)
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-01-01 01:00")

        df = client.fetch_archive(start, end)

        assert len(df) == 2
        assert "temperature_2m" in df.columns
        assert df.index[0] == pd.Timestamp("2023-01-01 00:00")


def test_weather_service_caching(tmp_path, mock_weather_response):
    """Test caching logic in WeatherService."""
    cache_file = tmp_path / "cache.parquet"

    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_weather_response

        service = WeatherService(50.0, 10.0, cache_path=cache_file)
        start = "2023-01-01 00:00"
        end = "2023-01-01 01:00"

        # First call: hits API (mock)
        df1 = service.get_dataframe(start, end, timezone="UTC")
        assert mock_get.call_count >= 1
        assert cache_file.exists()

        # Second call: hits cache (mock shouldn't be called again for this range logic ideally,
        # but implementation might check API if range not fully covered?
        # Our implementation checks if cache fully covers range.)

        mock_get.reset_mock()
        df2 = service.get_dataframe(start, end, timezone="UTC")

        # Should not hit API if cache covers it
        assert mock_get.call_count == 0
        pd.testing.assert_frame_equal(df1, df2)


def test_fallback_logic(tmp_path):
    """Test fallback when API fails."""
    # Create valid cache for "yesterday"
    cache_file = tmp_path / "cache.parquet"
    idx = pd.date_range("2023-01-01 00:00", periods=24, freq="h", tz="UTC")
    df_cache = pd.DataFrame({"temperature_2m": range(24)}, index=idx)
    df_cache.to_parquet(cache_file)

    with patch(
        "spotforecast2.weather.weather_client.WeatherClient._fetch"
    ) as mock_fetch:
        mock_fetch.side_effect = Exception("API Error")

        service = WeatherService(50.0, 10.0, cache_path=cache_file)

        # Request "tomorrow" (needs API, which fails) -> Fallback
        start = "2023-01-02 00:00"
        end = "2023-01-02 05:00"

        df_fallback = service.get_dataframe(
            start, end, timezone="UTC", fallback_on_failure=True
        )

        assert len(df_fallback) == 6  # 0 to 5 inclusive is 6 hours
        # Check values are repeated from end of cache
        # Cache end was 23 (range(24)).
        # Fallback repeats last 24h.
        assert df_fallback.iloc[0]["temperature_2m"] == 0  # Should cycle?
        # Logic: concat([last_24] * repeats).
        # last_24 is 0..23.
        # So new data starts at 0. Correct.
