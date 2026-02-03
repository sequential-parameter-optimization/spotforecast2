import pandas as pd
from spotforecast2.data.data import Data
from pathlib import Path
from os import environ
from typing import Optional, Union
from spotforecast2.utils.generate_holiday import create_holiday_df
from pandas import Timestamp
from spotforecast2.weather.weather_client import WeatherService


def get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    """Return the location where datasets are to be stored.

    By default the data directory is set to a folder named 'spotforecast2_data' in the
    user home folder. Alternatively, it can be set by the 'SPOTFORECAST2_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Args:
        data_home (str or pathlib.Path, optional):
            The path to spotforecast data directory. If `None`, the default path
            is `~/spotforecast2_data`.

    Returns:
        data_home (pathlib.Path):
            The path to the spotforecast data directory.
    Examples:
        >>> from pathlib import Path
        >>> get_data_home()
        PosixPath('/home/user/spotforecast2_data')
        >>> get_data_home(Path('/tmp/spotforecast2_data'))
        PosixPath('/tmp/spotforecast2_data')
    """
    if data_home is None:
        data_home = environ.get(
            "SPOTFORECAST2_DATA", Path.home() / "spotforecast2_data"
        )
    # Ensure data_home is a Path() object pointing to an absolute path
    data_home = Path(data_home).expanduser().absolute()
    # Create data directory if it does not exists.
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def fetch_data(
    filename: str = "data_in.csv",
    columns: Optional[list] = None,
    index_col: int = 0,
    parse_dates: bool = True,
    dayfirst: bool = False,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetches the integrated raw dataset from a CSV file.

    Args:
        filename (str):
            Filename of the CSV file containing the dataset. It must be located in the data home directory, which can be get or set using `get_data_home()`.
        columns (list, optional):
            List of columns to be included in the dataset. If None, all columns are included.
            If an empty list is provided, a ValueError is blocked.
        index_col (int):
            Column index to be used as the index.
        parse_dates (bool):
            Whether to parse dates in the index column.
        dayfirst (bool):
            Whether the day comes first in date parsing.
        timezone (str):
            Timezone to set for the datetime index.

    Returns:
        pd.DataFrame: The integrated raw dataset.

    Raises:
        ValueError: If columns is an empty list.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_data
        >>> data = fetch_data(columns=["col1", "col2"])
        >>> data.head()
                        Header1  Header2  Header3
    """
    if columns is not None and len(columns) == 0:
        raise ValueError("columns must be specified and cannot be empty.")

    csv_path = get_data_home() / filename
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    dataset = Data.from_csv(
        csv_path=csv_path,
        index_col=index_col,
        parse_dates=parse_dates,
        dayfirst=dayfirst,
        timezone=timezone,
        columns=columns,
    )

    return dataset.data


def fetch_holiday_data(
    start: str | Timestamp,
    end: str | Timestamp,
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Fetches holiday data for the dataset period.

    Args:
        start (str or pd.Timestamp):
            Start date of the dataset period.
        end (str or pd.Timestamp):
            End date of the dataset period.
        tz (str):
            Timezone for the holiday data.
        freq (str):
            Frequency of the holiday data.
        country_code (str):
            Country code for the holidays.
        state (str):
            State code for the holidays.

    Returns:
        pd.DataFrame: DataFrame containing holiday information.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_holiday_data
        >>> holiday_df = fetch_holiday_data(
        ...     start='2023-01-01T00:00',
        ...     end='2023-01-10T00:00',
        ...     tz='UTC',
        ...     freq='h',
        ...     country_code='DE',
        ...     state='NW'
        ... )
        >>> holiday_df.head()
                        is_holiday
    """

    holiday_df = create_holiday_df(
        start=start, end=end, tz=tz, freq=freq, country_code=country_code, state=state
    )
    return holiday_df


def fetch_weather_data(
    cov_start: str,
    cov_end: str,
    latitude: float = 51.5136,
    longitude: float = 7.4653,
    timezone: str = "UTC",
    freq: str = "h",
    fallback_on_failure: bool = True,
    cached=True,
) -> pd.DataFrame:
    """Fetches weather data for the dataset period plus forecast horizon.
        Create weather dataframe using API with optional caching.
    Args:
        cov_start (str):
            Start date for covariate data.
        cov_end (str):
            End date for covariate data.
        latitude (float):
            Latitude of the location for weather data. Default is 51.5136 (Dortmund).
        longitude (float):
            Longitude of the location for weather data. Default is 7.4653 (Dortmund).
        timezone (str):
            Timezone for the weather data.
        freq (str):
            Frequency of the weather data.
        fallback_on_failure (bool):
            Whether to use fallback data in case of failure.

    Returns:
        pd.DataFrame: DataFrame containing weather information.

    Examples:
        >>> from spotforecast2.data.fetch_data import fetch_weather_data
        >>> weather_df = fetch_weather_data(
        ...     cov_start='2023-01-01T00:00',
        ...     cov_end='2023-01-11T00:00',
        ...     latitude=51.5136,
        ...     longitude=7.4653,
        ...     timezone='UTC',
        ...     freq='h',
        ...     fallback_on_failure=True,
        ...     cached=True
        ... )
        >>> weather_df.head()
    """
    if cached:
        cache_path = get_data_home() / "weather_cache.parquet"
    else:
        cache_path = None

    service = WeatherService(
        latitude=latitude, longitude=longitude, cache_path=cache_path
    )

    weather_df = service.get_dataframe(
        start=cov_start,
        end=cov_end,
        timezone=timezone,
        freq=freq,
        fallback_on_failure=fallback_on_failure,
    )
    return weather_df
