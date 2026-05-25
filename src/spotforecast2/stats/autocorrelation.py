# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations
from importlib.util import find_spec
import numpy as np
import pandas as pd


def calculate_lag_autocorrelation(
    data: pd.Series | pd.DataFrame,
    n_lags: int = 50,
    last_n_samples: int | None = None,
    sort_by: str = "partial_autocorrelation_abs",
    acf_kwargs: dict[str, object] | None = None,
    pacf_kwargs: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    Calculate autocorrelation and partial autocorrelation for a time series.

    This is a wrapper around statsmodels.acf and statsmodels.pacf.

    Args:
        data: Time series to calculate autocorrelation. If a DataFrame is provided,
            it must have exactly one column.
        n_lags: Number of lags to calculate autocorrelation. Default is 50.
        last_n_samples: Number of most recent samples to use. If None, use the entire
            series. Note that partial correlations can only be computed for lags up to
            50% of the sample size. For example, if the series has 10 samples,
            n_lags must be less than or equal to 5. This parameter is useful
            to speed up calculations when the series is very long. Default is None.
        sort_by: Sort results by lag, partial_autocorrelation_abs,
            partial_autocorrelation, autocorrelation_abs or autocorrelation.
            Default is partial_autocorrelation_abs.
        acf_kwargs: Optional arguments to pass to statsmodels.tsa.stattools.acf.
            Default is {}.
        pacf_kwargs: Optional arguments to pass to statsmodels.tsa.stattools.pacf.
            Default is {}.

    Returns:
        DataFrame with columns: lag, partial_autocorrelation_abs,
            partial_autocorrelation, autocorrelation_abs, autocorrelation.

    Raises:
        TypeError: If data is not a pandas Series or DataFrame with a single column.
        ValueError: If data is a DataFrame with more than one column.
        TypeError: If n_lags is not a positive integer.
        TypeError: If last_n_samples is not None and not a positive integer.
        ValueError: If sort_by is not one of the valid options.

    Examples:
        Calculate autocorrelation for a simple Series:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.stats.autocorrelation import calculate_lag_autocorrelation

        rng = np.random.default_rng(0)
        data = pd.Series(rng.standard_normal(40).cumsum())
        result = calculate_lag_autocorrelation(data=data, n_lags=4)
        print(result)
        assert result.shape == (4, 5)
        assert list(result.columns) == [
            "lag",
            "partial_autocorrelation_abs",
            "partial_autocorrelation",
            "autocorrelation_abs",
            "autocorrelation",
        ]
        ```

        Calculate autocorrelation using only the last 20 samples:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.stats.autocorrelation import calculate_lag_autocorrelation

        rng = np.random.default_rng(0)
        data = pd.Series(rng.standard_normal(40).cumsum())
        result = calculate_lag_autocorrelation(
            data=data,
            n_lags=3,
            last_n_samples=20,
        )
        print(result)
        assert result.shape == (3, 5)
        ```

        Calculate autocorrelation from a DataFrame with a single column:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.stats.autocorrelation import calculate_lag_autocorrelation

        rng = np.random.default_rng(0)
        data = pd.DataFrame({"value": rng.standard_normal(40).cumsum()})
        result = calculate_lag_autocorrelation(data=data, n_lags=4)
        print(result)
        assert result.shape == (4, 5)
        ```

        Sort results by autocorrelation in descending order:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.stats.autocorrelation import calculate_lag_autocorrelation

        rng = np.random.default_rng(0)
        data = pd.Series(rng.standard_normal(40).cumsum())
        result = calculate_lag_autocorrelation(
            data=data,
            n_lags=4,
            sort_by="autocorrelation",
        )
        print(result[["lag", "autocorrelation"]])
        assert result["autocorrelation"].iloc[0] >= result["autocorrelation"].iloc[-1]
        ```

    """
    if find_spec("statsmodels") is None:
        raise ImportError(
            "'statsmodels' is required for calculate_lag_autocorrelation. "
            "Install it with: pip install 'spotforecast2[stats]'"
        )
    from statsmodels.tsa.stattools import acf, pacf

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"`data` must be a pandas Series or a DataFrame with a single column. "
            f"Got {type(data)}."
        )
    if isinstance(data, pd.DataFrame) and data.shape[1] != 1:
        raise ValueError(
            f"If `data` is a DataFrame, it must have exactly one column. "
            f"Got {data.shape[1]} columns."
        )
    if not isinstance(n_lags, int) or n_lags <= 0:
        raise TypeError(f"`n_lags` must be a positive integer. Got {n_lags}.")

    if last_n_samples is not None:
        if not isinstance(last_n_samples, int) or last_n_samples <= 0:
            raise TypeError(
                f"`last_n_samples` must be a positive integer. Got {last_n_samples}."
            )
        data = data.iloc[-last_n_samples:]

    if sort_by not in [
        "lag",
        "partial_autocorrelation_abs",
        "partial_autocorrelation",
        "autocorrelation_abs",
        "autocorrelation",
    ]:
        raise ValueError(
            "`sort_by` must be 'lag', 'partial_autocorrelation_abs', 'partial_autocorrelation', "
            "'autocorrelation_abs' or 'autocorrelation'."
        )

    series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    if series.nunique() <= 1:
        acf_values = np.full(n_lags + 1, np.nan)
        acf_values[0] = 1.0
        pacf_values = np.zeros(n_lags + 1)
        pacf_values[0] = 1.0
    else:
        pacf_kwargs_ = pacf_kwargs.copy() if pacf_kwargs is not None else {}
        acf_kwargs_ = acf_kwargs.copy() if acf_kwargs is not None else {}
        pacf_values = pacf(data, nlags=n_lags, **pacf_kwargs_)
        acf_values = acf(data, nlags=n_lags, **acf_kwargs_)

    results = pd.DataFrame(
        {
            "lag": range(n_lags + 1),
            "partial_autocorrelation_abs": np.abs(pacf_values),
            "partial_autocorrelation": pacf_values,
            "autocorrelation_abs": np.abs(acf_values),
            "autocorrelation": acf_values,
        }
    ).iloc[1:]

    if sort_by == "lag":
        results = results.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    else:
        results = results.sort_values(by=sort_by, ascending=False).reset_index(
            drop=True
        )

    return results
