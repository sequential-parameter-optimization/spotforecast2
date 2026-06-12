# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from importlib.util import find_spec

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


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


def select_pacf_lags(
    series: pd.Series,
    *,
    n_lags: int = 200,
    top_k: int = 8,
    fallback: list[int] | None = None,
) -> list[int]:
    """Select the most informative lags using the partial autocorrelation function.

    Computes the PACF up to `n_lags` via `calculate_lag_autocorrelation`, then
    returns the `top_k` lags whose `|PACF|` exceeds the 95 % significance band
    (1.96 / sqrt(N)), sorted in ascending order.

    This is a pure-compute helper ported from the operational team-4 pipeline
    (`select_key_lags` in ``bart26k-lecture/scripts/team4_4zones_submit.py``).
    No plotting, no side effects beyond an optional DEBUG log line.

    Args:
        series: The time series from which to estimate lags. Must contain at
            least `2 * n_lags + 1` observations for statsmodels PACF to run
            without truncating `n_lags`.
        n_lags: Number of lags passed to `calculate_lag_autocorrelation`.
            Default is 200.
        top_k: Maximum number of lags to return (the `top_k` significant lags
            ranked by descending `|PACF|`). Default is 8.
        fallback: Lag list returned when no lag exceeds the significance band
            (degenerate series: too short, nearly constant, or `n_lags` too
            large). If `None`, a `ValueError` is raised instead.

    Returns:
        Sorted list of lag integers (ascending). Length is at most `top_k`;
        may be shorter if fewer than `top_k` significant lags exist.

    Raises:
        ValueError: If no lag exceeds the significance band and `fallback` is
            `None`. Pass `fallback=[1, 2, 24, 168]` (or any operator-chosen
            constant) to suppress the error and use a safe default instead.

    Examples:
        Select significant lags from a synthetic AR(1) series:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.stats.autocorrelation import select_pacf_lags

        rng = np.random.default_rng(42)
        n = 24 * 120  # 120 days of hourly data
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.7 * ar[t - 1] + rng.standard_normal()
        series = pd.Series(ar)
        lags = select_pacf_lags(series, n_lags=50, top_k=8)
        print("selected lags:", lags)
        assert isinstance(lags, list)
        assert all(isinstance(x, int) for x in lags)
        assert lags == sorted(lags)
        assert len(lags) <= 8
        assert 1 in lags, "lag-1 AR component should be selected"
        ```

        Select significant lags from an AR(24) process — lag 24 dominates:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2.stats.autocorrelation import select_pacf_lags

        rng = np.random.default_rng(42)
        n = 24 * 200
        ar = np.zeros(n)
        for t in range(24, n):
            ar[t] = 0.8 * ar[t - 24] + rng.standard_normal()
        series = pd.Series(ar)
        lags = select_pacf_lags(series, n_lags=50, top_k=8)
        print("selected lags:", lags)
        assert 24 in lags, f"lag 24 expected in {lags}"
        ```

        Degenerate (constant) series — fallback is returned when provided:

        ```{python}
        import pandas as pd
        from spotforecast2.stats.autocorrelation import select_pacf_lags

        series = pd.Series([1.0] * 50)
        result = select_pacf_lags(series, n_lags=10, fallback=[1, 2, 24])
        print("fallback lags:", result)
        assert result == [1, 2, 24]
        ```

        Degenerate series with no fallback raises ValueError:

        ```{python}
        import pytest
        import pandas as pd
        from spotforecast2.stats.autocorrelation import select_pacf_lags

        series = pd.Series([1.0] * 50)
        with pytest.raises(ValueError, match="no significant"):
            select_pacf_lags(series, n_lags=10, fallback=None)
        ```

    """
    acf_df = calculate_lag_autocorrelation(series, n_lags=n_lags)
    conf = 1.96 / np.sqrt(len(series))
    significant = acf_df[acf_df["partial_autocorrelation_abs"] > conf]
    top = significant.nlargest(top_k, "partial_autocorrelation_abs")
    key_lags = sorted(top["lag"].astype(int).tolist())
    _logger.debug(
        "select_pacf_lags: N=%d band=+/-%.4f significant=%d top_%d=%s",
        len(series),
        conf,
        len(significant),
        top_k,
        key_lags,
    )
    if not key_lags:
        if fallback is not None:
            return list(fallback)
        raise ValueError(
            f"select_pacf_lags: no significant PACF lags found "
            f"(N={len(series)}, band={conf:.4f}, n_lags={n_lags}). "
            "Pass fallback=<list> to use a safe default instead of raising."
        )
    return key_lags
