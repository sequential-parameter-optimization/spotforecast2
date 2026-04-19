# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

from importlib.util import find_spec

from spotforecast2_safe.exceptions import set_skforecast_warnings
from spotforecast2_safe.forecaster.utils import (
    check_exog,
    check_exog_dtypes,
    check_extract_values_and_index,
    check_interval,
    check_predict_input,
    check_preprocess_exog_multiseries,
    check_preprocess_series,
    check_residuals_input,
    check_select_fit_kwargs,
    check_y,
    date_to_index_position,
    exog_to_direct,
    exog_to_direct_numpy,
    expand_index,
    get_exog_dtypes,
    get_style_repr_html,
    initialize_estimator,
    initialize_lags,
    initialize_transformer_series,
    initialize_weights,
    initialize_window_features,
    input_to_frame,
    predict_multivariate,
    prepare_steps_direct,
    select_n_jobs_fit_forecaster,
    transform_dataframe,
    transform_numpy,
)

optional_dependencies = {
    "stats": ["statsmodels>=0.12, <0.16"],
    "deeplearning": [
        "matplotlib>=3.10.8",
    ],
    "plotting": [
        "matplotlib>=3.10.8",
        "seaborn>=0.11, <0.15",
        "statsmodels>=0.12, <0.16",
    ],
}


def _find_optional_dependency(
    package_name: str,
    optional_dependencies: dict[str, list[str]] = optional_dependencies,
) -> tuple[str, str] | None:
    """
    Find if a package is an optional dependency. If True, find the version and
    the extension it belongs to.
    """
    for extra, packages in optional_dependencies.items():
        package_version = [package for package in packages if package_name in package]
        if package_version:
            return extra, package_version[0]

    return None


def check_optional_dependency(package_name: str) -> None:
    """
    Check if an optional dependency is installed, if not raise an ImportError
    with installation instructions.

    Args:
        package_name (str): Name of the package to check.

    Raises:
        ImportError: If the package is not installed.
    """
    if find_spec(package_name) is None:
        try:
            _, _ = _find_optional_dependency(package_name=package_name)
            msg = (
                f"\n'{package_name}' is an optional dependency not included in the "
                f"default spotforecast installation."
            )
        except Exception:
            msg = f"\n'{package_name}' is needed but not installed. Please install it."

        raise ImportError(msg)


__all__ = [
    "check_exog",
    "check_exog_dtypes",
    "check_extract_values_and_index",
    "check_interval",
    "check_optional_dependency",
    "check_predict_input",
    "check_preprocess_exog_multiseries",
    "check_preprocess_series",
    "check_residuals_input",
    "check_select_fit_kwargs",
    "check_y",
    "date_to_index_position",
    "exog_to_direct",
    "exog_to_direct_numpy",
    "expand_index",
    "get_exog_dtypes",
    "get_style_repr_html",
    "initialize_estimator",
    "initialize_lags",
    "initialize_transformer_series",
    "initialize_weights",
    "initialize_window_features",
    "input_to_frame",
    "predict_multivariate",
    "prepare_steps_direct",
    "select_n_jobs_fit_forecaster",
    "set_skforecast_warnings",
    "transform_dataframe",
    "transform_numpy",
]
