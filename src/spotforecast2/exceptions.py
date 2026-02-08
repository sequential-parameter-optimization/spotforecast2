# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""Custom exceptions and warnings for spotforecast2.

This module contains all the custom warnings and error classes used
across spotforecast2.

Examples:
    Using custom warnings::

        import warnings
        from spotforecast2.exceptions import MissingValuesWarning

        # Raise a warning
        warnings.warn(
            "Missing values detected in input data.",
            MissingValuesWarning
        )

        # Suppress a specific warning
        warnings.simplefilter('ignore', category=MissingValuesWarning)
"""

import warnings
import textwrap
from spotforecast2_safe.exceptions import set_warnings_style

__all__ = [
    "DataTypeWarning",
    "DataTransformationWarning",
    "ExogenousInterpretationWarning",
    "FeatureOutOfRangeWarning",
    "IgnoredArgumentWarning",
    "InputTypeWarning",
    "LongTrainingWarning",
    "MissingExogWarning",
    "MissingValuesWarning",
    "OneStepAheadValidationWarning",
    "ResidualsUsageWarning",
    "UnknownLevelWarning",
    "SaveLoadSkforecastWarning",
    "SpotforecastVersionWarning",
    "NotFittedError",
]


from spotforecast2_safe.exceptions import (
    DataTypeWarning,
    DataTransformationWarning,
    ExogenousInterpretationWarning,
    FeatureOutOfRangeWarning,
    IgnoredArgumentWarning,
    InputTypeWarning,
    LongTrainingWarning,
    MissingExogWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning,
    ResidualsUsageWarning,
    UnknownLevelWarning,
    SaveLoadSkforecastWarning,
    SpotforecastVersionWarning,
    NotFittedError,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


warn_skforecast_categories = [
    DataTypeWarning,
    DataTransformationWarning,
    ExogenousInterpretationWarning,
    FeatureOutOfRangeWarning,
    IgnoredArgumentWarning,
    InputTypeWarning,
    LongTrainingWarning,
    MissingExogWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning,
    ResidualsUsageWarning,
    UnknownLevelWarning,
    SaveLoadSkforecastWarning,
    SpotforecastVersionWarning,
]


def format_warning_handler(
    message: str,
    category: str,
    filename: str,
    lineno: str,
    file: object = None,
    line: str = None,
) -> None:
    """Custom warning handler to format warnings in a box.

    Args:
        message: Warning message.
        category: Warning category.
        filename: Filename where the warning was raised.
        lineno: Line number where the warning was raised.
        file: File where the warning was raised.
        line: Line where the warning was raised.

    Returns:
        None
    """

    if isinstance(message, tuple(warn_skforecast_categories)):
        width = 88
        title = type(message).__name__
        output_text = ["\n"]

        wrapped_message = textwrap.fill(
            str(message), width=width - 2, expand_tabs=True, replace_whitespace=True
        )
        title_top_border = f"╭{'─' * ((width - len(title) - 2) // 2)} {title} {'─' * ((width - len(title) - 2) // 2)}╮"
        if len(title) % 2 != 0:
            title_top_border = title_top_border[:-1] + "─" + "╮"
        bottom_border = f"╰{'─' * width}╯"
        output_text.append(title_top_border)

        for line in wrapped_message.split("\n"):
            output_text.append(f"│ {line.ljust(width - 2)} │")

        output_text.append(bottom_border)
        output_text = "\n".join(output_text)
        color = "\033[38;5;208m"
        reset = "\033[0m"
        output_text = f"{color}{output_text}{reset}"
        print(output_text)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


def rich_warning_handler(
    message: str,
    category: str,
    filename: str,
    lineno: str,
    file: object = None,
    line: str = None,
) -> None:
    """Custom handler for warnings that uses rich to display formatted panels.

    Args:
        message: Warning message.
        category: Warning category.
        filename: Filename where the warning was raised.
        lineno: Line number where the warning was raised.
        file: File where the warning was raised.
        line: Line where the warning was raised.

    Returns:
        None
    """

    if isinstance(message, tuple(warn_skforecast_categories)):
        if not HAS_RICH:
            # Fallback to format_warning_handler if rich is not available
            format_warning_handler(message, category, filename, lineno, file, line)
            return

        console = Console()

        category_name = category.__name__
        text = (
            f"{message.message}\n\n"
            f"Category : spotforecast2.exceptions.{category_name}\n"
            f"Location : {filename}:{lineno}\n"
            f"Suppress : warnings.simplefilter('ignore', category={category_name})"
        )

        panel = Panel(
            Text(text, justify="left"),
            title=category_name,
            title_align="center",
            border_style="color(214)",
            width=88,
        )

        console.print(panel)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


set_warnings_style(style="skforecast")
