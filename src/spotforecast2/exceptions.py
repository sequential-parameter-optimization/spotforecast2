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
import inspect
from functools import wraps
import textwrap

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
    "runtime_deprecated",
    "set_warnings_style",
    "set_skforecast_warnings",
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


def runtime_deprecated(
    replacement: str = None,
    version: str = None,
    removal: str = None,
    category: type[Warning] = FutureWarning,
) -> object:
    """Decorator to mark functions or classes as deprecated.

    Works for both function and class targets, and ensures warnings are visible
    even inside Jupyter notebooks.

    Args:
        replacement: Name of the replacement function/class to use instead.
        version: Version in which the function/class was deprecated.
        removal: Version in which the function/class will be removed.
        category: Warning category to use. Default is FutureWarning.

    Returns:
        Decorator function.

    Examples:
        >>> @runtime_deprecated(replacement='new_function', version='0.5', removal='1.0')
        ... def old_function():
        ...     pass
        >>> old_function()  # doctest: +SKIP
        FutureWarning: old_function() is deprecated since version 0.5; use new_function instead...
    """

    def decorator(obj):
        is_function = inspect.isfunction(obj) or inspect.ismethod(obj)
        is_class = inspect.isclass(obj)

        if not (is_function or is_class):
            raise TypeError(
                "@runtime_deprecated can only be used on functions or classes"
            )

        # ----- Build warning message -----
        name = obj.__name__
        message = (
            f"{name}() is deprecated" if is_function else f"{name} class is deprecated"
        )
        if version:
            message += f" since version {version}"
        if replacement:
            message += f"; use {replacement} instead"
        if removal:
            message += f". It will be removed in version {removal}."
        else:
            message += "."

        def issue_warning():
            """Emit warning in a way that always shows in notebooks."""
            with warnings.catch_warnings():
                warnings.simplefilter("always", category)
                warnings.warn(message, category, stacklevel=3)

        # ----- Case 1: decorating a function -----
        if is_function:

            @wraps(obj)
            def wrapper(*args, **kwargs):
                issue_warning()
                return obj(*args, **kwargs)

            # Add metadata
            wrapper.__deprecated__ = True
            wrapper.__replacement__ = replacement
            wrapper.__version__ = version
            wrapper.__removal__ = removal
            return wrapper

        # ----- Case 2: decorating a class -----
        elif is_class:
            orig_init = getattr(obj, "__init__", None)
            orig_new = getattr(obj, "__new__", None)

            # Only wrap whichever exists (some classes use __new__, others __init__)
            if orig_new and (orig_new is not object.__new__):

                @wraps(orig_new)
                def wrapped_new(cls, *args, **kwargs):
                    issue_warning()
                    return orig_new(cls, *args, **kwargs)

                obj.__new__ = staticmethod(wrapped_new)

            elif orig_init:

                @wraps(orig_init)
                def wrapped_init(self, *args, **kwargs):
                    issue_warning()
                    return orig_init(self, *args, **kwargs)

                obj.__init__ = wrapped_init

            # Add metadata
            obj.__deprecated__ = True
            obj.__replacement__ = replacement
            obj.__version__ = version
            obj.__removal__ = removal

            return obj

    return decorator


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

    Examples:
        >>> # This is used internally by the warnings module
        >>> set_warnings_style('skforecast')  # doctest: +SKIP
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

    Examples:
        >>> # This is used internally when rich is available
        >>> set_warnings_style('skforecast')  # doctest: +SKIP
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


def set_warnings_style(style: str = "skforecast") -> None:
    """Set the warning handler based on the provided style.

    Args:
        style: The style of the warning handler. Either 'skforecast' or 'default'.

    Returns:
        None

    Examples:
        >>> set_warnings_style('skforecast')
        >>> # Now warnings will be displayed with formatting
        >>> set_warnings_style('default')
        >>> # Back to default Python warning format
    """
    if style == "skforecast":
        if not hasattr(warnings, "_original_showwarning"):
            warnings._original_showwarning = warnings.showwarning
        if HAS_RICH:
            warnings.showwarning = rich_warning_handler
        else:
            warnings.showwarning = format_warning_handler
    else:
        if hasattr(warnings, "_original_showwarning"):
            warnings.showwarning = warnings._original_showwarning


set_warnings_style(style="skforecast")


def set_skforecast_warnings(suppress_warnings: bool, action: str = "ignore") -> None:
    """
    Suppress spotforecast warnings.

    Args:
        suppress_warnings: bool
            If True, spotforecast warnings will be suppressed.
        action: str, default 'ignore'
            Action to take regarding the warnings.
    """
    if suppress_warnings:
        for category in warn_skforecast_categories:
            warnings.simplefilter(action, category=category)
