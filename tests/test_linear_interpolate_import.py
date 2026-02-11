import pytest
from spotforecast2_safe.preprocessing.linearly_interpolate_ts import (
    LinearlyInterpolateTS as LinearFull,
)
from spotforecast2_safe.preprocessing import LinearlyInterpolateTS as LinearPre
from spotforecast2_safe import LinearlyInterpolateTS as LinearTop


def test_import_linear_full():
    """Verify import from spotforecast2_safe.preprocessing.linearly_interpolate_ts."""
    interpolator = LinearFull()
    assert isinstance(interpolator, LinearFull)


def test_import_linear_pre():
    """Verify import from spotforecast2_safe.preprocessing."""
    interpolator = LinearPre()
    assert isinstance(interpolator, LinearPre)


def test_import_linear_top():
    """Verify import from spotforecast2_safe."""
    interpolator = LinearTop()
    assert isinstance(interpolator, LinearTop)
