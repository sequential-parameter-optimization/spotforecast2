import pytest
from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder as ExogFull
from spotforecast2_safe.preprocessing import ExogBuilder as ExogPre
from spotforecast2_safe import ExogBuilder as ExogTop

def test_import_exog_full():
    """Verify import from spotforecast2_safe.preprocessing.exog_builder."""
    builder = ExogFull(periods=[], country_code=None)
    assert builder.periods == []

def test_import_exog_pre():
    """Verify import from spotforecast2_safe.preprocessing."""
    builder = ExogPre(periods=[], country_code=None)
    assert builder.periods == []

def test_import_exog_top():
    """Verify import from spotforecast2_safe."""
    builder = ExogTop(periods=[], country_code=None)
    assert builder.periods == []


