from spotforecast2_safe.data.data import Period as PeriodFull
from spotforecast2_safe.data import Period as PeriodData
from spotforecast2_safe import Period as PeriodTop


def test_import_period_full():
    """Verify import from spotforecast2_safe.data.data."""
    period = PeriodFull(name="test", n_periods=10, column="test", input_range=(0, 10))
    assert period.name == "test"


def test_import_period_data():
    """Verify import from spotforecast2_safe.data."""
    period = PeriodData(name="test", n_periods=10, column="test", input_range=(0, 10))
    assert period.name == "test"


def test_import_period_top():
    """Verify import from spotforecast2_safe."""
    period = PeriodTop(name="test", n_periods=10, column="test", input_range=(0, 10))
    assert period.name == "test"
