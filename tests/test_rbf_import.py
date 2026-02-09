import pytest
from spotforecast2_safe.preprocessing.repeating_basis_function import RepeatingBasisFunction as RBFFull
from spotforecast2_safe.preprocessing import RepeatingBasisFunction as RBFPre
from spotforecast2_safe import RepeatingBasisFunction as RBFTop

from spotforecast2_safe.data.data import Period as PeriodFull
from spotforecast2_safe.data import Period as PeriodData
from spotforecast2_safe import Period as PeriodTop

def test_import_rbf_full():
    """Verify import from spotforecast2_safe.preprocessing.repeating_basis_function."""
    rbf = RBFFull(n_periods=10, column="test", input_range=(0, 10))
    assert rbf.n_periods == 10

def test_import_rbf_pre():
    """Verify import from spotforecast2_safe.preprocessing."""
    rbf = RBFPre(n_periods=10, column="test", input_range=(0, 10))
    assert rbf.n_periods == 10

def test_import_rbf_top():
    """Verify import from spotforecast2_safe."""
    rbf = RBFTop(n_periods=10, column="test", input_range=(0, 10))
    assert rbf.n_periods == 10

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


