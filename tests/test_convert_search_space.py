import pytest
import numpy as np
from spotoptim.hyperparameters import ParameterSet
from spotforecast2.model_selection.spotoptim_search import convert_search_space


def test_convert_search_space_parameterset():
    """Test conversion of a SpotOptim ParameterSet."""
    ps = ParameterSet()
    ps.add_float("alpha", 0.0, 1.0)
    ps.add_int("lags", 1, 10)
    ps.add_factor("model", ["Ridge", "Lasso"])

    bounds, vt, vn, vtr = convert_search_space(ps)

    assert vn == ["alpha", "lags", "model"]
    assert vt == ["float", "int", "factor"]
    assert bounds == [(0.0, 1.0), (1, 10), ["Ridge", "Lasso"]]
    assert vtr == [None, None, None]


def test_convert_search_space_simple_dict():
    """Test conversion of a simple dictionary with tuples/lists."""
    ss = {
        "alpha": (0.01, 0.99),  # float inferred
        "max_depth": (2, 10),  # int inferred
        "lags": [1, 5, 24],  # factor (list)
        "log_alpha": (0.01, 1.0, np.log10),  # with transform
    }

    bounds, vt, vn, vtr = convert_search_space(ss)

    assert "alpha" in vn
    assert vt[vn.index("alpha")] == "float"

    assert "max_depth" in vn
    assert vt[vn.index("max_depth")] == "int"

    assert "lags" in vn
    assert vt[vn.index("lags")] == "factor"
    assert bounds[vn.index("lags")] == [1, 5, 24]

    assert "log_alpha" in vn
    assert vtr[vn.index("log_alpha")] == np.log10


def test_convert_search_space_raw_spotoptim_dict():
    """Test conversion of a dictionary already in SpotOptim format."""
    ss = {
        "bounds": [(0, 1)],
        "var_type": ["float"],
        "var_name": ["x"],
        "var_trans": [None],
    }
    bounds, vt, vn, vtr = convert_search_space(ss)
    assert bounds == ss["bounds"]
    assert vt == ss["var_type"]
    assert vn == ss["var_name"]
    assert vtr == ss["var_trans"]


def test_convert_search_space_mixed_types():
    """Test that it correctly distinguishes floats and ints in tuples."""
    ss = {"f1": (1.0, 2.0), "f2": (1, 2.0), "i1": (1, 2)}
    bounds, vt, vn, _ = convert_search_space(ss)

    assert vt[vn.index("f1")] == "float"
    assert vt[vn.index("f2")] == "float"
    assert vt[vn.index("i1")] == "int"


def test_convert_search_space_errors():
    """Test error handling for invalid search space specifications."""
    # Invalid overall type
    with pytest.raises(TypeError, match="search_space must be ParameterSet or dict"):
        convert_search_space([1, 2, 3])

    # Invalid dict value
    with pytest.raises(ValueError, match="Invalid search space for 'x'"):
        convert_search_space({"x": 10})

    # Invalid tuple length
    with pytest.raises(ValueError, match="Invalid search space for 'x'"):
        convert_search_space({"x": (1,)})

    with pytest.raises(ValueError, match="Invalid search space for 'x'"):
        convert_search_space({"x": (1, 2, 3, 4)})
