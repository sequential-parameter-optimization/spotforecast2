import pytest
from spotforecast2.model_selection.spotoptim_search import parse_lags_from_strings


def test_parse_lags_from_strings_types():
    """Validate that parse_lags_from_strings handles different input types correctly."""

    # 1. Passthrough cases
    assert parse_lags_from_strings(24) == 24
    assert parse_lags_from_strings([1, 10, 24]) == [1, 10, 24]

    # 2. String integer cases
    assert parse_lags_from_strings("24") == 24
    assert parse_lags_from_strings(" 48 ") == 48

    # 3. String list cases
    assert parse_lags_from_strings("[1, 2, 3]") == [1, 2, 3]
    assert parse_lags_from_strings(" [1, 2, 3] ") == [1, 2, 3]
    assert parse_lags_from_strings("[1,2,3]") == [1, 2, 3]

    # 4. Fallback cases (unrecognized strings returned as-is)
    assert parse_lags_from_strings("lag_name") == "lag_name"
    assert (
        parse_lags_from_strings("24.5") == "24.5"
    )  # Float string not handled by int()
    assert parse_lags_from_strings("") == ""


def test_parse_lags_from_strings_complex_lists():
    """Verify that complex list strings are handled correctly via literal_eval."""
    assert parse_lags_from_strings("[24, 48, 168]") == [24, 48, 168]
    # Nested lists or other literals (though not typical for lags)
    assert parse_lags_from_strings("[[1, 2], [3, 4]]") == [[1, 2], [3, 4]]


def test_parse_lags_from_strings_error_cases():
    """Verify behavior on malformed list strings."""
    # Malformed list strings will raise ValueError or SyntaxError from ast.literal_eval
    with pytest.raises(SyntaxError):
        parse_lags_from_strings("[1, 2,")

    with pytest.raises(ValueError):
        # literal_eval only handles literals; complex expressions like [1+2] fail
        parse_lags_from_strings("[1+2]")


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (1, 1),
        ("1", 1),
        ([1], [1]),
        ("[1]", [1]),
        (" [ 1 ] ", [1]),
    ],
)
def test_parse_lags_from_strings_parametrized(input_val, expected):
    """Parametrized testing for consistency across common scenarios."""
    assert parse_lags_from_strings(input_val) == expected
