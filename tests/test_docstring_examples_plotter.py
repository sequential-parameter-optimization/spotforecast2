# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test suite for plotter module docstring examples.

This module validates all examples in the plotter module documentation
to ensure they execute correctly and produce expected results.
"""

import doctest

from spotforecast2.manager import plotter


def test_docstring_examples_plotter():
    """Test all docstring examples in plotter module."""
    results = doctest.testmod(plotter, verbose=False)
    assert results.failed == 0, f"Doctest failed: {results.failed} failures"
