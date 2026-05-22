# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the ``multitask.strategies`` module (Step 4 of ADR-001).

The strategies are introduced as scaffolding — they are not yet wired into
``BaseTask``.  These tests only confirm that the classes instantiate, that
``DefaultsStrategy`` raises ``NotImplementedError`` (so the seam is visible
to anyone exploring the code), and that each strategy carries the expected
``name`` attribute used by the future dispatcher in Step 5.
"""

from spotforecast2.multitask.strategies import (
    DefaultsStrategy,
    LazyStrategy,
    OptunaStrategy,
    SpotOptimStrategy,
)


def test_lazy_strategy_instantiates():
    strategy = LazyStrategy()
    assert strategy.name == "lazy"
    assert strategy.use_tuned_params is True


def test_defaults_strategy_instantiates():
    strategy = DefaultsStrategy()
    assert strategy.name == "defaults"


def test_optuna_strategy_instantiates():
    strategy = OptunaStrategy()
    assert strategy.name == "optuna"
    assert strategy.search_space is None


def test_spotoptim_strategy_instantiates():
    strategy = SpotOptimStrategy()
    assert strategy.name == "spotoptim"
    assert strategy.search_space is None


def test_defaults_strategy_returns_forecaster_unchanged():
    """``DefaultsStrategy.prepare_forecaster`` is a no-op pre-fit step that
    returns the forecaster unchanged (ADR-002 Step 2 implemented the stub)."""
    sentinel = object()
    strategy = DefaultsStrategy()
    result = strategy.prepare_forecaster(
        task=None,
        target="any",
        forecaster=sentinel,
        y_train=None,
        exog_train=None,
    )
    assert result is sentinel
