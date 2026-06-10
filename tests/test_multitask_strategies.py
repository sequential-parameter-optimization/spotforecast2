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


class _FakeForecaster:
    """Minimal stand-in accepting set_params/set_lags."""

    def set_params(self, **kwargs):
        self.params = kwargs

    def set_lags(self, lags):
        self.lags = lags


def _make_fake_task(**config_extra):
    """Fake task carrying just what SpotOptimStrategy.prepare_forecaster reads."""
    import logging
    import types

    cfg = types.SimpleNamespace(
        n_trials_spotoptim=2,
        n_initial_spotoptim=1,
        random_state=0,
        warm_start_lags=False,
        **config_extra,
    )
    return types.SimpleNamespace(
        config=cfg,
        logger=logging.getLogger("test-strategies"),
        cv_ts=lambda y: None,
        create_forecaster=lambda target=None: _FakeForecaster(),
        save_tuning_results=lambda **kw: None,
        _show_progress=False,
    )


def test_spotoptim_strategy_forwards_tensorboard_kwargs(monkeypatch):
    """config.tensorboard_* attrs must reach kwargs_spotoptim unchanged.

    ``prepare_forecaster`` imports ``spotoptim_search_forecaster`` from
    ``spotforecast2.model_selection`` at call time, so patching the module
    attribute intercepts the call.
    """
    import pandas as pd

    import spotforecast2.model_selection as ms

    captured = {}

    def fake_search_forecaster(*args, **kwargs):
        captured["kwargs_spotoptim"] = kwargs.get("kwargs_spotoptim")
        results = pd.DataFrame({"params": [{"alpha": 1.0}], "lags": [[1, 2]]})
        return results, object()

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", fake_search_forecaster)

    task = _make_fake_task(
        tensorboard_log=True,
        tensorboard_path="runs/test-tb",
    )
    tuned = SpotOptimStrategy(search_space={"alpha": (0.1, 1.0)}).prepare_forecaster(
        task, "A", _FakeForecaster(), y_train=None
    )

    ks = captured["kwargs_spotoptim"]
    assert ks["tensorboard_log"] is True
    assert ks["tensorboard_path"] == "runs/test-tb"
    # With logging enabled and tensorboard_clean unset, the strategy
    # defaults it to True so every run starts with a fresh dashboard.
    assert ks["tensorboard_clean"] is True
    assert isinstance(tuned, _FakeForecaster)
    assert tuned.params == {"alpha": 1.0}


def test_spotoptim_strategy_respects_explicit_tensorboard_clean_false(monkeypatch):
    """An explicit tensorboard_clean=False must not be overridden."""
    import pandas as pd

    import spotforecast2.model_selection as ms

    captured = {}

    def fake_search_forecaster(*args, **kwargs):
        captured["kwargs_spotoptim"] = kwargs.get("kwargs_spotoptim")
        results = pd.DataFrame({"params": [{"alpha": 1.0}], "lags": [[1, 2]]})
        return results, object()

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", fake_search_forecaster)

    task = _make_fake_task(
        tensorboard_log=True,
        tensorboard_path="runs/test-tb",
        tensorboard_clean=False,
    )
    SpotOptimStrategy(search_space={"alpha": (0.1, 1.0)}).prepare_forecaster(
        task, "A", _FakeForecaster(), y_train=None
    )

    ks = captured["kwargs_spotoptim"]
    assert ks["tensorboard_log"] is True
    assert ks["tensorboard_clean"] is False


def test_spotoptim_strategy_no_clean_default_without_logging(monkeypatch):
    """tensorboard_clean is only defaulted when tensorboard_log is enabled."""
    import pandas as pd

    import spotforecast2.model_selection as ms

    captured = {}

    def fake_search_forecaster(*args, **kwargs):
        captured["kwargs_spotoptim"] = kwargs.get("kwargs_spotoptim")
        results = pd.DataFrame({"params": [{"alpha": 1.0}], "lags": [[1, 2]]})
        return results, object()

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", fake_search_forecaster)

    # tensorboard_log=False is forwarded (it is not None) but must not
    # trigger the clean default.
    task = _make_fake_task(tensorboard_log=False)
    SpotOptimStrategy(search_space={"alpha": (0.1, 1.0)}).prepare_forecaster(
        task, "A", _FakeForecaster(), y_train=None
    )

    ks = captured["kwargs_spotoptim"]
    assert ks["tensorboard_log"] is False
    assert "tensorboard_clean" not in ks


def test_spotoptim_strategy_no_tensorboard_attrs_no_kwargs(monkeypatch):
    """Without tensorboard config attrs, no tensorboard keys are forwarded."""
    import pandas as pd

    import spotforecast2.model_selection as ms

    captured = {}

    def fake_search_forecaster(*args, **kwargs):
        captured["kwargs_spotoptim"] = kwargs.get("kwargs_spotoptim")
        results = pd.DataFrame({"params": [{"alpha": 1.0}], "lags": [[1, 2]]})
        return results, object()

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", fake_search_forecaster)

    task = _make_fake_task()
    SpotOptimStrategy(search_space={"alpha": (0.1, 1.0)}).prepare_forecaster(
        task, "A", _FakeForecaster(), y_train=None
    )

    # No tensorboard attrs -> empty kwargs dict -> None passed through.
    assert captured["kwargs_spotoptim"] is None


def test_spotoptim_strategy_forwards_max_time(monkeypatch):
    """config.max_time_spotoptim (minutes) must reach kwargs_spotoptim as max_time."""
    import pandas as pd

    import spotforecast2.model_selection as ms

    captured = {}

    def fake_search_forecaster(*args, **kwargs):
        captured["kwargs_spotoptim"] = kwargs.get("kwargs_spotoptim")
        results = pd.DataFrame({"params": [{"alpha": 1.0}], "lags": [[1, 2]]})
        return results, object()

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", fake_search_forecaster)

    task = _make_fake_task(max_time_spotoptim=2.5)
    SpotOptimStrategy(search_space={"alpha": (0.1, 1.0)}).prepare_forecaster(
        task, "A", _FakeForecaster(), y_train=None
    )

    assert captured["kwargs_spotoptim"]["max_time"] == 2.5


def test_spotoptim_strategy_none_max_time_not_forwarded(monkeypatch):
    """max_time_spotoptim=None (the config default) must forward nothing."""
    import pandas as pd

    import spotforecast2.model_selection as ms

    captured = {}

    def fake_search_forecaster(*args, **kwargs):
        captured["kwargs_spotoptim"] = kwargs.get("kwargs_spotoptim")
        results = pd.DataFrame({"params": [{"alpha": 1.0}], "lags": [[1, 2]]})
        return results, object()

    monkeypatch.setattr(ms, "spotoptim_search_forecaster", fake_search_forecaster)

    task = _make_fake_task(max_time_spotoptim=None)
    SpotOptimStrategy(search_space={"alpha": (0.1, 1.0)}).prepare_forecaster(
        task, "A", _FakeForecaster(), y_train=None
    )

    assert captured["kwargs_spotoptim"] is None
