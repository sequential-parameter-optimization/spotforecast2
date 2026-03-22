# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests verifying that multitask submodules are public (not private).

Coverage:
- All six public module paths are importable:
    spotforecast2.manager.multitask.base
    spotforecast2.manager.multitask.lazy
    spotforecast2.manager.multitask.train
    spotforecast2.manager.multitask.optuna
    spotforecast2.manager.multitask.spotoptim
    spotforecast2.manager.multitask.multi
- Classes imported from public submodules are the same objects as those
  exported from spotforecast2.manager.multitask (the package __init__)
- Private module names (_base, _lazy, etc.) no longer exist
- Source files for each class are in the public module, not the private one
"""

import importlib
import inspect
import sys

import pytest

# ---------------------------------------------------------------------------
# Classes from the package __init__ (canonical public API)
# ---------------------------------------------------------------------------

from spotforecast2.manager.multitask import (
    BaseTask,
    LazyTask,
    MultiTask,
    OptunaTask,
    SpotOptimTask,
    TrainTask,
)

# ===========================================================================
# Public module importability
# ===========================================================================


class TestPublicModulesImportable:
    """Each submodule must be importable under its public name."""

    def test_base_module_importable(self):
        mod = importlib.import_module("spotforecast2.manager.multitask.base")

        assert mod is not None

    def test_lazy_module_importable(self):
        mod = importlib.import_module("spotforecast2.manager.multitask.lazy")

        assert mod is not None

    def test_train_module_importable(self):
        mod = importlib.import_module("spotforecast2.manager.multitask.train")

        assert mod is not None

    def test_optuna_module_importable(self):
        mod = importlib.import_module("spotforecast2.manager.multitask.optuna")

        assert mod is not None

    def test_spotoptim_module_importable(self):
        mod = importlib.import_module("spotforecast2.manager.multitask.spotoptim")

        assert mod is not None

    def test_multi_module_importable(self):
        mod = importlib.import_module("spotforecast2.manager.multitask.multi")

        assert mod is not None


# ===========================================================================
# Classes from public submodules are the same objects as from __init__
# ===========================================================================


class TestPublicModulesExportSameObjects:
    """Classes re-exported via __init__ must be identical to the source."""

    def test_base_task_is_same_object(self):
        from spotforecast2.manager.multitask.base import BaseTask as _BaseTask

        assert _BaseTask is BaseTask

    def test_lazy_task_is_same_object(self):
        from spotforecast2.manager.multitask.lazy import LazyTask as _LazyTask

        assert _LazyTask is LazyTask

    def test_train_task_is_same_object(self):
        from spotforecast2.manager.multitask.train import TrainTask as _TrainTask

        assert _TrainTask is TrainTask

    def test_optuna_task_is_same_object(self):
        from spotforecast2.manager.multitask.optuna import OptunaTask as _OptunaTask

        assert _OptunaTask is OptunaTask

    def test_spotoptim_task_is_same_object(self):
        from spotforecast2.manager.multitask.spotoptim import (
            SpotOptimTask as _SpotOptimTask,
        )

        assert _SpotOptimTask is SpotOptimTask

    def test_multi_task_is_same_object(self):
        from spotforecast2.manager.multitask.multi import MultiTask as _MultiTask

        assert _MultiTask is MultiTask


# ===========================================================================
# Source files use public module names
# ===========================================================================


class TestSourceFilesArePublic:
    """inspect.getfile() must point to the public module, not a private one."""

    def test_base_task_source_is_public(self):
        src = inspect.getfile(BaseTask)
        assert "base.py" in src
        assert "_base.py" not in src

    def test_lazy_task_source_is_public(self):
        src = inspect.getfile(LazyTask)
        assert "lazy.py" in src
        assert "_lazy.py" not in src

    def test_train_task_source_is_public(self):
        src = inspect.getfile(TrainTask)
        assert "train.py" in src
        assert "_train.py" not in src

    def test_optuna_task_source_is_public(self):
        src = inspect.getfile(OptunaTask)
        assert "optuna.py" in src
        assert "_optuna.py" not in src

    def test_spotoptim_task_source_is_public(self):
        src = inspect.getfile(SpotOptimTask)
        assert "spotoptim.py" in src
        assert "_spotoptim.py" not in src

    def test_multi_task_source_is_public(self):
        src = inspect.getfile(MultiTask)
        assert "multi.py" in src
        assert "_multi.py" not in src


# ===========================================================================
# Private module names no longer exist
# ===========================================================================


class TestPrivateModulesGone:
    """The old _base, _lazy, … modules must no longer be importable."""

    @pytest.mark.parametrize(
        "private_module",
        [
            "spotforecast2.manager.multitask._base",
            "spotforecast2.manager.multitask._lazy",
            "spotforecast2.manager.multitask._train",
            "spotforecast2.manager.multitask._optuna",
            "spotforecast2.manager.multitask._spotoptim",
            "spotforecast2.manager.multitask._multi",
        ],
    )
    def test_private_module_not_importable(self, private_module):
        # Remove from sys.modules if cached from old runs
        sys.modules.pop(private_module, None)
        with pytest.raises((ModuleNotFoundError, ImportError)):
            importlib.import_module(private_module)


# ===========================================================================
# Execute functions are exported from the public submodules
# ===========================================================================


class TestExecuteFunctionsPublic:
    """Helper execute_* functions must be accessible from public submodules."""

    def test_execute_lazy_importable(self):
        from spotforecast2.manager.multitask.lazy import execute_lazy

        assert callable(execute_lazy)

    def test_execute_training_importable(self):
        from spotforecast2.manager.multitask.train import execute_training

        assert callable(execute_training)

    def test_execute_optuna_importable(self):
        from spotforecast2.manager.multitask.optuna import execute_optuna

        assert callable(execute_optuna)

    def test_execute_spotoptim_importable(self):
        from spotforecast2.manager.multitask.spotoptim import execute_spotoptim

        assert callable(execute_spotoptim)
