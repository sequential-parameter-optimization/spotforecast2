# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Documentation consistency tests for the four pipeline tasks.

Ensures that all task classes (lazy, optuna, spotoptim, predict) are
consistently referenced across docstrings, __init__.py exports,
quartodoc configuration, and multi.py dispatcher.
"""

import importlib
import inspect
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_CLASSES = ["LazyTask", "OptunaTask", "SpotOptimTask", "PredictTask"]
TASK_KEYS = ["lazy", "optuna", "spotoptim", "predict"]
ALL_EXPORTED_CLASSES = TASK_CLASSES + ["BaseTask", "MultiTask", "agg_predictor"]

MULTITASK_PKG = "spotforecast2.manager.multitask"
MANAGER_PKG = "spotforecast2.manager"

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "spotforecast2"
MULTITASK_DIR = SRC_DIR / "manager" / "multitask"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUARTO_YML = PROJECT_ROOT / "_quarto.yml"


# ===========================================================================
# 1. Import / export consistency
# ===========================================================================


class TestImportExportConsistency:
    """All task classes must be importable from the expected packages."""

    @pytest.mark.parametrize("cls_name", TASK_CLASSES)
    def test_task_importable_from_multitask(self, cls_name):
        mod = importlib.import_module(MULTITASK_PKG)
        assert hasattr(mod, cls_name), f"{cls_name} missing from {MULTITASK_PKG}"

    @pytest.mark.parametrize("cls_name", TASK_CLASSES)
    def test_task_importable_from_manager(self, cls_name):
        mod = importlib.import_module(MANAGER_PKG)
        assert hasattr(mod, cls_name), f"{cls_name} missing from {MANAGER_PKG}"

    @pytest.mark.parametrize("cls_name", ALL_EXPORTED_CLASSES)
    def test_all_exported_in_multitask(self, cls_name):
        mod = importlib.import_module(MULTITASK_PKG)
        assert (
            cls_name in mod.__all__
        ), f"{cls_name} missing from {MULTITASK_PKG}.__all__"

    @pytest.mark.parametrize("cls_name", ALL_EXPORTED_CLASSES)
    def test_all_exported_in_manager(self, cls_name):
        mod = importlib.import_module(MANAGER_PKG)
        assert cls_name in mod.__all__, f"{cls_name} missing from {MANAGER_PKG}.__all__"


# ===========================================================================
# 2. BaseTask class docstring consistency
# ===========================================================================


class TestBaseTaskDocstring:
    """BaseTask docstring must reference all four task keys."""

    def _get_basetask_docstring(self):
        from spotforecast2.manager.multitask.base import BaseTask

        return inspect.getdoc(BaseTask)

    @pytest.mark.parametrize("task_key", TASK_KEYS)
    def test_task_key_in_basetask_docstring(self, task_key):
        doc = self._get_basetask_docstring()
        assert task_key in doc, f"Task key '{task_key}' not found in BaseTask docstring"

    def test_no_stale_training_reference(self):
        doc = self._get_basetask_docstring()
        assert (
            "training, optuna" not in doc.lower()
        ), "Stale '(lazy, training, optuna, spotoptim)' reference in BaseTask docstring"

    def test_no_colon_class_formatting(self):
        doc = self._get_basetask_docstring()
        assert ":class:" not in doc, ":class: formatting found in BaseTask docstring"
        assert ":meth:" not in doc, ":meth: formatting found in BaseTask docstring"


# ===========================================================================
# 3. BaseTask.run() NotImplementedError message
# ===========================================================================


class TestBaseTaskRunMessage:
    """BaseTask.run() error message must list all concrete task classes."""

    def test_run_error_lists_all_tasks(self):
        from spotforecast2.manager.multitask.base import BaseTask

        task = BaseTask.__new__(BaseTask)
        with pytest.raises(NotImplementedError) as exc_info:
            task.run()
        msg = str(exc_info.value)
        for cls_name in TASK_CLASSES:
            assert (
                cls_name in msg
            ), f"'{cls_name}' missing from BaseTask.run() error message"


# ===========================================================================
# 4. MultiTask dispatcher consistency
# ===========================================================================


class TestMultiTaskDispatcher:
    """MultiTask.run() must support all four task keys."""

    @pytest.mark.parametrize("task_key", TASK_KEYS)
    def test_dispatcher_accepts_task_key(self, task_key):
        from spotforecast2.manager.multitask.multi import MultiTask

        mt = MultiTask.__new__(MultiTask)
        mt.TASK = task_key
        dispatch = {
            "lazy": "run_task_lazy",
            "optuna": "run_task_optuna",
            "spotoptim": "run_task_spotoptim",
            "predict": "run_task_predict",
        }
        assert hasattr(
            mt, dispatch[task_key]
        ), f"MultiTask missing method '{dispatch[task_key]}'"

    def test_dispatcher_rejects_unknown_task(self):
        from spotforecast2.manager.multitask.multi import MultiTask

        mt = MultiTask(task="lazy", predict_size=24)
        with pytest.raises(ValueError, match="Unknown task"):
            mt.run(task="nonexistent", show=False)

    def test_multitask_docstring_lists_predict(self):
        from spotforecast2.manager.multitask.multi import MultiTask

        doc = inspect.getdoc(MultiTask)
        assert (
            "predict" in doc.lower()
        ), "'predict' not mentioned in MultiTask class docstring"


# ===========================================================================
# 5. Module docstring consistency
# ===========================================================================


class TestModuleDocstrings:
    """Module docstrings must be consistent with actual exports."""

    def test_init_docstring_lists_all_classes(self):
        mod = importlib.import_module(MULTITASK_PKG)
        doc = mod.__doc__
        for cls_name in TASK_CLASSES + ["BaseTask", "MultiTask"]:
            assert (
                cls_name in doc
            ), f"{cls_name} not found in {MULTITASK_PKG} module docstring"

    def test_init_docstring_no_colon_class(self):
        mod = importlib.import_module(MULTITASK_PKG)
        doc = mod.__doc__
        assert ":class:" not in doc, f":class: found in {MULTITASK_PKG} docstring"

    def test_base_module_docstring_mentions_predict(self):
        from spotforecast2.manager.multitask import base

        doc = base.__doc__
        assert "PredictTask" in doc, "PredictTask not in base module docstring"

    def test_base_module_docstring_no_colon_class(self):
        from spotforecast2.manager.multitask import base

        doc = base.__doc__
        assert ":class:" not in doc, ":class: found in base module docstring"
        assert ":meth:" not in doc, ":meth: found in base module docstring"
        assert ":func:" not in doc, ":func: found in base module docstring"


# ===========================================================================
# 6. No forbidden formatting in any multitask Python file
# ===========================================================================


class TestNoForbiddenFormatting:
    """No multitask .py file may contain :class:, :meth:, or :func:."""

    @pytest.mark.parametrize(
        "filename",
        [
            "__init__.py",
            "base.py",
            "lazy.py",
            "optuna.py",
            "spotoptim.py",
            "predict.py",
            "multi.py",
        ],
    )
    def test_no_sphinx_crossrefs(self, filename):
        filepath = MULTITASK_DIR / filename
        if not filepath.exists():
            pytest.skip(f"{filename} does not exist")
        content = filepath.read_text()
        for pattern in [":class:", ":meth:", ":func:"]:
            assert pattern not in content, f"Forbidden '{pattern}' found in {filename}"


# ===========================================================================
# 7. Quartodoc configuration consistency
# ===========================================================================


class TestQuartodocConfig:
    """_quarto.yml must include entries for all task classes."""

    @pytest.fixture()
    def quarto_content(self):
        if not QUARTO_YML.exists():
            pytest.skip("_quarto.yml not found")
        return QUARTO_YML.read_text()

    @pytest.mark.parametrize("cls_name", TASK_CLASSES + ["BaseTask", "MultiTask"])
    def test_sidebar_entry(self, quarto_content, cls_name):
        assert cls_name in quarto_content, f"'{cls_name}' not found in _quarto.yml"

    @pytest.mark.parametrize("cls_name", TASK_CLASSES + ["BaseTask", "MultiTask"])
    def test_quartodoc_section_entry(self, quarto_content, cls_name):
        expected = f"manager.multitask.{cls_name}"
        assert (
            expected in quarto_content
        ), f"'{expected}' not found in _quarto.yml quartodoc section"


# ===========================================================================
# 8. Task subclass hierarchy consistency
# ===========================================================================


class TestTaskHierarchy:
    """All task classes must inherit from BaseTask and have a run method."""

    @pytest.mark.parametrize("cls_name", TASK_CLASSES + ["MultiTask"])
    def test_inherits_from_base(self, cls_name):
        from spotforecast2.manager.multitask.base import BaseTask

        mod = importlib.import_module(MULTITASK_PKG)
        cls = getattr(mod, cls_name)
        assert issubclass(cls, BaseTask), f"{cls_name} does not inherit BaseTask"

    @pytest.mark.parametrize("cls_name", TASK_CLASSES + ["MultiTask"])
    def test_has_run_method(self, cls_name):
        mod = importlib.import_module(MULTITASK_PKG)
        cls = getattr(mod, cls_name)
        assert hasattr(cls, "run"), f"{cls_name} missing run() method"

    @pytest.mark.parametrize(
        "cls_name,expected_task_name",
        [
            ("LazyTask", "lazy"),
            ("OptunaTask", "optuna"),
            ("SpotOptimTask", "spotoptim"),
            ("PredictTask", "predict"),
        ],
    )
    def test_task_name_attribute(self, cls_name, expected_task_name):
        mod = importlib.import_module(MULTITASK_PKG)
        cls = getattr(mod, cls_name)
        assert cls._task_name == expected_task_name, (
            f"{cls_name}._task_name is '{cls._task_name}', "
            f"expected '{expected_task_name}'"
        )


# ===========================================================================
# 9. auto_save_models default behavior
# ===========================================================================


class TestAutoSaveModels:
    """Training tasks must auto-save models by default."""

    def test_base_task_auto_save_default_true(self):
        from spotforecast2.manager.multitask.base import BaseTask

        task = BaseTask(predict_size=24)
        assert (
            task.auto_save_models is True
        ), "BaseTask.auto_save_models should default to True"

    def test_base_task_auto_save_can_be_disabled(self):
        from spotforecast2.manager.multitask.base import BaseTask

        task = BaseTask(predict_size=24, auto_save_models=False)
        assert task.auto_save_models is False

    @pytest.mark.parametrize("cls_name", ["LazyTask", "OptunaTask", "SpotOptimTask"])
    def test_training_task_inherits_auto_save_true(self, cls_name):
        mod = importlib.import_module(MULTITASK_PKG)
        cls = getattr(mod, cls_name)
        task = cls(predict_size=24)
        assert (
            task.auto_save_models is True
        ), f"{cls_name}.auto_save_models should default to True"

    @pytest.mark.parametrize("cls_name", ["LazyTask", "OptunaTask", "SpotOptimTask"])
    def test_training_task_auto_save_can_be_disabled(self, cls_name):
        mod = importlib.import_module(MULTITASK_PKG)
        cls = getattr(mod, cls_name)
        task = cls(predict_size=24, auto_save_models=False)
        assert task.auto_save_models is False

    def test_predict_task_auto_save_attribute(self):
        from spotforecast2.manager.multitask import PredictTask

        task = PredictTask(predict_size=24)
        assert hasattr(
            task, "auto_save_models"
        ), "PredictTask should have auto_save_models attribute via BaseTask"

    def test_execute_lazy_calls_save_models_when_enabled(self):
        from unittest.mock import MagicMock

        from spotforecast2.manager.multitask.lazy import execute_lazy

        task = MagicMock()
        task.auto_save_models = True
        task.config.targets = ["t1"]
        task._get_target_data.return_value = (MagicMock(), MagicMock(), MagicMock())
        task.create_forecaster.return_value = MagicMock()
        task.load_tuning_results.return_value = None
        task._train_and_predict_target.return_value = {"future_pred": MagicMock()}
        task._aggregate_and_show.return_value = {}

        execute_lazy(task, show=False)

        task.save_models.assert_called_once_with(task_name="lazy")

    def test_execute_lazy_skips_save_models_when_disabled(self):
        from unittest.mock import MagicMock

        from spotforecast2.manager.multitask.lazy import execute_lazy

        task = MagicMock()
        task.auto_save_models = False
        task.config.targets = ["t1"]
        task._get_target_data.return_value = (MagicMock(), MagicMock(), MagicMock())
        task.create_forecaster.return_value = MagicMock()
        task.load_tuning_results.return_value = None
        task._train_and_predict_target.return_value = {"future_pred": MagicMock()}
        task._aggregate_and_show.return_value = {}

        execute_lazy(task, show=False)

        task.save_models.assert_not_called()

    def test_execute_optuna_calls_save_models_when_enabled(self):
        from unittest.mock import MagicMock, patch

        from spotforecast2.manager.multitask.optuna import execute_optuna

        task = MagicMock()
        task.auto_save_models = True
        task.config.targets = ["t1"]
        task.config.n_trials_optuna = 1
        task.config.random_state = 42
        task._get_target_data.return_value = (MagicMock(), MagicMock(), MagicMock())
        task.create_forecaster.return_value = MagicMock()
        task.cv_ts.return_value = MagicMock()
        task._train_and_predict_target.return_value = {}
        task._aggregate_and_show.return_value = {}

        mock_results = MagicMock()
        mock_row = MagicMock()
        mock_row.params = {}
        mock_row.lags = 24
        mock_results.iloc.__getitem__.return_value = mock_row

        with patch(
            "spotforecast2.manager.multitask.optuna.bayesian_search_forecaster",
            return_value=(mock_results, None),
        ):
            execute_optuna(task, show=False)

        task.save_models.assert_called_once_with(task_name="optuna")

    def test_execute_spotoptim_calls_save_models_when_enabled(self):
        from unittest.mock import MagicMock, patch

        from spotforecast2.manager.multitask.spotoptim import execute_spotoptim

        task = MagicMock()
        task.auto_save_models = True
        task.config.targets = ["t1"]
        task.config.n_trials_spotoptim = 1
        task.config.n_initial_spotoptim = 1
        task.config.random_state = 42
        task._get_target_data.return_value = (MagicMock(), MagicMock(), MagicMock())
        task.create_forecaster.return_value = MagicMock()
        task.cv_ts.return_value = MagicMock()
        task._train_and_predict_target.return_value = {}
        task._aggregate_and_show.return_value = {}

        mock_results = MagicMock()
        mock_row = MagicMock()
        mock_row.params = {}
        mock_row.lags = 24
        mock_results.iloc.__getitem__.return_value = mock_row

        with patch(
            "spotforecast2.manager.multitask.spotoptim.spotoptim_search_forecaster",
            return_value=(mock_results, MagicMock()),
        ):
            execute_spotoptim(task, show=False)

        task.save_models.assert_called_once_with(task_name="spotoptim")
