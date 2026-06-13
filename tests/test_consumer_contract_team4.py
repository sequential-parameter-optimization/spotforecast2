# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Consumer-contract test protecting ``bart26l-vorlesung/14_team_4_submission.qmd``.

Decider requirement (ADR ``adr-multitask-configmulti-merge``, 2026-06-06):
every refactoring or code change must keep the lecture document valid and
tested — always against its *most recent* version. The source of truth is the
German lecture book ``bart26l-vorlesung``; ``bart26k-lecture`` is its generated
English mirror with byte-identical python cells, so the contract is checked
against the German source. This test therefore parses the qmd at test time
(no frozen copy) and statically verifies its API surface against the installed
packages:

1. every ``spotforecast2`` / ``spotforecast2_safe`` import in its python cells
   resolves,
2. the ``ConfigEntsoe(...)`` constructor keywords bind against the current
   signature,
3. every attribute the document reads or writes on its config object exists,
4. the ``MultiTask(...)`` call and every pipeline method invoked on the
   resulting instance exist and accept the keywords used.

The document is located via the ``TEAM4_QMD`` environment variable (falling
back to the canonical workspace path) and the whole module is skipped when it
is absent, so CI machines without the lecture repo stay green. The test never
*renders* the document — rendering trains models and needs ENTSO-E
credentials; a manual render remains the final pre-release check.
"""

import ast
import inspect
import os
import re
from pathlib import Path

import pytest
from spotforecast2_safe.configurator.config_entsoe import ConfigEntsoe

from spotforecast2.multitask import MultiTask

DEFAULT_QMD = Path.home() / "workspace" / "bart26l-vorlesung" / "14_team_4_submission.qmd"
QMD_PATH = Path(os.environ.get("TEAM4_QMD", DEFAULT_QMD))

pytestmark = pytest.mark.skipif(
    not QMD_PATH.is_file(),
    reason=f"protected consumer qmd not available: {QMD_PATH} (set TEAM4_QMD)",
)

_CELL_RE = re.compile(r"^```\{python[^}]*\}\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _python_cells(text: str) -> list[str]:
    """Extract the source of all ``{python}`` cells from a qmd document."""
    return [m.group(1) for m in _CELL_RE.finditer(text)]


def _parsed_cells() -> list[ast.Module]:
    """Parse all python cells; cells with qmd-only syntax are skipped."""
    text = QMD_PATH.read_text(encoding="utf-8")
    cells = _python_cells(text)
    assert cells, f"no python cells found in {QMD_PATH}"
    trees = []
    for cell in cells:
        try:
            trees.append(ast.parse(cell))
        except SyntaxError:
            # e.g. cells containing Quarto shortcodes; not part of the contract
            continue
    assert trees, "no python cell parsed — qmd structure changed?"
    return trees


@pytest.fixture(scope="module")
def trees() -> list[ast.Module]:
    return _parsed_cells()


def _walk(trees: list[ast.Module]):
    for tree in trees:
        yield from ast.walk(tree)


def _config_var_names(trees: list[ast.Module]) -> set[str]:
    """Names of variables assigned from ``ConfigEntsoe(...)``."""
    names = set()
    for node in _walk(trees):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "ConfigEntsoe"
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _multitask_var_names(trees: list[ast.Module]) -> set[str]:
    """Names of variables assigned from ``MultiTask(...)``."""
    names = set()
    for node in _walk(trees):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "MultiTask"
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _calls_of(trees: list[ast.Module], func_name: str):
    for node in _walk(trees):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
        ):
            yield node


def _bind(callable_obj, call: ast.Call) -> None:
    """Bind a call's argument shape against a signature; fail on mismatch."""
    sig = inspect.signature(callable_obj)
    args = [None] * len(call.args)
    kwargs = {kw.arg: None for kw in call.keywords if kw.arg is not None}
    sig.bind_partial(*args, **kwargs)  # raises TypeError on renamed/removed params


def test_spotforecast_imports_resolve(trees):
    """Every spotforecast2/spotforecast2_safe import in the qmd resolves."""
    import importlib

    checked = 0
    for node in _walk(trees):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.startswith("spotforecast2"):
                continue
            mod = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(
                    mod, alias.name
                ), f"{node.module} no longer exports {alias.name!r}"
                checked += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("spotforecast2"):
                    importlib.import_module(alias.name)
                    checked += 1
    assert checked > 0, "qmd no longer imports from spotforecast2 packages?"


def test_config_entsoe_constructor_kwargs_bind(trees):
    """The ConfigEntsoe(...) keywords used in the qmd bind to the signature."""
    calls = list(_calls_of(trees, "ConfigEntsoe"))
    assert calls, "qmd no longer constructs ConfigEntsoe directly"
    for call in calls:
        _bind(ConfigEntsoe, call)


def test_config_attribute_surface(trees):
    """Every attribute the qmd reads/writes on its config object exists.

    This is the core protection for the RunState migration: if the document
    ever reads a field that a refactor removed from the config (e.g. one of
    the derived window fields), this test fails before a release ships.
    """
    config_vars = _config_var_names(trees)
    assert config_vars, "qmd no longer assigns a ConfigEntsoe instance"
    cfg = ConfigEntsoe()
    missing = set()
    for node in _walk(trees):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in config_vars
        ):
            if not hasattr(cfg, node.attr):
                missing.add(node.attr)
    assert (
        not missing
    ), f"qmd uses config attributes missing on ConfigEntsoe: {sorted(missing)}"


def test_multitask_call_and_pipeline_methods(trees):
    """MultiTask(...) binds, and every method called on the instance exists."""
    mt_calls = list(_calls_of(trees, "MultiTask"))
    assert mt_calls, "qmd no longer constructs MultiTask"
    for call in mt_calls:
        _bind(MultiTask, call)
        # keywords that are not MultiTask params travel via **overrides into
        # config.set_params(); they must be valid config fields.
        mt_params = set(inspect.signature(MultiTask).parameters)
        for kw in call.keywords:
            if kw.arg is not None and kw.arg not in mt_params:
                assert (
                    kw.arg in ConfigEntsoe._PARAM_NAMES
                ), f"MultiTask override {kw.arg!r} is not a ConfigEntsoe field"

    mt_vars = _multitask_var_names(trees)
    assert mt_vars, "qmd no longer assigns a MultiTask instance"
    invoked = []
    for node in _walk(trees):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in mt_vars
        ):
            invoked.append(node)
    assert invoked, "qmd no longer calls any method on its MultiTask instance"
    for call in invoked:
        name = call.func.attr
        method = getattr(MultiTask, name, None)
        assert method is not None, f"MultiTask lost method {name!r} used by the qmd"
        _bind(
            method,
            ast.Call(  # account for the bound `self` slot
                func=call.func,
                args=[ast.Constant(value=None)] + list(call.args),
                keywords=call.keywords,
            ),
        )
