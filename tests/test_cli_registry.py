"""Smoke test for the CLI command registry.

The registry maps human-facing CLI command names (e.g. ``pipeline``,
``extended``, ``stress_vs_calm``) to module targets whose entry callable
gets invoked by ``run.py``. A target is either a bare ``module.path`` (in
which case ``module.main`` is invoked) or ``module.path:func_name`` (in
which case ``module.func_name`` is invoked). A typo or rename of an
experiment file or function leaves a dangling entry that would only
surface the next time someone actually typed that command. Running
``assert_commands_importable`` from a test catches the dangling entry at
CI time instead.
"""

from __future__ import annotations

from src.commands.cli_registry import COMMANDS, _split_target, assert_commands_importable


def test_assert_commands_importable_resolves_every_entry():
    assert_commands_importable()


def test_every_command_module_exposes_main():
    """Each registered target must resolve to a callable: bare module
    targets must expose ``main`` (the historical contract); explicit
    ``module:function`` targets must expose ``function``. The only
    exception is ``pipeline``, which is special-cased in ``run.py`` and
    points at ``cli_main`` (which exposes ``run`` rather than ``main``).
    """
    import importlib

    for name, target in COMMANDS.items():
        module_path, func_name = _split_target(target)
        mod = importlib.import_module(module_path)
        if name == "pipeline":
            assert callable(getattr(mod, "run", None)), (
                f"COMMANDS[{name!r}] -> {target!r} must expose a callable run()"
            )
        else:
            assert callable(getattr(mod, func_name, None)), (
                f"COMMANDS[{name!r}] -> {target!r} must expose a callable {func_name}()"
            )
