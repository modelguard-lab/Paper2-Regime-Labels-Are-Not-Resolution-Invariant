"""Import-only smoke tests for every experiment module.

Catches signature drift, broken imports, or renamed dependencies in the
experiments that do not have a dedicated unit test (exp_03/04/05/07/08/11-16).
A full functional run for these orchestrators requires real outputs/ CSVs
from a prior pipeline run, so these tests intentionally do not invoke
``main()``; they only verify the module loads and exposes a callable ``main``.

The CLI registry's ``assert_commands_importable`` (covered in
``test_cli_registry.py``) checks the entries that are wired through the CLI;
this file mirrors that check at the experiment-module level so an experiment
removed from the registry would still be caught.
"""

from __future__ import annotations

import importlib

import pytest

EXPERIMENT_MODULES = [
    "src.experiments.exp_01_majority_vote",
    "src.experiments.exp_02_bootstrap",
    "src.experiments.exp_03_hypothesis_tests",
    "src.experiments.exp_04_simulation_calibration",
    "src.experiments.exp_05_calm_subsample",
    "src.experiments.exp_06_var_uplift",
    "src.experiments.exp_07_stress_vs_calm",
    "src.experiments.exp_08_summarize_windows",
    "src.experiments.exp_11_garch_ms_calibration",
    "src.experiments.exp_12_kxw_sweep",
    "src.experiments.exp_13_k3_baseline",
    "src.experiments.exp_14_block_sweep_gld",
    "src.experiments.exp_15_disagree_config",
    "src.experiments.exp_16_asym_persistence_baseline",
    "src.experiments.exp_17_em_restart_placebo",
    "src.experiments.exp_18_per_asset_baseline",
]


@pytest.mark.parametrize("dotted", EXPERIMENT_MODULES)
def test_experiment_module_imports_and_exposes_main(dotted: str) -> None:
    mod = importlib.import_module(dotted)
    assert hasattr(mod, "main"), f"{dotted}: missing main()"
    assert callable(mod.main), f"{dotted}: main is not callable"
