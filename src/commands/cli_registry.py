"""Central command registry for the Paper 2 CLI.

Each entry maps a CLI command name to a module that exposes a ``main()``
callable. ``run_module_command(name)`` imports the module and calls
``main()``. The ``pipeline`` command is special-cased in :mod:`run` because
it parses additional CLI flags (config path, --download, --validate, etc.).

Commands are organised by functional tier rather than by reviewer round, so
referee-driven sub-experiments live next to peers that share the same role
(e.g. exp_05 calm_subsample and exp_06 var_uplift were Round 1 referee
responses but are baseline-vs-data post-processing steps, same tier as
exp_07 stress_vs_calm).

Naming convention:
- ``pipeline``         -- main multi-frequency pipeline (Tier 1)
- ``extended``         -- orchestrator that runs every Tier 2/3 sub-experiment
- ``extended_<name>``  -- one sub-experiment of the extended sweep
- everything else      -- post-processing paper artefacts and dev helpers
"""

from __future__ import annotations

import importlib

COMMANDS: dict[str, str] = {
    # Tier 1 -- Main pipeline (Tables 1-2, Fig 1, A.1-A.5/A.8-A.10/A.12-A.13,
    # A.14 OOS). CLI args parsed in run.py.
    "pipeline":                     "src.commands.cli_main",

    # Tier 2/3 -- Extended-analyses orchestrator: runs every sub-experiment
    # below in sequence. See src/workflows/pipeline_ext.py for the order.
    "extended":                     "src.workflows.pipeline_ext",

    # Tier 2 -- DGP baselines and parameter-sensitivity sweeps that supply
    # reference points / robustness ranges for the main pipeline headlines.
    "extended_simulation":          "src.experiments.exp_04_simulation_calibration",
    "extended_garch_ms":            "src.experiments.exp_11_garch_ms_calibration",
    "extended_k3_baseline":         "src.experiments.exp_13_k3_baseline",
    "extended_asym_baseline":       "src.experiments.exp_16_asym_persistence_baseline",  # Suppl Table tab:asym_baseline (R2 M2)
    "extended_per_asset_baseline":  "src.experiments.exp_18_per_asset_baseline",         # Suppl Table tab:per_asset_baseline (R3 M1)
    "extended_kxw_sweep":           "src.experiments.exp_12_kxw_sweep",          # Tables A.6/A.7/A.11
    "extended_block_sweep_gld":     "src.experiments.exp_14_block_sweep_gld",                # Suppl Table tab:block_sweep (GLD row)
    "extended_block_sweep_assets":  "src.experiments.exp_14_block_sweep_gld:main_assets",     # Suppl Table tab:block_sweep (SPY/QQQ/CL/USDJPY rows)

    # Tier 3 -- Post-processing paper artefacts (consume main-pipeline /
    # Tier-2 outputs). Round 1 referee responses (exp_05 Q7, exp_06 Q9) live
    # here next to their peers; grouping by function avoids splitting peers
    # across "round" buckets.
    "extended_majority_vote":       "src.experiments.exp_01_majority_vote",      # Table A.15
    "extended_bootstrap":           "src.experiments.exp_02_bootstrap",          # Table A.16
    "extended_hypothesis_tests":    "src.experiments.exp_03_hypothesis_tests",
    "extended_calm_subsample":      "src.experiments.exp_05_calm_subsample",     # Table A.17 (R1 Q7)
    "extended_var_uplift":          "src.experiments.exp_06_var_uplift",         # Table A.18 (R1 Q9)
    "extended_disagree_config":     "src.experiments.exp_15_disagree_config",
    "extended_em_restart_placebo":  "src.experiments.exp_17_em_restart_placebo", # R2 placebo for VaR uplift
    "stress_vs_calm":               "src.experiments.exp_07_stress_vs_calm",     # Table A.19
    "cross_asset":                  "src.visualization.cross_asset_resonance",   # Suppl. cross-asset resonance figure

    # Tier 4 -- Dev helpers (no paper artefact; not in the run.py all chain).
    "summarize":                    "src.experiments.exp_08_summarize_windows",
}


def _split_target(target: str) -> tuple[str, str]:
    if ":" in target:
        module_path, func_name = target.split(":", 1)
        return module_path, func_name
    return target, "main"


def run_module_command(name: str) -> None:
    module_path, func_name = _split_target(COMMANDS[name])
    mod = importlib.import_module(module_path)
    getattr(mod, func_name)()


def assert_commands_importable() -> None:
    """Smoke-check that every entry in ``COMMANDS`` resolves to a real module.

    Catches typos in the registry early (e.g. a renamed experiment file
    leaving a dangling import path). Called from the test suite; not
    invoked at module-import time so a single broken entry does not block
    every CLI command from running. For ``module:function`` targets the
    function callability is also asserted; bare module targets are not
    callability-checked because some entries (notably ``pipeline``) are
    special-cased in :mod:`run` and intentionally have no module-level
    ``main`` callable.
    """
    for name, target in COMMANDS.items():
        module_path, func_name = _split_target(target)
        try:
            mod = importlib.import_module(module_path)
        except Exception as e:  # pragma: no cover - exercised by tests
            raise ImportError(
                f"COMMANDS[{name!r}] -> {target!r}: module not importable: {e}"
            ) from e
        if ":" in target and not callable(getattr(mod, func_name, None)):
            raise ImportError(
                f"COMMANDS[{name!r}] -> {target!r}: "
                f"{module_path}.{func_name} is missing or not callable"
            )
