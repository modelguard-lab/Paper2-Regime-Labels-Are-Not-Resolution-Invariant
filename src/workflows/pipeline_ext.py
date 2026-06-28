"""
Orchestrator for the extended analyses that complement the main pipeline.

This module lives in :mod:`src.workflows` (sibling of :mod:`src.workflows.pipeline`)
to keep the orchestrator out of the same directory as the experiments it
orchestrates. Each step delegates to the ``main()`` of one
:mod:`src.experiments.exp_NN_<name>` module; each sub-experiment is also
independently CLI-registered so reviewers can re-run a single step without
re-running the full sweep.

  exp_01_majority_vote               run_majority_vote
  exp_02_bootstrap                   run_bootstrap
  exp_03_hypothesis_tests            run_hypothesis_tests
  exp_04_simulation_calibration      run_simulation_calibration
  exp_11_garch_ms_calibration        run_garch_ms_calibration
  exp_13_k3_baseline                 run_k3_baseline
  exp_16_asym_persistence_baseline   run_asym_baseline
  exp_05_calm_subsample              run_calm_day_subsample
  exp_06_var_uplift                  run_var_uplift
  exp_12_kxw_sweep                   run_robustness (K x window-scale)
  exp_14_block_sweep_gld             run_block_sweep_gld + run_block_sweep_assets
  exp_15_disagree_config             run_disagree_config
  exp_17_em_restart_placebo          run_em_restart_placebo
  exp_18_per_asset_baseline          run_per_asset_baseline

Public CLI entry point: ``main(project_dir=None)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..experiments import exp_01_majority_vote, exp_02_bootstrap, exp_03_hypothesis_tests
from ..experiments import exp_04_simulation_calibration, exp_05_calm_subsample, exp_06_var_uplift
from ..experiments import exp_11_garch_ms_calibration, exp_12_kxw_sweep
from ..experiments import exp_13_k3_baseline, exp_14_block_sweep_gld, exp_15_disagree_config
from ..experiments import exp_16_asym_persistence_baseline, exp_17_em_restart_placebo
from ..experiments import exp_18_per_asset_baseline

logger = logging.getLogger(__name__)

_STEPS = (
    ("[ 1/15] Majority-vote",                  exp_01_majority_vote.main),
    ("[ 2/15] Bootstrap 5d windows",           exp_02_bootstrap.main),
    ("[ 3/15] Hypothesis tests",               exp_03_hypothesis_tests.main),
    ("[ 4/15] Simulation calibration",         exp_04_simulation_calibration.main),
    ("[ 5/15] GARCH(1,1)-MS calibration",      exp_11_garch_ms_calibration.main),
    ("[ 6/15] K=3 baseline",                   exp_13_k3_baseline.main),
    ("[ 7/15] Asymmetric persistence baseline", exp_16_asym_persistence_baseline.main),
    ("[ 8/15] Calm-day subsample",             exp_05_calm_subsample.main),
    ("[ 9/15] VaR uplift",                     exp_06_var_uplift.main),
    ("[10/15] K x window-scale sweep",         exp_12_kxw_sweep.main),
    ("[11/15] GLD block-sweep",                exp_14_block_sweep_gld.main),
    ("[12/15] SPY/QQQ/CL/USDJPY block-sweep", exp_14_block_sweep_gld.main_assets),
    ("[13/15] Disagree-day config",            exp_15_disagree_config.main),
    ("[14/15] EM-restart placebo",             exp_17_em_restart_placebo.main),
    ("[15/15] Per-asset full ML baseline",     exp_18_per_asset_baseline.main),
)


def main(project_dir: Path | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for label, step in _STEPS:
        logger.info(label)
        step(project_dir)


if __name__ == "__main__":
    main()
