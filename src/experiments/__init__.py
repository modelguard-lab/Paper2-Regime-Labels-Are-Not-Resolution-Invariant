"""Per-experiment modules for Paper 2.

Each ``exp_NN_<name>.py`` contains the implementation of one paper experiment
and exposes a ``main(project_dir=None)`` CLI entry point so it can be run
individually:

    python run.py extended_majority_vote
    python run.py extended_bootstrap

The orchestrator at :mod:`src.workflows.pipeline_ext` calls each of
``main()`` in sequence to reproduce the full extended-analyses suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class ProjectLayout(NamedTuple):
    raw_dir: Path
    outputs_dir: Path
    raw_dir_2022: Path
    outputs_dir_2022: Path
    assets: tuple[str, ...]
    assets_2022: tuple[str, ...] = ("SPY", "USDJPY", "GLD")


def project_layout(project_dir: Path | None = None) -> ProjectLayout:
    """Resolve the standard 2026/2022 raw_dir / outputs_dir layout.

    Used by every ``exp_NN_*.main()`` so the project-root resolution is
    not duplicated across experiment modules.
    """
    project_dir = project_dir or Path(__file__).resolve().parents[2]
    return ProjectLayout(
        raw_dir=project_dir / "data",
        outputs_dir=project_dir / "outputs",
        raw_dir_2022=project_dir / "data_2022",
        outputs_dir_2022=project_dir / "outputs_2022",
        # CL (not "CL=F"): canonical_stem strips "=" so "CL=F" produced "CLF"
        # which collides with Cleveland-Cliffs and (more importantly) does not
        # match the "CL_5m.csv" filenames the main pipeline writes when the
        # config asset list uses bare "CL".
        # Round-2: QQQ added as a second equity probe (referee #5 in the
        # round-2 review). 2022 OOS keeps the 3-asset universe because IB
        # CONTFUT pre-2024 5m history is forward-fill placeholders and
        # QQQ 2022 5m bars were not pulled in this revision round.
        assets=("SPY", "QQQ", "USDJPY", "CL", "GLD"),
        assets_2022=("SPY", "USDJPY", "GLD"),
    )
