"""
K x window-scale robustness sweep for both episodes (2026 primary, 2022 OOS).

Wraps ``src.workflows.pipeline.run_robustness`` with the canonical Paper 2
sweep grid (K in {2, 3}, window-scale in {0.5x, 1.0x, 2.0x}) for the
2026 panel and the 2022 OOS panel. Writes
``robustness_summary.csv`` and ``robustness_ranges.csv`` per episode.

Public entry point: ``main()``. Registered in ``cli_registry`` as
``extended_kxw_sweep``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import project_layout
from ..workflows.pipeline import run_robustness

logger = logging.getLogger(__name__)


def _run_one(label: str, raw_dir: Path, outputs_dir: Path, assets: list[str]) -> None:
    logger.info("K x window sweep: %s", label)
    summary = run_robustness(
        raw_dir=raw_dir,
        outputs_dir=outputs_dir,
        assets=assets,
        k_values=(2, 3),
        window_scales=(0.5, 1.0, 2.0),
    )
    if summary.empty:
        logger.warning("Sweep produced no rows for %s; check raw_dir=%s", label, raw_dir)
        return
    cols = [c for c in ("symbol", "k", "window_scale", "overall_mean_ari") if c in summary.columns]
    logger.info("\n%s", summary[cols].to_string(index=False))


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("K x window-scale robustness sweep")

    _run_one(
        label="2026 primary",
        raw_dir=layout.raw_dir,
        outputs_dir=layout.outputs_dir,
        assets=list(layout.assets),
    )
    _run_one(
        label="2022 OOS",
        raw_dir=layout.raw_dir_2022,
        outputs_dir=layout.outputs_dir_2022,
        assets=list(layout.assets_2022),
    )


if __name__ == "__main__":
    main()
