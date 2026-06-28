"""K=3 calibrated Markov-Gaussian baseline (Suppl Table tab:k3_baseline).

Replicates ``exp_04_simulation_calibration``'s calibrated DGP and recipe
but with a K=3 GMM fit per frequency. The canonical pipeline binarises
each K=3 fit at the per-frequency stage (highest-mean cluster = crisis,
lower two collapsed to calm; see ``fit_regime`` in ``src.core.models``),
so the resulting cross-frequency ARI is directly comparable to the K=2
empirical reading in the K=3 row of Table tab:k3_baseline.

DGP parameters: loaded from ``outputs/calibrated_ms_params.json``. The
sweep grid mirrors ``exp_04`` so K=3 vs K=2 effects are isolated from
DGP differences.

Public entry point: ``run_k3_baseline(target_dir, ...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import project_layout
from ..core.sim_dgp import aggregate_reps
from .exp_04_simulation_calibration import ensure_calibration as _ensure_calibration

logger = logging.getLogger(__name__)


def run_k3_baseline(
    target_dir: Path,
    n_reps: int = 200,
    seed: int = 42,
    raw_dir: Path | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the K=3 calibrated Gaussian-MS baseline through the canonical pipeline.

    DGP remains 2-state (calibrated CalibrationAt5m); n_components=3 only changes the per-frequency GMM fit downstream.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    cal = _ensure_calibration(target_dir, raw_dir=raw_dir)

    # P_grid entries do not define a 3-state DGP cleanly; for K=3 baseline,
    # sweep over the calibrated point only.
    sweep: list[tuple[str, float | None, float | None]] = [
        ("calibrated", None, None),
    ]

    rows: list[dict[str, object]] = []
    for label, p12, p21 in sweep:
        agg = aggregate_reps(
            cal, n_reps=n_reps, n_components=3,
            P_12_override=p12, P_21_override=p21, seed=seed,
            n_jobs=n_jobs,
        )
        eff_p12 = float(cal.P_12 if p12 is None else p12)
        eff_p21 = float(cal.P_21 if p21 is None else p21)
        rows.append({
            "row_label": label,
            "K": 3,
            "P_12": eff_p12,
            "P_21": eff_p21,
            **agg,
        })
        logger.info(
            "%s: K=3 P_12=%.4f P_21=%.4f -> alt_mean_all4=%.4f "
            "[q25 %.4f, q75 %.4f], intraday=%.4f",
            label, eff_p12, eff_p21,
            rows[-1]["alt_mean_all4"], rows[-1]["alt_q25_all4"],
            rows[-1]["alt_q75_all4"], rows[-1]["alt_mean_intraday"],
        )

    df = pd.DataFrame(rows)
    out_path = target_dir / "simulation_rss_k3.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved K=3 baseline: %s", out_path)
    return df


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("K=3 calibrated Markov-Gaussian baseline (canonical pipeline)")
    run_k3_baseline(layout.outputs_dir, raw_dir=layout.raw_dir)


if __name__ == "__main__":
    main()
