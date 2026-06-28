"""GARCH(1,1)-Markov-switching DGP simulation for the Paper 2 baseline.

Companion to ``exp_04_simulation_calibration`` (which uses an i.i.d.
Gaussian DGP per regime). Here the within-regime conditional variance
follows GARCH(1,1) with regime-specific unconditional variance and
shared persistence, so within-state heterogeneity (volatility clustering,
conditional fat tails) is present. The Markov chain on the regime label
is unchanged.

DGP parameters
--------------
- The MS-Gaussian regime parameters (mu, sigma2, P_12, P_21) come from
  ``outputs/calibrated_ms_params.json`` (a real ML fit on SPY 1h
  returns; see ``src.core.calibration.fit_and_persist_ms_params``).
- The GARCH(1,1) coefficients (alpha, beta) come from a single-regime
  GARCH fit on SPY 5m returns, persisted to
  ``outputs/calibrated_garch_params.json``.
- Per-regime omega is set on the fly so the within-regime stationary
  variance matches the MS-calibrated sigma_k^2.

Synthetic 5m paths are run through the *canonical* analysis pipeline
(``fit_regimes_per_frequency`` + ``cross_freq_ari_matrix``), so the
resulting ARI is directly comparable to the empirical and the i.i.d.
``exp_04`` baselines.

Public entry point: ``run_garch_ms_calibration(target_dir, ...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import project_layout
from ..core.calibration import (
    GARCH_CALIBRATION_FILENAME,
    fit_and_persist_garch_params,
    load_garch_params,
)
from ..core.sim_dgp import aggregate_garch_reps
from .exp_04_simulation_calibration import ensure_calibration as _ensure_calibration, DEFAULT_P_GRID

logger = logging.getLogger(__name__)


def _ensure_garch_calibration(
    target_dir: Path,
    raw_dir: Path | None = None,
    spy_filename: str = "SPY_5m.csv",
):
    """Load (or fit-and-persist) the calibrated GARCH(1,1) coefficients."""
    json_path = target_dir / GARCH_CALIBRATION_FILENAME
    if not json_path.exists():
        if raw_dir is None:
            raw_dir = project_layout().raw_dir
        spy_path = Path(raw_dir) / spy_filename
        logger.info(
            "GARCH calibration JSON missing at %s; fitting from %s.",
            json_path, spy_path,
        )
        fit_and_persist_garch_params(spy_path, json_path)
    return load_garch_params(json_path)


def run_garch_ms_calibration(
    target_dir: Path,
    n_reps: int = 200,
    P_grid: tuple[float, ...] | None = DEFAULT_P_GRID,
    seed: int = 4242,
    raw_dir: Path | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the MS-GARCH baseline sweep through the canonical pipeline.

    Defaults: n_reps=200, P_grid={0.005, 0.003, 0.001}, plus the calibrated
    point. The DGP is fully data-calibrated; alpha + beta are re-estimated
    from SPY 5m returns. The within-regime stationary variance equals the
    MS-calibrated sigma_k^2 by construction.
    """
    if P_grid:
        for _p in P_grid:
            if not (0 < _p < 1):
                raise ValueError(f"run_garch_ms_calibration: P_grid entries must be in (0, 1), got {_p}")
    target_dir.mkdir(parents=True, exist_ok=True)
    cal = _ensure_calibration(target_dir, raw_dir=raw_dir)
    garch = _ensure_garch_calibration(target_dir, raw_dir=raw_dir)
    logger.info(
        "MS-GARCH DGP: alpha=%.4f, beta=%.4f, alpha+beta=%.4f. "
        "Per-regime omega from MS sigma2_k * (1 - alpha - beta).",
        garch.alpha, garch.beta, garch.alpha + garch.beta,
    )

    sweep: list[tuple[str, float | None, float | None]] = [
        ("calibrated", None, None),
    ]
    if P_grid is not None:
        for p in P_grid:
            sweep.append((f"P={p:.4f}", float(p), float(p)))

    rows: list[dict[str, object]] = []
    for label, p12, p21 in sweep:
        agg = aggregate_garch_reps(
            cal, garch, n_reps=n_reps, n_components=2,
            P_12_override=p12, P_21_override=p21, seed=seed,
            n_jobs=n_jobs,
        )
        eff_p12 = float(cal.P_12 if p12 is None else p12)
        eff_p21 = float(cal.P_21 if p21 is None else p21)
        rows.append({
            "row_label": label,
            "P_12": eff_p12,
            "P_21": eff_p21,
            "alpha_garch": garch.alpha,
            "beta_garch": garch.beta,
            **agg,
        })
        logger.info(
            "%s: P_12=%.4f -> alt_mean_all4=%.3f intra=%.3f null=%.4f",
            label, eff_p12,
            rows[-1]["alt_mean_all4"], rows[-1]["alt_mean_intraday"],
            rows[-1]["null_mean_all4"],
        )

    df = pd.DataFrame(rows)
    df.to_csv(target_dir / "simulation_rss_garchms.csv", index=False)
    return df


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("GARCH(1,1)-Markov-switching baseline calibration (canonical pipeline)")
    run_garch_ms_calibration(layout.outputs_dir, raw_dir=layout.raw_dir)


if __name__ == "__main__":
    main()
