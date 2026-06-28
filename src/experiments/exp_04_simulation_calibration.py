"""Simulation-based RSS calibration from a 2-state Markov-switching Gaussian DGP.

Generates synthetic 5m return paths from a calibrated 2-state Markov-switching
DGP, builds a synthetic OHLC frame on a real RTH 5m DatetimeIndex, runs that
frame through the *canonical* analysis pipeline (``fit_regimes_per_frequency``
+ ``cross_freq_ari_matrix``), and reports the cross-frequency ARI under
both the alternative (true regimes) and a null (coarse labels permuted).

Design contract: the synthetic ARI reported here is on the *same scale* as
the empirical ARI reported by ``analyze_asset`` for SPY / USDJPY / CL / GLD,
because the same recipe (``window_spec`` features, GMM K-means warmstart,
``robust_filter_returns``, ``cross_freq_ari_matrix`` reindex-aligned) is
applied to both. This resolves the comparability concern P0-4 in
REVIEW_FINDINGS.md.

DGP parameters: loaded from ``outputs/calibrated_ms_params.json`` (written
by ``src.core.calibration.fit_and_persist_ms_params`` from a real ML fit on
SPY 1h log returns). The persistence-rate sweep ``P_grid`` overrides the
calibrated transition probabilities for sensitivity analysis; pass
``P_grid=None`` to use the calibrated rates only.

Public entry point: ``run_simulation_calibration(target_dir, ...)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import project_layout
from ..core.calibration import (
    CALIBRATION_FILENAME,
    fit_and_persist_ms_params,
    load_ms_params,
)
from ..core.sim_dgp import aggregate_reps

logger = logging.getLogger(__name__)


# Default sweep grid over the per-bar transition probability at the 5m
# scale. Symmetric (P_12 = P_21 = p) so the stationary crisis share is fixed
# at 50% across the sweep, isolating the effect of persistence.
DEFAULT_P_GRID: tuple[float, ...] = (0.005, 0.003, 0.001)


def ensure_calibration(
    target_dir: Path,
    raw_dir: Path | None = None,
    spy_filename: str = "SPY_5m.csv",
):
    """Load (or fit-and-persist) the calibrated MS-Gaussian DGP.

    If ``outputs/calibrated_ms_params.json`` is missing, fit it from the
    SPY 5m CSV in ``raw_dir`` (or, if ``raw_dir`` is None, fall back to
    ``project_layout().raw_dir``). The fit step is idempotent: rerunning
    with the same data overwrites the JSON with the same values.
    """
    json_path = target_dir / CALIBRATION_FILENAME
    if not json_path.exists():
        if raw_dir is None:
            raw_dir = project_layout().raw_dir
        spy_path = Path(raw_dir) / spy_filename
        logger.info(
            "Calibration JSON missing at %s; fitting from %s.",
            json_path, spy_path,
        )
        fit_and_persist_ms_params(spy_path, json_path)
    return load_ms_params(json_path)


def run_simulation_calibration(
    target_dir: Path,
    n_reps: int = 200,
    P_grid: tuple[float, ...] | None = DEFAULT_P_GRID,
    seed: int = 42,
    baseline_p: float | None = None,
    raw_dir: Path | None = None,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Run the calibrated Gaussian-MS sweep through the canonical pipeline.

    Parameters
    ----------
    target_dir : Path
        Output directory; ``simulation_rss.csv`` and
        ``simulation_detection_rate.csv`` are written here.
    n_reps : int, default 200
        Number of replications per ``P_grid`` row.
    P_grid : tuple of float, optional
        Per-bar 5m transition probabilities to sweep (symmetric: P_12 = P_21
        = p). Pass ``None`` to use only the calibrated rates from the JSON.
    seed : int, default 42
        Master RNG seed for reproducibility.
    baseline_p : float, optional
        ``P_grid`` value used for the detection-rate companion sweep. Defaults
        to the median of ``P_grid`` (or the calibrated P_12 if ``P_grid`` is
        None).
    raw_dir : Path, optional
        Override the raw-data directory used to locate ``SPY_5m.csv`` for the
        one-shot ML calibration step.
    n_jobs : int, default 1
        Worker count forwarded to :func:`aggregate_reps` and the
        detection-rate replication loop. ``1`` is serial (default;
        bit-identical to the legacy implementation); ``-1`` uses every
        available core. Per-rep RNG is seeded from ``SeedSequence(seed +
        1).spawn(n_reps)[i]`` so results are bit-stable across worker
        counts.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    cal = ensure_calibration(target_dir, raw_dir=raw_dir)

    # Always include the calibrated point in the table; sweep extras are stacked
    # below and labelled with absolute P. The "calibrated_baseline" row is the
    # apples-to-apples comparison the paper now relies on.
    sweep: list[tuple[str, float | None, float | None]] = [
        ("calibrated", None, None),  # uses cal.P_12, cal.P_21 directly
    ]
    if P_grid is not None:
        for p in P_grid:
            sweep.append((f"P={p:.4f}", float(p), float(p)))

    rows: list[dict[str, object]] = []
    for label, p12, p21 in sweep:
        agg = aggregate_reps(
            cal, n_reps=n_reps, n_components=2,
            P_12_override=p12, P_21_override=p21, seed=seed,
            n_jobs=n_jobs,
        )
        # Resolve effective transition probabilities for the row.
        eff_p12 = float(cal.P_12 if p12 is None else p12)
        eff_p21 = float(cal.P_21 if p21 is None else p21)
        rows.append({
            "row_label": label,
            "P_trans": eff_p12 if p12 is None else float(p12),
            "P_12": eff_p12,
            "P_21": eff_p21,
            **agg,
        })
        logger.info(
            "%s: P_12=%.4f P_21=%.4f -> alt_mean_all4=%.3f intra=%.3f "
            "(null_all4=%.4f, n=%d)",
            label, eff_p12, eff_p21,
            rows[-1]["alt_mean_all4"], rows[-1]["alt_mean_intraday"],
            rows[-1]["null_mean_all4"], rows[-1]["n_reps_used"],
        )

    df = pd.DataFrame(rows)
    df.to_csv(target_dir / "simulation_rss.csv", index=False)

    # Detection-rate companion sweep: same DGP at one chosen baseline P, count
    # reps where intraday-only ARI falls below each delta threshold.
    if baseline_p is None:
        if P_grid:
            for _p in P_grid:
                if not (0 < _p < 1):
                    raise ValueError(f"P_grid entries must be in (0, 1), got {_p}")
            sorted_p = sorted(P_grid)
            baseline_p = float(sorted_p[len(sorted_p) // 2])
        else:
            baseline_p = float(cal.P_12)

    det_master = np.random.SeedSequence(seed + 1)
    det_children = det_master.spawn(n_reps)
    from ..core.sim_dgp import run_one_sim_replication

    def _one_det(i: int) -> float:
        rng = np.random.default_rng(det_children[i])
        res = run_one_sim_replication(
            cal, rng, n_components=2,
            P_12_override=baseline_p, P_21_override=baseline_p,
        )
        return res.intraday_mean_ari

    if n_jobs == 1:
        det_values = [_one_det(i) for i in range(n_reps)]
    else:
        from joblib import Parallel, delayed
        det_values = Parallel(n_jobs=n_jobs)(
            delayed(_one_det)(i) for i in range(n_reps)
        )
    det_intra = np.fromiter(det_values, dtype=float, count=n_reps)
    det_intra = det_intra[np.isfinite(det_intra)]
    detection: list[dict[str, object]] = []
    for delta in (0.05, 0.10, 0.15, 0.20):
        rate = float(np.mean(det_intra < delta)) if det_intra.size else float("nan")
        detection.append({"delta_bar": delta, "detection_rate": rate})
    pd.DataFrame(detection).to_csv(
        target_dir / "simulation_detection_rate.csv", index=False,
    )

    # The headline "summary" row used by downstream callers / the supplement
    # table: prefer the calibrated row, falling back to the row matching
    # baseline_p.
    summary = next((r for r in rows if r["row_label"] == "calibrated"), rows[0])
    logger.info("Simulation done (calibrated baseline): %s", summary)
    return summary


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Simulation-based RSS calibration (canonical pipeline + ML-fit DGP)")
    run_simulation_calibration(
        layout.outputs_dir,
        n_reps=200,
        P_grid=DEFAULT_P_GRID,
        raw_dir=layout.raw_dir,
    )


if __name__ == "__main__":
    main()
