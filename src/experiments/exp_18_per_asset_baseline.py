"""Per-asset full ML calibration baseline (peer-review M1).

The default calibrated baseline (`exp_04_simulation_calibration`) ML-fits
SPY 1h log returns and uses that single ``(P_12, P_21, mu_0, mu_1,
sigma_0, sigma_1)`` tuple as the canonical reference for every empirical
asset. The asymmetric-persistence baseline (`exp_16`) lets ``pi``
(stationary crisis share) vary per asset but keeps ``tau = P_12 + P_21``
fixed at the SPY fit. Both choices isolate one degree of variation while
holding the rest at SPY values.

This module gives each asset its own full ML fit: per-asset 1h returns
feed an independent MarkovRegression that returns asset-specific
``(mu_0, mu_1, sigma2_0, sigma2_1, P_12, P_21)``. The fitted DGP is then
run through the canonical pipeline (200 replications) to produce a
per-asset calibrated mean off-diagonal ARI, IQR, and intraday-only
mean / IQR.

The output table answers the referee question "is the SPY-1h baseline
generalisable to USD/JPY / CL / GLD / QQQ?" empirically rather than by
appeal to the symmetric-vs-asymmetric persistence robustness check.

Public entry points:

- :func:`run_per_asset_baseline` -- the per-asset sweep, writes
  ``outputs/simulation_rss_per_asset.csv``.
- :func:`main` -- CLI wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import project_layout
from ..core.calibration import (
    CALIBRATION_FILENAME,
    CalibratedMSParams,
    fit_and_persist_ms_params,
    load_ms_params,
)
from ..core.sim_dgp import aggregate_reps
from ..data.data_ib import canonical_stem

logger = logging.getLogger(__name__)


def run_per_asset_baseline(
    target_dir: Path,
    raw_dir: Path,
    assets: Iterable[str],
    n_reps: int = 200,
    seed_base: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Per-asset full ML calibration sweep.

    For each asset, ML-fit MS-Gaussian on its 1h returns, persist params
    to ``outputs/calibrated_ms_params_<stem>.json``, then run the
    canonical-pipeline simulation ``n_reps`` times and report the
    per-asset baseline.

    Different asset gets a different seed (``seed_base + i``) so the
    cross-asset comparisons are not Common-Random-Numbers-confounded;
    use a fixed ``seed_base`` for replication.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for i, asset in enumerate(assets):
        stem = canonical_stem(asset)
        asset_5m = Path(raw_dir) / f"{stem}_5m.csv"
        if not asset_5m.exists():
            logger.warning("per_asset_baseline: skip %s (no 5m data at %s)", stem, asset_5m)
            continue
        params_path = target_dir / f"calibrated_ms_params_{stem}.json"
        if not params_path.exists():
            logger.info("Fitting MS-Gaussian on %s 1h returns...", stem)
            fit_and_persist_ms_params(asset_5m, params_path)
        cal = load_ms_params(params_path)
        agg = aggregate_reps(
            cal, n_reps=n_reps, n_components=2, seed=seed_base + i,
            n_jobs=n_jobs,
        )
        # The 1h-fit P values from the persisted JSON are at 1h scale; the
        # 5m-scale P used by the simulator is in cal.P_12 / cal.P_21.
        raw: CalibratedMSParams = cal.raw  # type: ignore[assignment]
        rows.append({
            "asset": stem,
            "P_12_1h_fit": raw.P_12,
            "P_21_1h_fit": raw.P_21,
            "tau_1h_fit": raw.P_12 + raw.P_21,
            "mu_0_1h": raw.mu_0,
            "mu_1_1h": raw.mu_1,
            "sigma_0_1h": (raw.sigma2_0 ** 0.5),
            "sigma_1_1h": (raw.sigma2_1 ** 0.5),
            "P_12_5m": cal.P_12,
            "P_21_5m": cal.P_21,
            **agg,
        })
        logger.info(
            "%s: P_1h=(%.3f, %.3f) -> alt_mean_all4=%.4f [q25 %.4f, q75 %.4f] intra=%.4f",
            stem, raw.P_12, raw.P_21,
            rows[-1]["alt_mean_all4"], rows[-1]["alt_q25_all4"],
            rows[-1]["alt_q75_all4"], rows[-1]["alt_mean_intraday"],
        )
    df = pd.DataFrame(rows)
    out_path = target_dir / "simulation_rss_per_asset.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved per-asset full calibration baseline: %s", out_path)
    return df


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Per-asset full ML calibration baseline (M1)")
    run_per_asset_baseline(
        layout.outputs_dir,
        layout.raw_dir,
        layout.assets,
    )


if __name__ == "__main__":
    main()
