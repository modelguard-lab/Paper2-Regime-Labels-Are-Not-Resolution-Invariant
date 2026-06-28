"""
Block-permutation sweep (Suppl Table block_sweep).

Runs ``block_permute_pvalue_mean_offdiag_ari`` for a configurable
block-size grid on each asset's full-sample aligned-regime labels.
The default sweep covers GLD with the full grid {10, 25, 50, 100, 250,
500} 5m bars (Suppl Table tab:block_sweep, GLD row).

A second public entry point fills the per-asset cells of the same
table at the four block sizes the table reports for SPY/CL/USDJPY
(25, 50, 100, 250); the main pipeline records only the block=50
column for those assets, so the other three columns are reproduced
here under one canonical seed.

Public entry points:
- ``run_block_sweep_gld(raw_dir, outputs_dir, n_perm, seed)``
- ``run_block_sweep_assets(raw_dir, outputs_dir, assets, block_sizes, n_perm, seed)``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import project_layout
from ..core.models import fit_aligned_regimes
from ..core.stability import block_permute_pvalue_mean_offdiag_ari
from ..data.data_ib import iter_loaded_assets
from ..core.config import FREQS

logger = logging.getLogger(__name__)

# Block-size grid spans sub-session (10 = 50min) through multi-session
# (500 ≈ 2 trading days for SPY RTH). The default block_size of 50
# (~4h) is at the centre. The endpoints stress-test the null at scales
# above and below typical regime persistence.
BLOCK_SIZES: tuple[int, ...] = (10, 25, 50, 100, 250, 500)

# Block sizes that appear as columns in tab:block_sweep for the
# non-GLD assets. The main pipeline persists block=50 only; the other
# three columns are populated by ``run_block_sweep_assets``.
TABLE_BLOCK_SIZES: tuple[int, ...] = (25, 50, 100, 250)


def _sweep_one_asset(
    stem: str,
    aligned: dict[str, pd.Series],
    block_sizes: Iterable[int],
    n_perm: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bs in block_sizes:
        p, ci, obs = block_permute_pvalue_mean_offdiag_ari(
            aligned, FREQS, n_perm=n_perm, block_size=bs, seed=seed
        )
        if p is None or obs is None:
            logger.warning(
                "%s block=%d  could not test (block_permute returned None); skipping",
                stem, bs,
            )
            continue
        rows.append({
            "symbol": stem,
            "block_size": bs,
            "p_value": p,
            "obs_stat": obs,
            "null_ci_low": ci[0] if ci is not None else None,
            "null_ci_high": ci[1] if ci is not None else None,
        })
        logger.info(
            "%s block=%d  obs=%.4f  null95UB=%.4f  p=%.4f",
            stem, bs, obs, ci[1] if ci is not None else float("nan"), p,
        )
    return rows


def run_block_sweep_gld(
    raw_dir: Path,
    outputs_dir: Path,
    n_perm: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Block-permutation sweep over BLOCK_SIZES for GLD only.

    Returns the per-block-size statistics and writes
    ``outputs/gld_block_sweep.csv``. The CSV schema predates the
    multi-asset extension and so does not carry a ``symbol`` column.
    """
    aligned: dict[str, pd.Series] | None = None
    for _symbol, stem, df_5m in iter_loaded_assets(raw_dir, ["GLD"]):
        if stem != "GLD":
            continue
        aligned = fit_aligned_regimes(df_5m, stem, FREQS)
        break

    if aligned is None:
        raise FileNotFoundError(f"GLD raw data not found under {raw_dir}")

    rows = _sweep_one_asset("GLD", aligned, BLOCK_SIZES, n_perm, seed)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "symbol"} for r in rows])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "gld_block_sweep.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved GLD block-sweep: %s", out_path)
    return df


def run_block_sweep_assets(
    raw_dir: Path,
    outputs_dir: Path,
    assets: Iterable[str] = ("SPY", "QQQ", "CL", "USDJPY"),
    block_sizes: Iterable[int] = TABLE_BLOCK_SIZES,
    n_perm: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Block-permutation sweep over ``block_sizes`` for each asset.

    Default asset list covers the four non-GLD rows of
    Suppl Table tab:block_sweep at the four columns the table
    reports (25, 50, 100, 250). Output is written to
    ``outputs/block_sweep_assets.csv`` with one row per
    ``(symbol, block_size)`` pair.
    """
    asset_list = list(assets)
    rows: list[dict[str, object]] = []
    for _symbol, stem, df_5m in iter_loaded_assets(raw_dir, asset_list):
        aligned = fit_aligned_regimes(df_5m, stem, FREQS)
        rows.extend(_sweep_one_asset(stem, aligned, block_sizes, n_perm, seed))

    df = pd.DataFrame(rows)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "block_sweep_assets.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved multi-asset block-sweep: %s", out_path)
    return df


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("GLD block-permutation sweep (Suppl Table block_sweep, GLD row)")
    run_block_sweep_gld(layout.raw_dir, layout.outputs_dir)


def main_assets(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info(
        "Multi-asset block-permutation sweep (Suppl Table block_sweep, SPY/CL/USDJPY rows)"
    )
    run_block_sweep_assets(layout.raw_dir, layout.outputs_dir)


if __name__ == "__main__":
    main()
