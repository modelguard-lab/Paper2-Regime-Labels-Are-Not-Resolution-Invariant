"""
Majority-vote upward aggregation across resolutions (paper experiment).

Symmetric counterpart to the forward-fill baseline used in the main
pipeline: instead of broadcasting coarse labels down to the fine 5m
grid, this module takes finer labels and aggregates them upward into
coarse-grid bins via majority vote.

Public entry point: ``run_majority_vote(raw_dir, outputs_dir, assets)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import project_layout
from ..data.data_ib import iter_loaded_assets
from ..core.models import fit_regimes_per_frequency
from ..core.config import FREQS, TZ
from ..core.metrics import mean_offdiag_ari

logger = logging.getLogger(__name__)


def _coarse_bin_keys(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Map each timestamp to the coarse bin it belongs to (as a timestamp).

    ``resample_ohlc`` builds 15m / 1h coarse bars with ``closed='right',
    label='right'`` (see ``core/features.py``), so the bar at timestamp ``T``
    aggregates fine timestamps in the half-open interval ``(T - bar, T]``.
    To match that semantics, fine timestamps must be ``ceil``'d to the
    next bin boundary, NOT ``floor``'d.

    The previous ``floor`` implementation bound a 5m bar at 10:05 to bin
    10:00 (which actually represents fine bars 09:50-10:00) and produced
    a systematic bin-width misalignment in cross-frequency ARI: ~5min at
    15m, ~1h at 1h. For 1d, ``resample_ohlc`` shifts the label to 16:00
    of the calendar day; ``normalize()`` reduces both fine and coarse
    keys to that day's midnight, so the 1d path is unaffected.

    Returns a tz-aware ``DatetimeIndex`` so the caller can attach it
    either to ``fine_idx`` (for groupby binning) or to ``coarse.index``
    (for reindex lookup) without an extra tz round-trip.
    """
    idx_ny = index.tz_convert(TZ) if index.tz is not None else index
    if freq == "5m":
        return idx_ny
    if freq == "15m":
        return idx_ny.ceil("15min")
    if freq == "1h":
        return idx_ny.ceil("1h")
    if freq == "1d":
        return idx_ny.normalize()
    raise ValueError(freq)


def majority_vote_ari(df_5m: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Symmetric counterpart to the forward-fill baseline.

    For each pair (fa, fb) with fa finer than fb, fit GMM independently at
    both frequencies and compare the labels at the coarser grid: finer
    labels are majority-aggregated upward into fb-sized bins. The output
    matrix is symmetric. Ties (50/50) resolve to crisis (1).
    """
    labels_per_freq = fit_regimes_per_frequency(df_5m, stem, FREQS)

    freqs = list(FREQS)
    n = len(freqs)
    ari = np.eye(n, dtype=float)
    for i, fa in enumerate(freqs):
        for j, fb in enumerate(freqs):
            if j <= i:
                continue
            fine = labels_per_freq[fa]
            coarse = labels_per_freq[fb]
            if fine.empty or coarse.empty:
                ari[i, j] = ari[j, i] = np.nan
                continue
            fine_idx = fine.index
            bins = _coarse_bin_keys(fine_idx, fb)
            df_bin = pd.DataFrame({"bin": bins, "lab": fine.values}, index=fine_idx)
            crisis_share = df_bin.groupby("bin")["lab"].mean()
            # Preserve NaN for bins whose entire fine-bar slice is NaN (data
            # gap / multi-bar warmup): a naive ``(NaN >= 0.5).astype(int)``
            # silently maps the empty bin to 0 (calm) and lets it through
            # the downstream mask, contaminating the ARI with spurious
            # "calm-vs-X" pair contributions.
            agg_to_coarse = (
                (crisis_share >= 0.5).astype(float).where(crisis_share.notna())
            )
            coarse_key = _coarse_bin_keys(coarse.index, fb)
            agg_aligned = pd.Series(agg_to_coarse.reindex(coarse_key).values, index=coarse.index)
            # Joint-NaN mask: ``coarse`` carries NaN at warm-up / non-finite-vol
            # bars (see fit_regime in core.models); ``agg_aligned`` carries NaN
            # for bins whose entire fine slice was NaN. Filtering only on one
            # side would let those NaNs into a numpy ``.astype(int)`` cast,
            # which is undefined behaviour (numpy 1.x silently produces
            # garbage; numpy 2.x raises).
            mask = agg_aligned.notna() & coarse.notna()
            if mask.sum() < 10:
                ari[i, j] = np.nan
                ari[j, i] = np.nan
                continue
            a_int = agg_aligned[mask].round().astype(int).values
            b_int = coarse[mask].round().astype(int).values
            score = adjusted_rand_score(a_int, b_int)
            ari[i, j] = ari[j, i] = float(round(np.clip(score, -1.0, 1.0), 6))
    return pd.DataFrame(ari, index=freqs, columns=freqs)


def run_majority_vote(raw_dir: Path, outputs_dir: Path, assets: Iterable[str]) -> pd.DataFrame:
    """Run majority-vote aggregation for all assets and write per-asset CSVs
    plus a summary CSV with mean off-diagonal ARI.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def _warn_missing(symbol, path):
        logger.warning("majority_vote: missing %s", path)

    for _symbol, stem, df_5m in iter_loaded_assets(
        raw_dir, list(assets), on_missing=_warn_missing
    ):
        try:
            if df_5m.index.tz is None:
                df_5m.index = df_5m.index.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
            else:
                df_5m.index = df_5m.index.tz_convert(TZ)

            ari_df = majority_vote_ari(df_5m, stem)
            ari_df.to_csv(outputs_dir / f"{stem}_majority_vote_cross_freq_ari.csv")
            rows.append({
                "symbol": stem,
                "mean_offdiag_ari_majority_vote": mean_offdiag_ari(ari_df),
            })
        except Exception:
            logger.exception("majority_vote: failed for %s; skipping", stem)
            continue
    summary = pd.DataFrame(rows)
    summary.to_csv(outputs_dir / "majority_vote_summary.csv", index=False)
    logger.info("Saved majority-vote summary with %d rows", len(summary))
    return summary


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Majority-vote (2026)")
    run_majority_vote(layout.raw_dir, layout.outputs_dir, layout.assets)
    if layout.raw_dir_2022.exists():
        logger.info("Majority-vote (2022)")
        run_majority_vote(layout.raw_dir_2022, layout.outputs_dir_2022, layout.assets_2022)


if __name__ == "__main__":
    main()
