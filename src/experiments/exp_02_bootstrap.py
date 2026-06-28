"""
Bootstrap test over 1,000 random 5-day windows (paper experiment).

Measures the sampling variability of the mean off-diagonal cross-frequency
ARI by resampling 5-day calendar windows from the full data, holding the
GMM regime fits fixed at the full-sample boundary. Used to test whether
the dissonance signal in event windows persists under window resampling.

This module reports two complementary statistics in
``bootstrap_five_day_windows.csv``:

1. **5-day window resampling** (``boot_*``): random contiguous 5-day calendar
   windows. The aligned label dict is ffill'd to the 5m grid, so within a
   5-day window every frequency (including 1d) has well above the
   ``min_valid=10`` floor in ``cross_freq_ari_matrix``; the resulting mean
   is the **full 6-pair** statistic computed on the aligned 5m-indexed
   labels restricted to the window.

2. **Block bootstrap** (``boot_block_*``): a true non-parametric block
   bootstrap on the aligned 5m-indexed label dict, using contiguous blocks
   of ``DEFAULT_BLOCK_SIZE`` 5m bars (default 50 bars, ~4 hours). Sampling
   is contiguous-block-with-replacement-of-position. The resampled index
   spans the full sample length, so the resulting mean off-diagonal ARI is
   also a **6-pair** statistic and is the recommended CI for the headline
   ARI in supplement Table A.16. The ``boot_*`` and ``boot_block_*``
   families differ in the resampling unit (calendar 5-day window vs.
   contiguous 5m-bar block), not in the number of pairs entering the mean.

Additionally reports two ``matched_5d_*_ari`` values: the off-diagonal ARI
on a *matched* 5 consecutive trading days within each event/calm window,
selected by ``select_matched_5day_window``. This addresses the manuscript
claim that the supplement comparison is on matched 5-day spans rather than
on the full episode windows.

Public entry point: ``run_bootstrap(raw_dir, outputs_dir, assets, calm_window, stress_window, n_boot)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import project_layout
from ..core.config import EPISODES
from ..data.data_ib import iter_loaded_assets
from ..core.features import daily_realised_vol
from ..core.models import fit_aligned_regimes
from ..core.config import FREQS, TZ
from ..core.metrics import cross_freq_ari_matrix, mean_offdiag_ari
from ..core.time_utils import subset_index_by_dates

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 50
DEFAULT_PERM_SEED = 42


def select_matched_5day_window(
    df_5m: pd.DataFrame,
    search_window: tuple[str, str],
    mode: Literal["stress", "calm"],
    days: int = 5,
) -> pd.DatetimeIndex:
    """Select a contiguous ``days``-trading-day window matched on realised vol.

    Within ``search_window`` (NY-calendar inclusive ``(start_date, end_date)``),
    enumerate all contiguous spans of ``days`` consecutive trading days
    (consecutive in the order they appear in ``df_5m``'s daily-normalised index,
    not by calendar date). For each candidate span compute the average daily
    realised volatility, where per-day RV is
    ``sqrt(sum_t r_t^2)`` from the 5m log returns of that day (matching the
    definition used by ``exp_05_calm_subsample.calm_day_subsample_ari``).

    * ``mode="stress"``: return the span with the highest mean daily RV.
    * ``mode="calm"``: return the span whose mean daily RV is closest to the
      median of the candidate-span means.

    Returns the 5m-bar timestamps belonging to the selected span. If
    ``search_window`` contains fewer than ``days`` trading days, returns the
    full search window's 5m index (so callers can still compute a usable
    ARI; the calling code logs a warning in that case).
    """
    if mode not in {"stress", "calm"}:
        raise ValueError(f"mode must be 'stress' or 'calm', got {mode!r}")

    base_idx = subset_index_by_dates(df_5m.index, search_window[0], search_window[1])
    if len(base_idx) == 0:
        return base_idx

    idx_ny = base_idx.tz_convert(TZ)
    day_key_full = pd.Series(df_5m.index.tz_convert(TZ).normalize(), index=df_5m.index)

    # Use the canonical per-day RV helper (drops O=H=L=C placeholder
    # bars before the sum). For 2022 GLD this shifts ~12/62 calm-day
    # vs non-calm-day classifications, and propagates the same shift
    # into the matched-5d window selection here.
    rv_per_day = daily_realised_vol(df_5m)

    days_in_window = pd.DatetimeIndex(sorted(set(idx_ny.normalize()))).intersection(
        rv_per_day.index
    )
    if len(days_in_window) < days:
        return base_idx

    n = len(days_in_window)
    span_means: list[tuple[int, float]] = []
    for start in range(n - days + 1):
        span_days = days_in_window[start:start + days]
        mean_rv = float(rv_per_day.loc[span_days].mean())
        span_means.append((start, mean_rv))

    means_arr = np.asarray([m for _, m in span_means], dtype=float)
    if mode == "stress":
        chosen = int(np.argmax(means_arr))
    else:  # calm: closest to median of candidate means
        med = float(np.median(means_arr))
        chosen = int(np.argmin(np.abs(means_arr - med)))

    chosen_days = set(days_in_window[chosen:chosen + days])
    mask = day_key_full.loc[base_idx].isin(chosen_days).values
    return base_idx[mask]


def block_bootstrap_mean_ari(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
    n_boot: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = DEFAULT_PERM_SEED,
) -> np.ndarray:
    """Block bootstrap the mean off-diagonal ARI on a 5m-aligned label dict.

    Resamples contiguous blocks of ``block_size`` 5m bars from the original
    aligned 5m index. Each draw concatenates ``ceil(N / block_size)`` blocks
    (with replacement of starting position) and trims to the original length
    ``N``, then evaluates the mean off-diagonal ARI on the resampled block
    sequence via ``cross_freq_ari_matrix``.

    Because the resampled length matches the full sample, all four
    frequencies retain enough non-null labels and the resulting mean is the
    **full 6-pair** statistic (in contrast to the 5-day window resampler,
    which is intraday-only-3-pair).

    Returns the array of finite bootstrap statistics (length ``<= n_boot``).
    """
    base_freq = freqs[0]
    base_index = aligned[base_freq].index
    n = len(base_index)
    if n < block_size * 2:
        return np.array([], dtype=float)

    for f in freqs:
        if len(aligned[f]) != n:
            raise ValueError(f"block_bootstrap_mean_ari: aligned['{f}'] length {len(aligned[f])} != aligned['{freqs[0]}'] length {n}")
    n_blocks_needed = int(np.ceil(n / block_size))
    n_starts = n - block_size + 1
    rng = np.random.default_rng(seed + 1)

    out = np.empty(int(n_boot), dtype=float)
    for k in range(int(n_boot)):
        starts = rng.integers(0, n_starts, size=n_blocks_needed)
        positions = np.concatenate([
            np.arange(s, s + block_size, dtype=np.int64) for s in starts
        ])[:n]
        sub_index = base_index[positions]
        # Resample each label series in lockstep so within-block alignment
        # across frequencies is preserved.
        resampled = {f: aligned[f].iloc[positions].reset_index(drop=True) for f in freqs}
        # Re-attach a synthetic monotone index so cross_freq_ari_matrix's
        # reindex-aligned semantics still apply (positions, not timestamps).
        synth_index = pd.RangeIndex(n)
        resampled = {f: pd.Series(s.values, index=synth_index) for f, s in resampled.items()}
        ari_mat = cross_freq_ari_matrix(resampled, freqs, synth_index)
        out[k] = mean_offdiag_ari(ari_mat)

    finite_out = out[np.isfinite(out)]
    if len(finite_out) < 0.9 * len(out):
        import warnings as _w
        _w.warn(f"block_bootstrap_mean_ari: only {len(finite_out)}/{len(out)} replicates were finite", RuntimeWarning)
    return finite_out


def bootstrap_five_day_windows(
    df_5m: pd.DataFrame,
    stem: str,
    calm_window: tuple[str, str],
    stress_window: tuple[str, str],
    n_boot: int = 1000,
    window_days: int = 5,
    seed: int = DEFAULT_PERM_SEED,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, float]:
    """Bootstrap the mean off-diagonal ARI over random 5-day windows.

    GMM regimes are first fitted per frequency on the full sample so that
    the bootstrap measures sampling variability of ARI conditional on the
    fitted boundary, not on a re-fitted boundary per resample.

    Reports three ARI families (see module docstring for details):

    * ``calm_ari`` / ``stress_ari``: full episode-window ARIs.
    * ``matched_5d_calm_ari`` / ``matched_5d_stress_ari``: ARIs on a matched
      5 trading-day span selected by realised volatility within the
      respective episode window.
    * ``boot_*``: 5-day window resampling stats (intraday-only, 3-pair).
    * ``boot_block_*``: block bootstrap stats with ``block_size`` 5m bars
      (full 6-pair).

    The keys ``tail_rank_calm_vs_boot`` / ``tail_rank_stress_vs_boot`` are
    the share of bootstrap window-mean draws at least as far from the
    bootstrap median as the observed value (a *tail rank*, not a null-
    hypothesis p-value). The legacy keys ``p_calm_vs_boot`` /
    ``p_stress_vs_boot`` are emitted with the same value as deprecated
    aliases for backward compatibility (P1-13).
    """
    aligned = fit_aligned_regimes(df_5m, stem, FREQS)

    calm_idx = subset_index_by_dates(df_5m.index, calm_window[0], calm_window[1])
    stress_idx = subset_index_by_dates(df_5m.index, stress_window[0], stress_window[1])
    calm_ari = mean_offdiag_ari(cross_freq_ari_matrix(aligned, FREQS, calm_idx)) if len(calm_idx) else None
    stress_ari = mean_offdiag_ari(cross_freq_ari_matrix(aligned, FREQS, stress_idx)) if len(stress_idx) else None

    # Matched 5-trading-day windows inside each episode window (P0-1).
    matched_calm_idx = select_matched_5day_window(df_5m, calm_window, mode="calm", days=window_days)
    matched_stress_idx = select_matched_5day_window(df_5m, stress_window, mode="stress", days=window_days)
    matched_calm_ari = (
        mean_offdiag_ari(cross_freq_ari_matrix(aligned, FREQS, matched_calm_idx))
        if len(matched_calm_idx) else None
    )
    matched_stress_ari = (
        mean_offdiag_ari(cross_freq_ari_matrix(aligned, FREQS, matched_stress_idx))
        if len(matched_stress_idx) else None
    )

    idx_ny = df_5m.index.tz_convert(TZ)
    days = pd.DatetimeIndex(idx_ny.normalize().unique()).sort_values()

    # Build the per-bar day key once; the previous version re-built this
    # million-row Series inside the n_boot loop.
    day_key = pd.Series(idx_ny.normalize(), index=df_5m.index)

    if len(days) < window_days + 1:
        return {
            "calm_ari": calm_ari, "stress_ari": stress_ari,
            "matched_5d_calm_ari": matched_calm_ari,
            "matched_5d_stress_ari": matched_stress_ari,
            "n_boot": 0,
        }

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(int(n_boot), dtype=float)
    feasible = len(days) - window_days + 1
    for i in range(int(n_boot)):
        start_pos = int(rng.integers(0, feasible))
        win_days = days[start_pos:start_pos + window_days]
        mask = day_key.isin(win_days).values
        sub = df_5m.index[mask]
        if len(sub) < 50:
            boot_stats[i] = np.nan
            continue
        boot_stats[i] = mean_offdiag_ari(cross_freq_ari_matrix(aligned, FREQS, sub))

    boot_stats = boot_stats[np.isfinite(boot_stats)]

    # P0-3: true block bootstrap on the full-sample aligned label dict.
    block_stats = block_bootstrap_mean_ari(
        aligned, FREQS, n_boot=int(n_boot),
        block_size=block_size, seed=seed,
    )

    result: dict[str, float] = {
        "calm_ari": calm_ari,
        "stress_ari": stress_ari,
        "matched_5d_calm_ari": matched_calm_ari,
        "matched_5d_stress_ari": matched_stress_ari,
        "n_boot": int(len(boot_stats)),
        "boot_mean": float(np.mean(boot_stats)) if boot_stats.size else np.nan,
        "boot_median": float(np.median(boot_stats)) if boot_stats.size else np.nan,
        "boot_q025": float(np.quantile(boot_stats, 0.025)) if boot_stats.size else np.nan,
        "boot_q975": float(np.quantile(boot_stats, 0.975)) if boot_stats.size else np.nan,
        "n_boot_block": int(len(block_stats)),
        "boot_block_mean": float(np.mean(block_stats)) if block_stats.size else np.nan,
        "boot_block_median": float(np.median(block_stats)) if block_stats.size else np.nan,
        "boot_block_q025": float(np.quantile(block_stats, 0.025)) if block_stats.size else np.nan,
        "boot_block_q975": float(np.quantile(block_stats, 0.975)) if block_stats.size else np.nan,
        "block_size_5m_bars": int(block_size),
    }
    # Matched-5d observed values for tail-rank apples-to-apples comparison
    # against the 5-day window bootstrap distribution.
    calm_ari_5d = matched_calm_ari if matched_calm_ari is not None else float("nan")
    stress_ari_5d = matched_stress_ari if matched_stress_ari is not None else float("nan")
    result["calm_ari_5d_matched"] = calm_ari_5d
    result["stress_ari_5d_matched"] = stress_ari_5d

    if not (isinstance(calm_ari_5d, float) and np.isnan(calm_ari_5d)) and boot_stats.size:
        dev = abs(calm_ari_5d - result["boot_median"])
        tail_count = int(np.sum(np.abs(boot_stats - result["boot_median"]) > dev))
        tail_rank = float((1 + tail_count) / (1 + len(boot_stats)))
        # P1-13: report as a tail rank rather than a p-value; keep the
        # legacy key as a deprecated alias.
        result["tail_rank_calm_vs_boot"] = tail_rank
        result["p_calm_vs_boot"] = tail_rank  # DEPRECATED alias (P1-13).
    if not (isinstance(stress_ari_5d, float) and np.isnan(stress_ari_5d)) and boot_stats.size:
        dev = abs(stress_ari_5d - result["boot_median"])
        tail_count = int(np.sum(np.abs(boot_stats - result["boot_median"]) > dev))
        tail_rank = float((1 + tail_count) / (1 + len(boot_stats)))
        result["tail_rank_stress_vs_boot"] = tail_rank
        result["p_stress_vs_boot"] = tail_rank  # DEPRECATED alias (P1-13).
    return result


def run_bootstrap(
    raw_dir: Path,
    outputs_dir: Path,
    assets: Iterable[str],
    calm_window: tuple[str, str],
    stress_window: tuple[str, str],
    n_boot: int = 1000,
) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _symbol, stem, df_5m in iter_loaded_assets(raw_dir, list(assets)):
        try:
            res = bootstrap_five_day_windows(
                df_5m, stem,
                calm_window=calm_window, stress_window=stress_window,
                n_boot=n_boot,
            )
            res["symbol"] = stem
            rows.append(res)
        except Exception:
            logger.exception("bootstrap: failed for %s; skipping", stem)
            continue
    summary = pd.DataFrame(rows)
    summary.to_csv(outputs_dir / "bootstrap_five_day_windows.csv", index=False)
    logger.info("Saved bootstrap summary with %d rows", len(summary))
    return summary


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stress_2026, calm_2026 = EPISODES["2026_iran"]
    stress_2022, calm_2022 = EPISODES["2022_ukraine"]
    logger.info("Bootstrap five-day windows (2026)")
    run_bootstrap(
        layout.raw_dir, layout.outputs_dir, layout.assets,
        calm_window=calm_2026, stress_window=stress_2026, n_boot=1000,
    )
    if layout.raw_dir_2022.exists():
        logger.info("Bootstrap five-day windows (2022)")
        run_bootstrap(
            layout.raw_dir_2022, layout.outputs_dir_2022, layout.assets_2022,
            calm_window=calm_2022, stress_window=stress_2022, n_boot=1000,
        )


if __name__ == "__main__":
    main()
