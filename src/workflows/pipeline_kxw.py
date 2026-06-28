"""K x window_scale parameter sensitivity sweep (robustness layer).

Trimmed-down :func:`_analyze_asset_lightweight` skips the two 500-iter
permutation tests, time-of-day refit, expanding-window fit, GMM
diagnostics, CL roll analysis, event/calm subwindows, and the HMM
companion fit. These are not consumed by the K x window_scale grid.

:func:`run_robustness` is the public entrypoint wired through
``extended_kxw_sweep`` in the CLI registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import (
    DEFAULT_GMM_K,
    DEFAULT_ROLLING_DAYS,
    FREQS,
    MODEL_GMM,
)
from ..data.data_ib import canonical_stem
from ..core.metrics import cross_freq_ari_matrix, mean_offdiag_ari
from ..core.models import align_regimes_to_5m
from ..core.aggregation import compute_daily_outputs
from .pipeline_asset import (
    _build_features_cache,
    compute_crisis_shares,
    _fit_baseline_regimes,
    _load_asset_5m,
)
from .pipeline_robustness import (
    _attach_robustness_baseline_deltas,
    _compute_robustness_ranges,
    _write_robustness_report,
)

logger = logging.getLogger(__name__)


def _analyze_asset_lightweight(
    symbol: str,
    df_5m: pd.DataFrame,
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = 1.0,
    rolling_days: int = DEFAULT_ROLLING_DAYS,
    features_by_freq: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Trimmed-down analyze_asset for K x window_scale parameter sweeps.

    Computes only the three statistics consumed by ``_sweep_kxwindow``
    (overall_mean_ari, latest_rolling_7d_mean_ari, crisis_shares).  Skips
    the two 500-iter permutation tests, time-of-day robustness refit,
    expanding-window fit, GMM fit-quality diagnostics, CL roll analysis,
    event/calm subwindow ARIs, and the HMM fit -- none of which are read
    by the sweep.  Default 3-asset x 2-K x 3-window_scale sweep drops from
    ~180 permutation tests + 18 expanding fits to zero.

    ``features_by_freq`` may be passed in to share features across K values
    at a given ``window_scale`` (features depend on ``window_scale`` only,
    not on K).
    """
    stem = canonical_stem(symbol)
    regimes_by_freq, _, _ = _fit_baseline_regimes(
        df_5m, stem, MODEL_GMM, n_components, window_scale,
        features_by_freq=features_by_freq,
    )
    aligned = align_regimes_to_5m(regimes_by_freq, df_5m.index)
    ari_df = cross_freq_ari_matrix(aligned, FREQS)
    _, _, rolling_df, _ = compute_daily_outputs(aligned, FREQS, rolling_days)
    # Per-pair ARI for the (15m, 1h) cell: this is the load-bearing pair for
    # the supplement claim that "the 15m->1h break is preserved at every cell"
    # of the K x window_scale sweep. Saved alongside overall_mean_ari so the
    # claim is verifiable from robustness_summary.csv directly.
    try:
        ari_15m_1h = float(ari_df.loc["15m", "1h"])
    except (KeyError, TypeError):
        ari_15m_1h = float("nan")
    return {
        "overall_mean_ari": mean_offdiag_ari(ari_df),
        "ari_15m_1h": ari_15m_1h,
        "latest_rolling_7d_mean_ari": (
            None if rolling_df.empty
            else float(rolling_df["mean_offdiag_ari"].iloc[-1])
        ),
        "crisis_shares": compute_crisis_shares(aligned),
    }


def _sweep_kxwindow(
    cached: dict[str, tuple[str, pd.DataFrame]],
    k_values: tuple[int, ...],
    window_scales: tuple[float, ...],
    rolling_days: int,
) -> list[dict[str, Any]]:
    """Cartesian sweep over (k_values x window_scales) per asset.

    Features depend on ``(window_scale, stem)`` only (not on K), so we
    iterate ``window_scale`` in the outer position and build the per-asset
    features cache once per ``window_scale``; the inner K loop reuses that
    cache. Halves the resample+features cost at K=(2,3).
    """
    rows: list[dict[str, Any]] = []
    for symbol, (stem, df_5m) in cached.items():
        for window_scale in window_scales:
            features_by_freq = _build_features_cache(df_5m, stem, window_scale)
            for n_components in k_values:
                analysis = _analyze_asset_lightweight(
                    symbol, df_5m,
                    n_components=n_components,
                    window_scale=window_scale,
                    rolling_days=rolling_days,
                    features_by_freq=features_by_freq,
                )
                row: dict[str, Any] = {
                    "symbol": symbol,
                    "k": int(n_components),
                    "window_scale": float(window_scale),
                    "window_label": f"{window_scale:.1f}x",
                    "overall_mean_ari": analysis["overall_mean_ari"],
                    "ari_15m_1h": analysis["ari_15m_1h"],
                    "latest_rolling_mean_ari": analysis["latest_rolling_7d_mean_ari"],
                }
                for freq, share in analysis["crisis_shares"].items():
                    row[f"crisis_share_{freq}"] = share
                rows.append(row)
    return rows


def run_robustness(
    raw_dir: Path | str,
    outputs_dir: Path | str,
    assets: list[str],
    k_values: tuple[int, ...] = (2, 3),
    window_scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    rolling_days: int = DEFAULT_ROLLING_DAYS,
) -> pd.DataFrame:
    """Run parameter sensitivity sweeps over K and rolling-volatility windows."""
    raw_dir = Path(raw_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[str, tuple[str, pd.DataFrame]] = {}
    for symbol in assets:
        loaded = _load_asset_5m(symbol, raw_dir, outputs_dir)
        if loaded is not None:
            cached[symbol] = loaded

    summary = pd.DataFrame(_sweep_kxwindow(cached, k_values, window_scales, rolling_days))
    if summary.empty:
        return summary

    summary = _attach_robustness_baseline_deltas(summary)
    summary.to_csv(outputs_dir / "robustness_summary.csv", index=False)

    ranges = _compute_robustness_ranges(summary)
    ranges.to_csv(outputs_dir / "robustness_ranges.csv", index=False)
    _write_robustness_report(ranges, outputs_dir)

    return summary
