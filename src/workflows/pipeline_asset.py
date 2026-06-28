"""Per-asset multi-frequency analysis (Tier 1 paper recipe).

Houses :func:`analyze_asset`, the canonical full-sample analyse-once
function and its private fit/diagnostic helpers. The module-level orchestrator
:func:`src.workflows.pipeline.run_asset` calls this once per (asset, model)
pair and persists the returned :class:`AssetAnalysis` dict via
``pipeline_io``.

The split from ``pipeline.py`` is structural: this module owns the analytic
recipe; ``pipeline.py`` owns I/O + the loop over assets.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from ..core.config import (
    BARS_PER_DAY,
    CALM_WINDOW,
    CL_ROLL_ASSETS,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_GMM_K,
    DEFAULT_PERM_N,
    DEFAULT_PERM_SEED,
    DEFAULT_ROLLING_DAYS,
    DEFAULT_WINDOW_SCALE,
    EVENT_WINDOW,
    EXPANDING_MIN_TRAIN_CEIL,
    EXPANDING_MIN_TRAIN_FLOOR,
    EXPANDING_MIN_TRAIN_FRACTION,
    EXPANDING_MIN_TRAIN_PER_STATE,
    FREQS,
    MIN_TRADING_DAYS,
    MODEL_GMM,
    TZ,
)
from ..data.data_ib import canonical_stem, load_5m_ohlc
from ..core.features import features, resample_ohlc
from ..core.metrics import (
    cross_freq_ari_matrix,
    cross_freq_extra_metrics,
    mean_offdiag_ari_with_counts,
)
from ..core.models import (
    DEFAULT_SEED,
    align_regimes_to_5m,
    fit_diagnostics,
    fit_regime_expanding,
    fit_regime_model,
    fit_regimes_per_frequency,
)
from ..core.diagnostics import tod_adjusted_volatility, tod_crisis_distribution
from ..core.stability import (
    block_permute_pvalue_mean_offdiag_ari,
    permute_pvalue_mean_offdiag_ari,
)
from ..core.aggregation import cl_roll_week_analysis, compute_daily_outputs
from ..core.time_utils import subset_index_by_dates

logger = logging.getLogger(__name__)


def _cleanup_asset_outputs(outputs_dir, stem: str) -> None:
    """Remove stale outputs for an asset when the current run skips it."""
    base_suffixes = (
        "cross_freq_ari.csv",
        "cross_freq_ami.csv",
        "cross_freq_vi.csv",
        "daily_summary.csv",
        "daily_pairwise_ari.csv",
        "rolling_7d_ari.csv",
        "rolling_7d_pairwise_ari.csv",
        "rolling_7d_ari.png",
        "event_cross_freq_ari.csv",
        "calm_cross_freq_ari.csv",
        "expanding_cross_freq_ari.csv",
        "calendar_window_cross_freq_ari.csv",
        "tod_adjusted_cross_freq_ari.csv",
        "tod_crisis_distribution.csv",
        "roll_week_ari.csv",
    )
    suffixes: list[str] = [
        "timeline.png",
        "hmm_timeline.png",
        "5m.csv", "5m_results.json",
        "fallback_triggers.csv",
    ]
    for s in base_suffixes:
        suffixes.append(s)
        suffixes.append(f"hmm_{s}")
    suffixes.append("gmm_diagnostics.csv")
    suffixes.append("hmm_diagnostics.csv")
    for suffix in suffixes:
        path = outputs_dir / f"{stem}_{suffix}"
        if path.exists():
            try:
                path.unlink()
                logger.info("Removed stale output %s", path)
            except (OSError, PermissionError) as e:
                logger.warning("Could not remove stale output %s: %s", path, e)


def _load_asset_5m(symbol: str, raw_dir, outputs_dir) -> tuple[str, pd.DataFrame] | None:
    """Load and validate the base 5m series for one asset."""
    stem = canonical_stem(symbol)
    path_5m = raw_dir / f"{stem}_5m.csv"
    if not path_5m.exists():
        logger.warning("Skip %s: no 5m file %s", symbol, path_5m)
        _cleanup_asset_outputs(outputs_dir, stem)
        return None

    df_5m = load_5m_ohlc(path_5m)
    if df_5m.empty or len(df_5m) < 20:
        logger.warning("Skip %s: too few bars", symbol)
        _cleanup_asset_outputs(outputs_dir, stem)
        return None
    if df_5m.index.tz is None:
        df_5m.index = df_5m.index.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    else:
        df_5m.index = df_5m.index.tz_convert(TZ)

    bars_per_day = BARS_PER_DAY.get(stem, 78)
    min_bars = bars_per_day * MIN_TRADING_DAYS
    if len(df_5m) < min_bars:
        logger.warning(
            "Skip %s: insufficient data (%d 5m bars < %d required for %d days at %d bars/day)",
            symbol,
            len(df_5m),
            min_bars,
            MIN_TRADING_DAYS,
            bars_per_day,
        )
        _cleanup_asset_outputs(outputs_dir, stem)
        return None
    return stem, df_5m


def _build_features_cache(
    df_5m: pd.DataFrame,
    stem: str,
    window_scale: float,
) -> dict[str, pd.DataFrame]:
    """Per-asset features cache: one entry per frequency in ``FREQS``.

    Built once at the top of :func:`analyze_asset` and reused by baseline
    fit, expanding-window fit, and GMM diagnostics. Each entry costs one
    ``resample_ohlc`` + one ``features`` call.
    """
    return {
        freq: features(resample_ohlc(df_5m, freq), freq, stem=stem, window_scale=window_scale)
        for freq in FREQS
    }


def _fit_baseline_regimes(
    df_5m: pd.DataFrame,
    stem: str,
    model: str,
    n_components: int,
    window_scale: float,
    features_by_freq: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, bool], dict[str, str]]:
    """Fit GMM/HMM regimes per frequency on the 5m+resampled features.

    Returns ``(regimes_by_freq, fallback_flags, fit_status)``. ``fit_status[freq]``
    is one of ``"normal" / "degenerate_skipped" / "pct_fallback"``; reading
    just ``fallback_flags`` cannot distinguish "model did not fit at all"
    from "model fit normally".
    """
    regimes_by_freq = fit_regimes_per_frequency(
        df_5m, stem, FREQS,
        n_components=n_components, model=model, window_scale=window_scale,
        features_by_freq=features_by_freq,
    )
    fallback_flags: dict[str, bool] = {}
    fit_status: dict[str, str] = {}
    for freq, labels in regimes_by_freq.items():
        fallback_flags[freq] = bool(labels.attrs.get("fallback_triggered", False))
        fit_status[freq] = labels.attrs.get("fit_status", "normal")
    return regimes_by_freq, fallback_flags, fit_status


def compute_crisis_shares(aligned: dict[str, pd.Series]) -> dict[str, float]:
    """Per-frequency share of bars labelled crisis (regime == 1), as a percentage.

    NaN when a frequency has no non-null labels.
    """
    shares: dict[str, float] = {}
    for freq in FREQS:
        s = aligned[freq].dropna()
        shares[freq] = float(100.0 * (s == 1).mean()) if not s.empty else np.nan
    return shares


def _log_crisis_shares(symbol: str, aligned: dict[str, pd.Series]) -> None:
    for freq in FREQS:
        s = aligned[freq].dropna()
        if s.empty:
            logger.warning("%s %s: no non-null labels (crisis share undefined)", symbol, freq)
            continue
        pct = 100.0 * (s == 1).mean()
        if pct < 1.0 or pct > 99.0:
            logger.warning("%s %s: crisis share %.1f%% (near trivial)", symbol, freq, pct)
        else:
            logger.info("%s %s: crisis share %.1f%%", symbol, freq, pct)


def _build_tod_robustness(
    df_5m: pd.DataFrame,
    stem: str,
    model: str,
    n_components: int,
    window_scale: float,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Refit regimes on time-of-day-adjusted vol; return aligned labels and intraday ARI."""
    tod_regimes_by_freq: dict[str, pd.Series] = {}
    for freq in FREQS:
        tod_feats = tod_adjusted_volatility(df_5m, freq, stem, window_scale=window_scale)
        tod_labels, _ = fit_regime_model(
            tod_feats, model=model, n_components=n_components, freq=freq,
            seed=DEFAULT_SEED,
        )
        tod_regimes_by_freq[freq] = tod_labels
    tod_aligned = align_regimes_to_5m(tod_regimes_by_freq, df_5m.index)
    tod_ari_full_df = cross_freq_ari_matrix(tod_aligned, FREQS)
    intraday_freqs = ["5m", "15m", "1h"]
    return tod_aligned, tod_ari_full_df.loc[intraday_freqs, intraday_freqs].copy()


def _build_expanding_regimes(
    df_5m: pd.DataFrame,
    stem: str,
    model: str,
    n_components: int,
    window_scale: float,
    features_by_freq: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, dict[str, Any]]]:
    """No-look-ahead expanding-window GMM/HMM fit per frequency."""
    expanding_regimes: dict[str, pd.Series] = {}
    expanding_diag: dict[str, dict[str, Any]] = {}
    for freq in FREQS:
        ohlc = resample_ohlc(df_5m, freq)
        n_obs = len(ohlc)
        # Adaptive warm-up so short-frequency samples (especially 1d) do not
        # collapse to all-missing expanding labels.
        min_train = min(
            EXPANDING_MIN_TRAIN_CEIL,
            max(
                EXPANDING_MIN_TRAIN_FLOOR,
                int(EXPANDING_MIN_TRAIN_FRACTION * n_obs),
                n_components * EXPANDING_MIN_TRAIN_PER_STATE,
            ),
        )
        min_train = min(min_train, max(5, n_obs - 5))
        if n_obs < min_train + 1:
            pass  # let downstream raise/skip naturally
        cached_feats = features_by_freq.get(freq) if features_by_freq is not None else None
        try:
            exp_labels, exp_info = fit_regime_expanding(
                ohlc, freq, stem=stem, model=model,
                n_components=n_components, window_scale=window_scale,
                min_train_bars=min_train, feats=cached_feats,
            )
        except ValueError as e:
            # fit_regime_expanding fail-louds when no OOS labels were produced
            # (e.g., 1d series too short for the warm-up).  Surface this as a
            # diagnostic without killing the whole pipeline run.
            logger.warning("expanding fit produced no OOS labels for %s %s: %s", stem, freq, e)
            exp_labels = pd.Series(np.nan, index=ohlc.index, dtype=float)
            exp_info = {
                "refit_count": 0,
                "min_train_bars_used": int(min_train),
                "status": "no_oos_labels",
            }
        expanding_regimes[freq] = exp_labels
        expanding_diag[freq] = exp_info
    return expanding_regimes, expanding_diag


def fit_diagnostics_per_freq(
    df_5m: pd.DataFrame,
    stem: str,
    model: str,
    n_components: int,
    window_scale: float,
    features_by_freq: dict[str, pd.DataFrame] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run model-appropriate fit-quality diagnostics (BIC/AIC/separation/overlap) per frequency."""
    out: dict[str, dict[str, Any]] = {}
    for freq in FREQS:
        ohlc = resample_ohlc(df_5m, freq)
        cached_feats = features_by_freq.get(freq) if features_by_freq is not None else None
        out[freq] = fit_diagnostics(
            ohlc, freq, stem=stem, model=model,
            n_components=n_components, window_scale=window_scale,
            feats=cached_feats,
        )
    return out


def _rolling_ari_percentiles(rolling_df: pd.DataFrame) -> tuple[float, float, float]:
    if rolling_df.empty:
        return np.nan, np.nan, np.nan
    ari_vals = rolling_df["mean_offdiag_ari"].dropna()
    if ari_vals.empty:
        return np.nan, np.nan, np.nan
    return (
        float(ari_vals.median()),
        float(ari_vals.quantile(0.25)),
        float(ari_vals.quantile(0.75)),
    )


class AssetAnalysis(TypedDict, total=False):
    """Typed return from analyze_asset. All fields are optional (total=False)
    because subsets are assembled incrementally before the final dict literal."""
    symbol: str
    model: str
    n_components: int
    window_scale: float
    rolling_days: int
    ari_matrix: pd.DataFrame
    event_ari_matrix: pd.DataFrame
    calm_ari_matrix: pd.DataFrame
    event_window: tuple[str, str]
    calm_window: tuple[str, str]
    regimes_aligned: dict[str, pd.Series]
    daily_df: pd.DataFrame
    daily_pair_df: pd.DataFrame
    rolling_df: pd.DataFrame
    rolling_pair_df: pd.DataFrame
    crisis_shares: dict[str, float]
    fallback_flags: dict[str, bool]
    fit_status: dict[str, str]
    ami_matrix: pd.DataFrame
    vi_matrix: pd.DataFrame
    tod_crisis_distribution: pd.DataFrame
    tod_adjusted_ari_matrix: pd.DataFrame
    tod_adjusted_mean_ari: float | None
    tod_adjusted_intraday_n_valid_pairs: int
    tod_adjusted_intraday_n_total_pairs: int
    overall_mean_ari_matrix: float | None
    overall_n_valid_pairs: int
    overall_n_total_pairs: int
    overall_mean_ari_perm_stat: float | None
    overall_mean_ari_pvalue_perm: float | None
    overall_mean_ari_null_ci: tuple[float, float] | None
    block_perm_observed_stat: float | None
    block_perm_pvalue: float | None
    block_perm_null_ci: tuple[float, float] | None
    event_mean_ari: float | None
    event_n_valid_pairs: int
    calm_mean_ari: float | None
    calm_n_valid_pairs: int
    latest_rolling_7d_mean_ari: float | None
    rolling_ari_median: float | None
    rolling_ari_q25: float | None
    rolling_ari_q75: float | None
    fit_diagnostics: dict[str, Any]
    expanding_ari_matrix: pd.DataFrame
    expanding_mean_ari: float | None
    expanding_n_valid_pairs: int
    expanding_diagnostics: list[dict[str, Any]]
    cl_roll_analysis: dict[str, Any] | None


def analyze_asset(
    symbol: str,
    df_5m: pd.DataFrame,
    model: str = MODEL_GMM,
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    rolling_days: int = DEFAULT_ROLLING_DAYS,
    event_window: tuple[str, str] | None = None,
    calm_window: tuple[str, str] | None = None,
    features_by_freq: dict[str, pd.DataFrame] | None = None,
) -> AssetAnalysis:
    """Run multi-frequency analysis for one already-loaded asset."""
    event_window = event_window or EVENT_WINDOW
    calm_window = calm_window or CALM_WINDOW
    stem = canonical_stem(symbol)
    if features_by_freq is None:
        features_by_freq = _build_features_cache(df_5m, stem, window_scale)

    regimes_by_freq, fallback_flags, fit_status = _fit_baseline_regimes(
        df_5m, stem, model, n_components, window_scale,
        features_by_freq=features_by_freq,
    )
    aligned = align_regimes_to_5m(regimes_by_freq, df_5m.index)
    _log_crisis_shares(symbol, aligned)

    tod_crisis = tod_crisis_distribution(aligned, FREQS)
    tod_aligned, tod_ari_df = _build_tod_robustness(
        df_5m, stem, model, n_components, window_scale
    )

    ari_df = cross_freq_ari_matrix(aligned, FREQS)
    extra_metrics = cross_freq_extra_metrics(aligned, FREQS)
    perm_p, perm_ci, perm_obs = permute_pvalue_mean_offdiag_ari(
        aligned, FREQS, n_perm=DEFAULT_PERM_N, seed=DEFAULT_PERM_SEED
    )
    event_index = subset_index_by_dates(df_5m.index, event_window[0], event_window[1])
    calm_index = subset_index_by_dates(df_5m.index, calm_window[0], calm_window[1])
    event_ari_df = cross_freq_ari_matrix(aligned, FREQS, event_index) if len(event_index) else pd.DataFrame()
    calm_ari_df = cross_freq_ari_matrix(aligned, FREQS, calm_index) if len(calm_index) else pd.DataFrame()
    daily_df, daily_pair_df, rolling_df, rolling_pair_df = compute_daily_outputs(
        aligned, FREQS, rolling_days,
    )
    crisis_shares = compute_crisis_shares(aligned)

    block_perm_p, block_perm_ci, block_perm_obs = block_permute_pvalue_mean_offdiag_ari(
        aligned, FREQS, n_perm=DEFAULT_PERM_N, block_size=DEFAULT_BLOCK_SIZE, seed=DEFAULT_PERM_SEED
    )
    fit_diag = fit_diagnostics_per_freq(
        df_5m, stem, model, n_components, window_scale, features_by_freq=features_by_freq,
    )
    expanding_regimes, expanding_diag = _build_expanding_regimes(
        df_5m, stem, model, n_components, window_scale, features_by_freq=features_by_freq,
    )
    expanding_aligned = align_regimes_to_5m(expanding_regimes, df_5m.index)
    expanding_ari_df = cross_freq_ari_matrix(expanding_aligned, FREQS)
    cl_roll_result = (
        cl_roll_week_analysis(aligned, df_5m.index, FREQS)
        if stem in CL_ROLL_ASSETS
        else None
    )
    rolling_ari_median, rolling_ari_q25, rolling_ari_q75 = _rolling_ari_percentiles(rolling_df)

    overall_mean_ari, overall_n_valid_pairs, overall_n_total_pairs = mean_offdiag_ari_with_counts(ari_df)
    tod_mean_ari, tod_n_valid_pairs, tod_n_total_pairs = mean_offdiag_ari_with_counts(tod_ari_df)
    event_mean_ari, event_n_valid_pairs, _ = mean_offdiag_ari_with_counts(event_ari_df)
    calm_mean_ari, calm_n_valid_pairs, _ = mean_offdiag_ari_with_counts(calm_ari_df)
    expanding_mean_ari, expanding_n_valid_pairs, _ = mean_offdiag_ari_with_counts(expanding_ari_df)

    return {
        "symbol": symbol,
        "model": str(model),
        "n_components": int(n_components),
        "window_scale": float(window_scale),
        "rolling_days": int(rolling_days),
        "ari_matrix": ari_df,
        "event_ari_matrix": event_ari_df,
        "calm_ari_matrix": calm_ari_df,
        "event_window_has_data": bool(len(event_index) > 0),
        "calm_window_has_data": bool(len(calm_index) > 0),
        "event_window": event_window,
        "calm_window": calm_window,
        "regimes_aligned": aligned,
        "daily_df": daily_df,
        "daily_pair_df": daily_pair_df,
        "rolling_df": rolling_df,
        "rolling_pair_df": rolling_pair_df,
        "crisis_shares": crisis_shares,
        "fallback_flags": fallback_flags,
        "fit_status": fit_status,
        "ami_matrix": extra_metrics["ami"],
        "vi_matrix": extra_metrics["vi"],
        "tod_crisis_distribution": tod_crisis,
        "tod_adjusted_ari_matrix": tod_ari_df,
        "tod_adjusted_mean_ari": tod_mean_ari,
        "tod_adjusted_intraday_n_valid_pairs": tod_n_valid_pairs,
        "tod_adjusted_intraday_n_total_pairs": tod_n_total_pairs,
        "overall_mean_ari_matrix": overall_mean_ari,
        "overall_n_valid_pairs": overall_n_valid_pairs,
        "overall_n_total_pairs": overall_n_total_pairs,
        "overall_mean_ari_perm_stat": perm_obs,
        "overall_mean_ari_pvalue_perm": perm_p,
        "overall_mean_ari_null_ci": perm_ci,
        "block_perm_observed_stat": block_perm_obs,
        "block_perm_pvalue": block_perm_p,
        "block_perm_null_ci": block_perm_ci,
        "event_mean_ari": event_mean_ari,
        "event_n_valid_pairs": event_n_valid_pairs,
        "calm_mean_ari": calm_mean_ari,
        "calm_n_valid_pairs": calm_n_valid_pairs,
        "latest_rolling_7d_mean_ari": None if rolling_df.empty else float(rolling_df["mean_offdiag_ari"].iloc[-1]),
        "rolling_ari_median": rolling_ari_median,
        "rolling_ari_q25": rolling_ari_q25,
        "rolling_ari_q75": rolling_ari_q75,
        "fit_diagnostics": fit_diag,
        "expanding_ari_matrix": expanding_ari_df,
        "expanding_mean_ari": expanding_mean_ari,
        "expanding_n_valid_pairs": expanding_n_valid_pairs,
        "expanding_diagnostics": expanding_diag,
        "cl_roll_analysis": cl_roll_result,
    }
