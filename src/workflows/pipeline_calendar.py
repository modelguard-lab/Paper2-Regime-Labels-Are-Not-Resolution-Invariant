"""Calendar-window robustness check: refit regimes with a fixed wall-clock
rolling window for the volatility feature, instead of the bar-count window
the main pipeline uses.

Iterates intraday frequencies only (5m / 15m / 1h). The 1d frequency is
excluded by paper definition: a 6h rolling window on daily bars yields
all-NaN volatility, collapsing the daily fit to the calm fallback.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.config import DEFAULT_GMM_K, MODEL_GMM
from ..data.data_ib import canonical_stem
from ..core.features import features, resample_ohlc
from ..core.metrics import cross_freq_ari_matrix, mean_offdiag_ari
from ..core.models import align_regimes_to_5m, fit_regime_model

_INTRADAY_FREQS: tuple[str, ...] = ("5m", "15m", "1h")


def _build_calendar_window_features(
    df_5m: pd.DataFrame, stem: str, calendar_window: str = "6h",
) -> dict[str, pd.DataFrame]:
    """Build the calendar-window features cache (intraday only).

    Daily frequency is intentionally excluded: a 6h rolling window on a
    daily bar series gives all-NaN volatility (no prior bar lies within
    6h of any given daily timestamp), so the daily fit collapses to the
    calm fallback and is dropped from the headline 3x3 ARI matrix.
    """
    return {
        freq: features(resample_ohlc(df_5m, freq), freq, stem=stem, calendar_window=calendar_window)
        for freq in _INTRADAY_FREQS
    }


def _run_calendar_window_robustness(
    symbol: str,
    df_5m: pd.DataFrame,
    calendar_window: str = "6h",
    model: str = MODEL_GMM,
    n_components: int = DEFAULT_GMM_K,
    features_by_freq: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run regime analysis with a fixed calendar-time rolling window for all frequencies.

    Iterates only the intraday frequencies (5m, 15m, 1h); 1d is excluded
    by paper definition because a 6h rolling window on daily bars is
    degenerate. Pass ``features_by_freq`` (built by
    :func:`_build_calendar_window_features`) to share features across
    successive GMM/HMM calls on the same asset.
    """
    stem = canonical_stem(symbol)
    regimes_by_freq: dict[str, pd.Series] = {}
    for freq in _INTRADAY_FREQS:
        if features_by_freq is not None and freq in features_by_freq:
            feats = features_by_freq[freq]
        else:
            ohlc = resample_ohlc(df_5m, freq)
            feats = features(ohlc, freq, stem=stem, calendar_window=calendar_window)
        labels, _ = fit_regime_model(feats, model=model, n_components=n_components, freq=freq)
        regimes_by_freq[freq] = labels

    aligned = align_regimes_to_5m(regimes_by_freq, df_5m.index)
    intraday_ari_df = cross_freq_ari_matrix(aligned, _INTRADAY_FREQS)
    return {
        "calendar_window": calendar_window,
        "ari_matrix": intraday_ari_df,
        "mean_offdiag_ari": mean_offdiag_ari(intraday_ari_df),
    }
