"""Time-of-day diagnostics on regime-label series and volatility features.

Two functions used by the multi-frequency pipeline and the extended-analyses
experiments:

- ``tod_crisis_distribution`` profiles the share of crisis labels by hour
  of the trading day, separately for each frequency. Useful for spotting
  session-boundary artefacts (e.g., RTH-only SPY vs full-session CL).
- ``tod_adjusted_volatility`` rebuilds features with each bar's volatility
  divided by the same hour's median, removing the deterministic intraday
  vol pattern before regime classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_WINDOW_SCALE
from .features import features, resample_ohlc
from .time_utils import ensure_ny_tz


def tod_crisis_distribution(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
) -> pd.DataFrame:
    """Crisis-label share by hour of day, per frequency.

    Returns a long-format frame with columns ``freq``, ``hour``,
    ``crisis_share`` (percent), and ``n_bars``. The hour comes from the
    timestamp converted to NY local time. For ``"1d"`` frequency the hour
    is fixed at 16 (NY close).
    """
    rows: list[dict[str, Any]] = []
    for freq in freqs:
        s = aligned.get(freq)
        if s is None or s.empty:
            continue
        s = s.dropna()
        if s.empty:
            continue
        assert set(np.unique(s)).issubset({0, 1}), "diagnostics expects binary labels"
        # P2 / diagnostics.py:46 -- previously a tz-naive index bypassed
        # tz_convert and ``.hour`` was read in whatever local zone the
        # input arrived in.  Coerce to NY explicitly so the bin labels are
        # always wall-clock NY hours.
        idx_ny = ensure_ny_tz(s.index)
        hours = idx_ny.hour
        for h in sorted(hours.unique()):
            mask = hours == h
            vals = s.values[mask]
            rows.append({
                "freq": freq,
                "hour": int(h),
                "crisis_share": float(100.0 * (vals == 1).mean()),
                "n_bars": int(len(vals)),
            })
    return pd.DataFrame(rows)


def tod_adjusted_volatility(
    df_5m: pd.DataFrame,
    freq: str,
    stem: str,
    window_scale: float = DEFAULT_WINDOW_SCALE,
) -> pd.DataFrame:
    """Build features whose volatility is divided by the same-hour median.

    !! NON-CAUSAL DIAGNOSTIC -- DO NOT USE FOR SR 26-2 PRODUCTION SCORING !!

    The per-hour median is computed over the FULL sample, so each bar's
    adjusted volatility leaks information from every other bar in the same
    hour bucket -- including future bars.  This function exists only as a
    descriptive diagnostic to compare against the standard pipeline; it
    must NEVER feed an out-of-sample regime classifier.  P1-S5: the prior
    docstring left this implicit.  Do not silently swap this in for
    ``features()`` in expanding-window or live-scoring contexts.

    Removes the deterministic intraday vol pattern before regime fitting
    (in-sample only).  Daily frequency is returned unchanged (TOD
    adjustment not meaningful for one bar per day).

    Implementation notes: the divisor falls back to the overall median
    when the per-hour median is zero, and the result is forward-filled.
    The forward-fill itself is causal but does not redeem the prior
    full-sample median computation.
    """
    ohlc = resample_ohlc(df_5m, freq)
    feats = features(ohlc, freq, stem=stem, window_scale=window_scale)
    if freq == "1d":
        return feats
    # P2 / diagnostics.py:46 analogue -- coerce to tz-aware NY before
    # reading ``.hour`` so a tz-naive caller doesn't bin into the wrong
    # local zone.
    idx_ny = ensure_ny_tz(feats.index)
    hours = idx_ny.hour
    if feats["vol"].dropna().empty:
        raise ValueError("tod_adjusted_volatility: feats['vol'] is entirely NaN")
    median_vol_by_hour = feats.groupby(hours)["vol"].transform("median")
    median_vol_by_hour = median_vol_by_hour.replace(0, np.nan).fillna(feats["vol"].median())
    feats_adj = feats.copy()
    feats_adj["vol"] = feats["vol"] / median_vol_by_hour
    # fillna(1.0): post-adjustment baseline; avoids bfill look-ahead
    feats_adj["vol"] = feats_adj["vol"].replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    return feats_adj
