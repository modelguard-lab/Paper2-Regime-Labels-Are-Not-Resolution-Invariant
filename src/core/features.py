"""Rolling feature builders: OHLC resampling, robust return filtering, and
log-return + rolling-volatility computation.

Causality: log-returns and the rolling-volatility window are causal (no
backfill, no look-ahead beyond the current bar). The CL-only
``robust_filter_returns`` clip uses the **full-sample** median + MAD as
the clipping anchor, which is intentionally offline: it is a one-shot
preprocessing step on the in-sample training data that holds the
clipping boundaries fixed for all downstream operations. Do NOT route
``robust_filter_returns`` through OOS / live-scoring code paths without
re-anchoring on a strictly past sample first; the pipeline calls it only
on the fixed full-sample frame, where the global anchor is not a
look-ahead concern. Used by ``workflows/pipeline.py`` and by the
extended-analyses experiments that resample on the fly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_WINDOW_SCALE, TZ
from .time_utils import ensure_ny_tz


# Frequency-aware default symmetric per-bar return cap. The cap is applied
# AFTER ``robust_filter_returns`` and AFTER the resample step, so a 5m cap
# of 3% bites flash-crash-scale bars while a 1d cap of 20% bites only true
# regime breaks (e.g., 1987 Black Monday -22.6%, GFC October 2008 -10%).
#
# Why frequency-dependent: a 3% absolute cap binds 0.06% of CL 5m bars but
# 22% of CL 1d bars in the 2026 sample, so a flat per-bar cap suppresses
# coarse-frequency tail information far more aggressively than fine, which
# is itself a frequency-dependent processing artefact in a paper whose
# claim is "regime labels are not resolution-invariant". The schedule
# below scales the cap so each frequency loses comparable tail mass on
# realistic stress data.
#
# Replication-affecting: any change here regenerates every cross-frequency
# ARI number in the main pipeline. Callers that need bit-for-bit historic
# reproduction should pass ``clip_pct=0.03`` explicitly.
DEFAULT_CLIP_PCT_BY_FREQ: dict[str, float] = {
    "5m": 0.03,
    "15m": 0.05,
    "1h": 0.10,
    "1d": 0.20,
}


def daily_realised_vol(
    df_5m: pd.DataFrame,
    *,
    drop_placeholder_bars: bool = True,
    min_bars_per_day: int = 10,
) -> pd.Series:
    """Per-day realised volatility from 5m OHLC.

    ``RV_d = sqrt(sum_t r_t^2)`` where the sum runs over 5m log-returns
    on calendar day ``d`` (NY-normalised). The result is a ``Series``
    indexed by ``Timestamp`` at midnight NY of each trading day.

    Parameters
    ----------
    df_5m : pd.DataFrame
        5m OHLC frame with tz-aware NY index and a ``Close`` column.
        ``Open`` / ``High`` / ``Low`` are required when
        ``drop_placeholder_bars`` is ``True``.
    drop_placeholder_bars : bool, default ``True``
        When ``True``, 5m bars whose ``O==H==L==C`` are filtered out
        of the return series before RV is summed. These are typically
        extended-hours forward-fill placeholders (most pronounced on
        2022 GLD where 38% of bars qualify; the validate_5m_ohlc
        hour-bucket diagnostic flags assets with this fingerprint).
        Including them does NOT change RV magnitude in expectation
        (a forward-fill bar has ``log(C_ff/C_prev)=0``, contributing
        zero to the sum-of-squares). However, including them
        contaminates the RV-based per-day classification used by
        ``exp_02.select_matched_5day_window`` and
        ``exp_05.calm_day_subsample_ari``: when the placeholder
        share differs day-to-day, including placeholder bars shifts
        which days fall above / below the median RV. Default
        ``True`` for cross-asset consistency; pass ``False`` to
        reproduce the historic behaviour.
    min_bars_per_day : int, default 10
        Trading days with fewer than this many finite returns are
        dropped (post-filtering, when ``drop_placeholder_bars`` is
        ``True``).

    Returns
    -------
    pd.Series
        Per-day realised vol indexed by NY-normalised midnight
        timestamps; sorted ascending by date.
    """
    idx_ny = ensure_ny_tz(df_5m.index)
    rets = np.log(df_5m["Close"] / df_5m["Close"].shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = pd.Series(rets.values, index=idx_ny)
    if drop_placeholder_bars:
        flat_mask = (
            (df_5m["Open"].values == df_5m["High"].values)
            & (df_5m["High"].values == df_5m["Low"].values)
            & (df_5m["Low"].values == df_5m["Close"].values)
        )
        rets = rets.where(~flat_mask)
    day = idx_ny.normalize()
    rv = (rets ** 2).groupby(day).sum().pipe(np.sqrt)
    bar_counts = rets.notna().groupby(day).sum()
    return rv[bar_counts >= min_bars_per_day].sort_index()


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 5m OHLC to 15m, 1h, or 1d. O=first, H=max, L=min, C=last. All in NY."""
    if freq == "5m":
        return df.copy()
    if freq == "15m":
        agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg_dict["Volume"] = "sum"
        res = df.resample("15min", label="right", closed="right").agg(agg_dict)
        ohlc_cols = ["Open", "High", "Low", "Close"]
        res = res.dropna(subset=ohlc_cols, how="all")
        return res
    if freq == "1h":
        agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg_dict["Volume"] = "sum"
        res = df.resample("1h", label="right", closed="right").agg(agg_dict)
        ohlc_cols = ["Open", "High", "Low", "Close"]
        res = res.dropna(subset=ohlc_cols, how="all")
        return res
    if freq == "1d":
        agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg_dict["Volume"] = "sum"
        df_ny = df.copy()
        df_ny.index = ensure_ny_tz(df_ny.index)
        res = df_ny.resample("1D").agg(agg_dict)
        # Build the 16:00-NY close-of-day timestamp in *wall-clock* terms.
        # ``res.index + pd.Timedelta(hours=16)`` would add UTC-elapsed 16
        # hours, which on DST-transition days lands at 17:00 (spring-forward)
        # or 15:00 (fall-back) because the day's midnight is on the
        # pre-transition offset and the elapsed-16h walks across the jump.
        # tz-strip then re-localise gives the intended 16:00 NY local time
        # on every day, including DST boundaries.
        res.index = (
            res.index.tz_localize(None) + pd.Timedelta(hours=16)
        ).tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
        ohlc_cols = ["Open", "High", "Low", "Close"]
        res = res.dropna(subset=ohlc_cols, how="all")
        return res
    raise ValueError(f"Unsupported freq: {freq}")


def robust_filter_returns(ret: pd.Series, stem: str, freq: str) -> pd.Series:
    """Mitigate extreme continuous-futures roll jumps before volatility estimation.

    Only active for the CL (continuous WTI) feed. Other tickers pass
    through unchanged. Robust scale uses the median absolute deviation
    (MAD x 1.4826 for normal-consistent scale).  The clip multiplier is
    ``8`` at 5m and ``10`` at all coarser frequencies (15m / 1h / 1d).

    P1-S3 clarification: the wider 5m-vs-coarser pair is the only
    intentional asymmetry; previously the docstring claimed the multiplier
    grew at coarser frequencies but the implementation kept ``10`` flat
    above 5m.  Choosing not to tier further (e.g., 8/10/14/20) preserves
    paper numbers; if you need a tiered multiplier you must rerun the
    main pipeline and update the supplement.
    """
    if stem != "CL" or ret.dropna().empty:
        return ret
    med = ret.median()
    mad = (ret - med).abs().median()
    if pd.isna(mad) or mad < 1e-10:
        return ret
    scale = 1.4826 * mad
    # 8 at 5m, 10 at 15m/1h/1d.  See docstring P1-S3 note above.
    clip_k = 8.0 if freq == "5m" else 10.0
    lower = med - clip_k * scale
    upper = med + clip_k * scale
    return ret.clip(lower=lower, upper=upper)


def window_spec(freq: str, window_scale: float = DEFAULT_WINDOW_SCALE) -> str | int:
    """Return the rolling-volatility window for a given frequency.

    Returned as a pandas-compatible window argument: a `"{minutes}min"` string
    for sub-hourly frequencies, an integer bar count for 1h and 1d.
    """
    scale = max(float(window_scale), 0.25)
    if freq in {"5m", "15m"}:
        minutes = max(30, int(round(120 * scale)))
        return f"{minutes}min"
    if freq == "1h":
        return max(2, int(round(24 * scale)))
    if freq == "1d":
        return max(2, int(round(5 * scale)))
    raise ValueError(f"Unsupported freq for window spec: {freq}")


def features(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str | None = None,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    calendar_window: str | None = None,
    clip_pct: float | dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build log-return and rolling-volatility features.

    Parameters
    ----------
    df_ohlc : pd.DataFrame
        OHLC frame at the target frequency, tz-aware index.
    freq : str
        One of ``("5m", "15m", "1h", "1d")``.
    stem : str, optional
        Asset stem used to gate ticker-specific filtering (currently CL only).
    window_scale : float, default 1.0
        Multiplier on the default rolling window.
    calendar_window : str, optional
        Override the frequency-specific rolling window with a fixed
        calendar-time window (e.g., ``"6h"``), enabling cross-frequency
        comparability in physical time.
    clip_pct : float | dict[str, float] | None, default ``None``
        Symmetric per-bar log-return cap applied AFTER
        :func:`robust_filter_returns`. Three accepted forms:

        * ``None``: use :data:`DEFAULT_CLIP_PCT_BY_FREQ` keyed by ``freq``,
          i.e. a frequency-aware schedule (5m=0.03, 15m=0.05, 1h=0.10,
          1d=0.20).
        * ``float``: legacy flat cap applied identically at every
          frequency. Pass ``0.03`` to reproduce the historic behaviour
          (a flat 3% cap that bound 22% of 1d CL bars in 2026).
        * ``dict``: explicit per-frequency override; missing keys fall
          back to :data:`DEFAULT_CLIP_PCT_BY_FREQ`.

    Returns
    -------
    pd.DataFrame
        Two columns: ``ret`` (clipped log return) and ``vol`` (rolling std
        of return). ``vol`` is forward-filled to remove warm-up gaps;
        never backfilled.
    """
    if clip_pct is None:
        clip_value = DEFAULT_CLIP_PCT_BY_FREQ.get(freq, 0.03)
    elif isinstance(clip_pct, dict):
        clip_value = clip_pct.get(freq, DEFAULT_CLIP_PCT_BY_FREQ.get(freq, 0.03))
    else:
        clip_value = float(clip_pct)

    out = pd.DataFrame(index=df_ohlc.index)
    close = df_ohlc["Close"]
    close_safe = close.where(close > 0)
    out["ret"] = np.log(close_safe / close_safe.shift(1))
    out["ret"] = out["ret"].replace([np.inf, -np.inf], np.nan)
    out["ret"] = robust_filter_returns(out["ret"], stem or "", freq)
    # P1-S4: hard symmetric cap applied AFTER robust_filter_returns. For
    # non-CL stems this is the only filter. The cap is now keyed by freq
    # via DEFAULT_CLIP_PCT_BY_FREQ when clip_pct is None; explicit float
    # / dict callers override.
    out["ret"] = out["ret"].clip(lower=-float(clip_value), upper=float(clip_value))
    win = calendar_window if calendar_window else window_spec(freq, window_scale=window_scale)
    out["vol"] = out["ret"].rolling(window=win, min_periods=2).std()
    out = out.replace([np.inf, -np.inf], np.nan)
    # P3 / features.py:122 - explicit ``limit=None`` documents that ffill
    # is unbounded.  Multi-week gaps (holidays, suspended tickers) are
    # carried forward indefinitely; downstream code must reindex to a
    # canonical 5m grid before treating ``vol`` as a strict warm-up
    # signal.
    out["vol"] = out["vol"].ffill(limit=None)
    return out
