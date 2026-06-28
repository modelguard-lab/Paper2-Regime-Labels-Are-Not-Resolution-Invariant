"""Tests for ``src.core.diagnostics``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.diagnostics import tod_adjusted_volatility, tod_crisis_distribution


def _aligned_series(n: int = 200, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="5min", tz="America/New_York")
    return {
        "5m": pd.Series(rng.integers(0, 2, size=n), index=idx),
        "15m": pd.Series(rng.integers(0, 2, size=n), index=idx),
    }


def test_tod_crisis_distribution_returns_long_format_frame():
    aligned = _aligned_series()
    out = tod_crisis_distribution(aligned, ("5m", "15m"))
    assert set(out.columns) == {"freq", "hour", "crisis_share", "n_bars"}
    # Both freqs present
    assert set(out["freq"].unique()) >= {"5m", "15m"}


def test_tod_crisis_distribution_skips_missing_freq():
    aligned = {
        "5m": _aligned_series()["5m"],
        "15m": pd.Series(dtype=int),  # empty
    }
    out = tod_crisis_distribution(aligned, ("5m", "15m"))
    # 15m should be absent because the series is empty
    assert "15m" not in out["freq"].unique()


def test_tod_crisis_distribution_crisis_share_is_percentage():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2026-01-01 09:30", periods=n, freq="5min", tz="America/New_York")
    s = pd.Series(np.ones(n, dtype=int), index=idx)  # all crisis
    out = tod_crisis_distribution({"5m": s}, ("5m",))
    assert (out["crisis_share"] == 100.0).all()


def test_tod_adjusted_volatility_passthrough_for_daily_freq():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=20, freq="D", tz="America/New_York")
    base = 100 + np.cumsum(rng.normal(0, 0.1, 20))
    df = pd.DataFrame(
        {"Open": base, "High": base + 0.05, "Low": base - 0.05, "Close": base + 0.01},
        index=idx,
    )
    feats_adj = tod_adjusted_volatility(df, freq="1d", stem="SPY")
    # 1d should pass through untouched (no TOD adjustment meaningful at daily)
    assert "vol" in feats_adj.columns


def test_tod_adjusted_volatility_normalises_intraday_vol():
    """Strengthened (P1-T1): the previous version asserted at most 70% of
    per-hour bins fell in (0.5, 1.5) over a 7-hour index, which would pass
    with 5/7 hours satisfied.  The structural property is tighter: after
    warm-up, EVERY per-hour median should sit close to 1.0.  Use (0.7, 1.3)
    and require ALL hours after warm-up to satisfy it.
    """
    rng = np.random.default_rng(0)
    n_days = 20  # longer history so the TOD denominator has many samples per hour
    n = 78 * n_days
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="5min", tz="America/New_York")
    base = 100 + np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame(
        {"Open": base, "High": base + 0.05, "Low": base - 0.05, "Close": base + 0.01},
        index=idx,
    )
    feats_adj = tod_adjusted_volatility(df, freq="5m", stem="SPY")
    # Drop the warm-up bars (first day) so the rolling-vol denominator is
    # populated for every hour-of-day group.
    warm_up = 78
    feats_post = feats_adj.iloc[warm_up:]
    hours = feats_post.index.tz_convert("America/New_York").hour
    medians = feats_post["vol"].groupby(hours).median().dropna()
    # All hour-of-day medians must lie in (0.7, 1.3) after warm-up.  This is
    # a tight bound: the TOD adjustment divides by the per-hour sample
    # median, so the post-adjustment per-hour median is by construction
    # near 1.0 modulo within-day Monte Carlo noise.
    assert ((medians > 0.7) & (medians < 1.3)).all(), (
        f"per-hour medians out of (0.7,1.3): {medians.to_dict()}"
    )
