"""Backfill tests for ``src.experiments.exp_02_bootstrap`` (P2-C).

Targets the two pure functions reused outside the experiment's ``main``:
- ``select_matched_5day_window``: pick a contiguous 5-trading-day window
  matched on realised volatility, in ``"stress"`` (max RV) or ``"calm"``
  (median-RV) mode.
- ``block_bootstrap_mean_ari``: block-resample the cross-frequency mean
  off-diagonal ARI from a 5m-aligned label dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.exp_02_bootstrap import (
    block_bootstrap_mean_ari,
    select_matched_5day_window,
)


def _synthetic_5m_ohlc_with_stress_block(seed: int = 0) -> pd.DataFrame:
    """Build a 30-day 5m OHLC frame with a deliberate high-vol week.

    Days 10-14 (Mon-Fri of week 3) use a 10x larger return std so they form
    the unique stress block; ``select_matched_5day_window(mode="stress")``
    must converge to those exact days.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 78
    days = pd.bdate_range("2026-01-02", periods=30, tz="America/New_York")
    parts = []
    for d in days:
        parts.append(
            pd.date_range(
                d + pd.Timedelta(hours=9, minutes=30),
                periods=bars_per_day, freq="5min", tz="America/New_York",
            )
        )
    idx = pd.DatetimeIndex(np.concatenate([p.values for p in parts])).tz_localize(
        "UTC"
    ).tz_convert("America/New_York")
    n = len(idx)
    # Per-bar return std: 1e-4 baseline, 1e-3 for the stress block.
    day_of_idx = pd.Series(idx.normalize(), index=idx)
    stress_days = set(days[10:15])  # Mon..Fri of week 3
    is_stress = day_of_idx.isin(stress_days).values
    base_std = np.where(is_stress, 1e-3, 1e-4)
    rets = rng.normal(0.0, base_std, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.05,
            "Low": close - 0.05,
            "Close": close,
            "Volume": np.full(n, 1000, dtype=int),
        },
        index=idx,
    ), days, stress_days


def test_select_matched_5day_window_stress_picks_high_rv_block() -> None:
    """In stress mode, the helper must pick the 5-day span with the largest
    realised volatility.  We construct a fixture with one obvious stress
    week and verify the returned 5m index covers exactly that week."""
    df, days, stress_days = _synthetic_5m_ohlc_with_stress_block(seed=0)
    sel = select_matched_5day_window(
        df, search_window=("2026-01-02", "2026-02-13"),
        mode="stress", days=5,
    )
    assert len(sel) == 5 * 78
    # All bars fall on the constructed stress days.
    sel_days = set(pd.DatetimeIndex(sel).tz_convert("America/New_York").normalize())
    assert sel_days == set(stress_days), (
        f"expected stress days {stress_days}, got {sel_days}"
    )


def test_select_matched_5day_window_calm_picks_median_rv_block() -> None:
    """Calm mode targets the span whose mean RV is closest to the median;
    the result must NOT coincide with the obvious stress block."""
    df, days, stress_days = _synthetic_5m_ohlc_with_stress_block(seed=1)
    sel = select_matched_5day_window(
        df, search_window=("2026-01-02", "2026-02-13"),
        mode="calm", days=5,
    )
    assert len(sel) > 0
    sel_days = set(pd.DatetimeIndex(sel).tz_convert("America/New_York").normalize())
    # Calm mode must not pick the stress week (its RV is far above median).
    assert sel_days != set(stress_days)


def test_select_matched_5day_window_invalid_mode_raises() -> None:
    df, _, _ = _synthetic_5m_ohlc_with_stress_block()
    with pytest.raises(ValueError, match="mode must be"):
        select_matched_5day_window(
            df, search_window=("2026-01-02", "2026-02-13"),
            mode="random", days=5,  # type: ignore[arg-type]
        )


def test_select_matched_5day_window_too_short_returns_full_window() -> None:
    """If the search window has fewer than ``days`` trading days, the helper
    must fall back to returning the full search-window index (so callers
    can still compute a usable ARI; calling code logs a warning)."""
    df, _, _ = _synthetic_5m_ohlc_with_stress_block()
    sel = select_matched_5day_window(
        df, search_window=("2026-01-02", "2026-01-03"),
        mode="stress", days=5,
    )
    # 1-2 trading days available, days=5 -> fallback to the search-window
    # index (which spans Jan 2 only since Jan 3 is Saturday).
    assert len(sel) > 0
    assert len(sel) <= 5 * 78


def test_block_bootstrap_mean_ari_returns_array_of_finite_floats() -> None:
    """Bootstrap output is a 1-D float array of length <= n_boot, every
    entry finite (NaNs are filtered before return)."""
    rng = np.random.default_rng(0)
    n = 78 * 20
    idx = pd.date_range(
        "2026-01-02 09:30", periods=n, freq="5min", tz="America/New_York"
    )
    aligned = {
        f: pd.Series(rng.integers(0, 2, size=n), index=idx)
        for f in ("5m", "15m", "1h", "1d")
    }
    boot = block_bootstrap_mean_ari(
        aligned, freqs=("5m", "15m", "1h", "1d"),
        n_boot=20, block_size=50, seed=123,
    )
    assert isinstance(boot, np.ndarray)
    assert boot.ndim == 1
    assert len(boot) <= 20
    if len(boot) > 0:
        assert np.isfinite(boot).all()
        # ARI is bounded in [-1, 1].
        assert (boot >= -1.0).all() and (boot <= 1.0).all()


def test_block_bootstrap_mean_ari_too_short_returns_empty() -> None:
    """If the aligned series has fewer than ``2 * block_size`` bars, the
    bootstrap returns an empty array (the canonical "skip" sentinel)."""
    idx = pd.date_range("2026-01-02 09:30", periods=50, freq="5min", tz="America/New_York")
    aligned = {f: pd.Series(np.zeros(50), index=idx) for f in ("5m", "15m")}
    boot = block_bootstrap_mean_ari(
        aligned, freqs=("5m", "15m"), n_boot=10, block_size=50, seed=0
    )
    assert boot.size == 0


def test_block_bootstrap_mean_ari_is_seed_deterministic() -> None:
    """Same seed -> same bootstrap statistics array."""
    rng = np.random.default_rng(0)
    n = 78 * 10
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="5min", tz="America/New_York")
    aligned = {
        f: pd.Series(rng.integers(0, 2, size=n), index=idx)
        for f in ("5m", "15m", "1h", "1d")
    }
    a = block_bootstrap_mean_ari(aligned, freqs=("5m", "15m", "1h", "1d"),
                                 n_boot=10, block_size=50, seed=42)
    b = block_bootstrap_mean_ari(aligned, freqs=("5m", "15m", "1h", "1d"),
                                 n_boot=10, block_size=50, seed=42)
    np.testing.assert_array_equal(a, b)
