"""Tests for ``src.core.aggregation``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.aggregation import cl_roll_week_analysis, compute_daily_outputs


def _aligned_5m(n_days: int = 10, seed: int = 0) -> dict[str, pd.Series]:
    """Build a 5m-aligned dict spanning ``n_days`` distinct calendar dates.

    Each calendar date contributes 78 5-minute bars (RTH-style 6.5-hour
    session) so the timestamps line up with the day-by-day grouping that
    ``compute_daily_outputs`` performs.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 78
    days = pd.bdate_range("2026-01-02", periods=n_days, tz="America/New_York")
    parts = [
        pd.date_range(
            d + pd.Timedelta(hours=9, minutes=30),
            periods=bars_per_day,
            freq="5min",
            tz="America/New_York",
        )
        for d in days
    ]
    idx = pd.DatetimeIndex(np.concatenate([p.values for p in parts])).tz_localize("UTC").tz_convert("America/New_York")
    n = len(idx)
    out = {}
    for freq in ("5m", "15m", "1h", "1d"):
        out[freq] = pd.Series(rng.integers(0, 2, size=n), index=idx)
    return out


def test_compute_daily_outputs_returns_four_frames():
    aligned = _aligned_5m()
    daily, daily_pairs, rolling, rolling_pairs = compute_daily_outputs(
        aligned, freqs=("5m", "15m", "1h", "1d"), rolling_days=3
    )
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(daily_pairs, pd.DataFrame)
    assert isinstance(rolling, pd.DataFrame)
    assert isinstance(rolling_pairs, pd.DataFrame)


def test_compute_daily_outputs_daily_frame_has_one_row_per_day():
    aligned = _aligned_5m(n_days=5)
    daily, _, _, _ = compute_daily_outputs(
        aligned, freqs=("5m", "15m", "1h", "1d"), rolling_days=3
    )
    assert len(daily) == 5
    assert "date" in daily.columns
    assert "mean_offdiag_ari" in daily.columns


def test_compute_daily_outputs_rolling_window_count():
    n_days = 10
    rolling_days = 3
    aligned = _aligned_5m(n_days=n_days)
    _, _, rolling, _ = compute_daily_outputs(
        aligned, freqs=("5m", "15m", "1h", "1d"), rolling_days=rolling_days
    )
    # n_days - rolling_days + 1 windows expected
    assert len(rolling) == n_days - rolling_days + 1


def test_compute_daily_outputs_pair_frames_have_six_pairs_per_period():
    n_days = 4
    aligned = _aligned_5m(n_days=n_days)
    _, daily_pairs, _, _ = compute_daily_outputs(
        aligned, freqs=("5m", "15m", "1h", "1d"), rolling_days=3
    )
    # 4 freqs choose 2 = 6 unique pairs per day, 4 days => 24 rows
    assert len(daily_pairs) == n_days * 6


def test_compute_daily_outputs_empty_input_returns_empty_frames():
    """Strengthened (P0-T6): assert ALL four returned frames are empty, not
    just ``daily``. A bug returning non-empty pair frames previously passed."""
    empty: dict[str, pd.Series] = {}
    daily, daily_pairs, rolling, rolling_pairs = compute_daily_outputs(
        empty, freqs=("5m", "15m", "1h", "1d"), rolling_days=7
    )
    assert daily.empty
    assert daily_pairs.empty
    assert rolling.empty
    assert rolling_pairs.empty


def test_cl_roll_week_analysis_default_third_week_mask():
    aligned = _aligned_5m(n_days=30)
    base = aligned["5m"]
    out = cl_roll_week_analysis(aligned, base.index, freqs=("5m", "15m", "1h", "1d"))
    assert set(out.keys()) >= {
        "roll_week_mean_ari",
        "nonroll_week_mean_ari",
        "roll_week_bars",
        "nonroll_week_bars",
    }
    assert out["roll_week_bars"] + out["nonroll_week_bars"] == len(base.index)


def test_cl_roll_week_analysis_explicit_roll_dates():
    """Strengthened (P1-T3): pin the exact roll-week coverage instead of just
    ``> 0``.  ±2 business days around a single roll date covers exactly 5
    trading days, so ``roll_week_bars ≈ 5 * bars_per_day`` (exactly equal
    on synthetic data with no holidays in the window).
    """
    bars_per_day = 78
    aligned = _aligned_5m(n_days=30)
    base = aligned["5m"]
    roll_dates = ["2026-01-15"]
    out = cl_roll_week_analysis(
        aligned, base.index, freqs=("5m", "15m", "1h", "1d"), roll_dates=roll_dates
    )
    # ±2 BDay around a single roll date = 5 trading days.  Synthetic fixture
    # has no US holidays in the window, so the count is exact, not approx.
    assert out["roll_week_bars"] == 5 * bars_per_day
    assert out["nonroll_week_bars"] == (len(base.index) - 5 * bars_per_day)
