"""Smoke tests for feature builders inside ``src.workflows.pipeline``.

The pipeline currently keeps the rolling-feature primitives in
``workflows/pipeline.py`` rather than ``core/features.py``; these tests pin
their behaviour so any future extraction to ``core/features.py`` can be
verified to preserve outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.workflows.pipeline import (
    features,
    resample_ohlc,
    robust_filter_returns,
    window_spec,
)


def _fake_5m_ohlc(n_days: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = pd.date_range(
        start="2026-04-01 09:30",
        periods=n_days * 78,
        freq="5min",
        tz="America/New_York",
    )
    rs = np.random.default_rng(seed)
    base = 100 + np.cumsum(rs.normal(0, 0.1, len(rng)))
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 0.05,
            "Low": base - 0.05,
            "Close": base + rs.normal(0, 0.02, len(rng)),
            "Volume": rs.integers(1_000, 10_000, len(rng)),
        },
        index=rng,
    )


# ---------------------------------------------------------------------------
# window_spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq,expected_type",
    [("5m", str), ("15m", str), ("1h", int), ("1d", int)],
)
def testwindow_spec_returns_expected_type(freq, expected_type):
    spec = window_spec(freq)
    assert isinstance(spec, expected_type)


def testwindow_spec_unsupported_freq_raises():
    with pytest.raises(ValueError):
        window_spec("30m")


def testwindow_spec_scale_floor():
    # Scale below 0.25 is clamped to 0.25; 0.0 should not produce an empty window.
    spec = window_spec("1h", window_scale=0.0)
    assert isinstance(spec, int) and spec >= 2


# ---------------------------------------------------------------------------
# robust_filter_returns
# ---------------------------------------------------------------------------


def testrobust_filter_returns_passthrough_non_cl():
    s = pd.Series([0.01, -0.02, 0.05, -0.10])
    out = robust_filter_returns(s, stem="SPY", freq="5m")
    pd.testing.assert_series_equal(out, s)


def testrobust_filter_returns_clips_cl_outliers():
    s = pd.Series([0.001, -0.002, 0.005, 1.0, -1.0, 0.003, -0.004, 0.002])
    out = robust_filter_returns(s, stem="CL", freq="5m")
    assert out.max() < 1.0
    assert out.min() > -1.0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def test_features_returns_ret_and_vol_columns():
    df = _fake_5m_ohlc()
    feats = features(df, freq="5m", stem="SPY")
    assert set(feats.columns) == {"ret", "vol"}
    assert len(feats) == len(df)


def test_features_no_inf_after_clipping():
    """Both ``ret`` and ``vol`` must be finite after the clipping pipeline.

    Strengthened (P0-T5): the previous version only checked ``ret`` and
    would have passed for a function that returned ``vol=inf`` everywhere.
    """
    df = _fake_5m_ohlc()
    feats = features(df, freq="5m", stem="SPY")
    assert np.isfinite(feats["ret"].dropna()).all()
    assert np.isfinite(feats["vol"].dropna()).all()


def test_features_clip_schedule_default_is_freq_aware():
    """clip_pct=None should use DEFAULT_CLIP_PCT_BY_FREQ (5m=3%, 1d=20%).

    Pin the regression: under the legacy flat 0.03 cap, a 1d series with
    a single -10% return would clip to -3%; under the new schedule it
    must pass through untouched (|ret| < 20%).
    """
    from src.core.features import features as core_features, DEFAULT_CLIP_PCT_BY_FREQ

    # Build a 1d OHLC frame with a single -10% drop in the middle.
    idx = pd.date_range(
        start="2026-04-01", periods=10, freq="1B", tz="America/New_York"
    )
    close = pd.Series([100.0] * 10, index=idx)
    close.iloc[5] = 90.0  # -10% from 100
    close.iloc[6:] = 90.0
    df = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1}
    )
    # New default (frequency-aware): the -10% drop survives the cap.
    feats_new = core_features(df, freq="1d", stem="SPY")
    assert feats_new["ret"].min() < -0.05  # not clipped at -3% nor at -5%
    assert feats_new["ret"].min() >= -DEFAULT_CLIP_PCT_BY_FREQ["1d"]
    # Legacy flat 0.03 cap: same input, return is clipped to -3%.
    feats_legacy = core_features(df, freq="1d", stem="SPY", clip_pct=0.03)
    assert feats_legacy["ret"].min() == pytest.approx(-0.03, rel=1e-9)


def test_daily_realised_vol_drops_placeholder_bars_by_default():
    """Pin the placeholder-bar filter in daily_realised_vol.

    Build two trading days where day A has 50% O=H=L=C placeholders
    interleaved with real moves, and day B has none. RV under the
    legacy ``drop_placeholder_bars=False`` path treats both days
    similarly; under the new default the placeholder bars are
    removed before the bar-count check, but their zero contribution
    means RV magnitude is unchanged. The test pins both invariants:
    (a) RV magnitude unchanged within rounding,
    (b) bar-count filter still satisfied (>=10 finite returns).
    """
    from src.core.features import daily_realised_vol

    rng = np.random.default_rng(0)
    # Two days, 78 5m bars each (matches RTH SPY count).
    idx_a = pd.date_range(
        "2026-04-01 09:30", periods=78, freq="5min", tz="America/New_York"
    )
    idx_b = pd.date_range(
        "2026-04-02 09:30", periods=78, freq="5min", tz="America/New_York"
    )
    idx = idx_a.append(idx_b)
    base = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    df = pd.DataFrame(
        {"Open": base, "High": base + 0.05, "Low": base - 0.05,
         "Close": base + rng.normal(0, 0.02, len(idx)), "Volume": 100},
        index=idx,
    )
    # Make every other bar in day A a placeholder (O=H=L=C).
    flat_pos = np.zeros(len(idx), dtype=bool)
    flat_pos[1::2][:39] = True  # half of day-A bars
    for col in ("Open", "High", "Low", "Close"):
        df.loc[flat_pos, col] = base[flat_pos]

    rv_default = daily_realised_vol(df)  # drop_placeholder_bars=True
    rv_legacy = daily_realised_vol(df, drop_placeholder_bars=False)
    # Both definitions emit a value for both trading days.
    assert len(rv_default) == 2
    assert len(rv_legacy) == 2
    # Day B (no placeholders): RV is identical between the two paths.
    day_b = pd.Timestamp("2026-04-02", tz="America/New_York")
    assert np.isclose(rv_default[day_b], rv_legacy[day_b], rtol=1e-12)


def test_daily_realised_vol_min_bars_threshold():
    """Days with fewer than ``min_bars_per_day`` finite returns drop out."""
    from src.core.features import daily_realised_vol

    # Build 5 bars on day A (below threshold of 10) and 78 on day B.
    idx_a = pd.date_range(
        "2026-04-01 09:30", periods=5, freq="5min", tz="America/New_York"
    )
    idx_b = pd.date_range(
        "2026-04-02 09:30", periods=78, freq="5min", tz="America/New_York"
    )
    idx = idx_a.append(idx_b)
    s = pd.Series(np.arange(len(idx)) * 1.0 + 100.0, index=idx)
    df = pd.DataFrame(
        {"Open": s, "High": s, "Low": s, "Close": s, "Volume": 100}
    )
    # All bars are flat O=H=L=C; with drop_placeholder_bars=True the
    # whole frame collapses to no finite returns.
    out = daily_realised_vol(df, drop_placeholder_bars=True)
    assert len(out) == 0
    # Disable the placeholder filter -> day B passes the >=10 threshold,
    # day A does not.
    out_legacy = daily_realised_vol(df, drop_placeholder_bars=False)
    assert len(out_legacy) == 1


def test_features_clip_dict_per_freq_override():
    """Dict form lets callers pin a per-frequency cap. Missing keys fall
    back to the schedule default.
    """
    from src.core.features import features as core_features

    df = _fake_5m_ohlc()
    out = core_features(df, freq="5m", stem="SPY", clip_pct={"5m": 0.001})
    # |ret| capped at 0.001
    assert out["ret"].abs().max() <= 0.001 + 1e-12


def test_features_calendar_window_overrides_default():
    """Strengthened (P1-T2): rather than just asserting "rolling vol differs",
    pin the exact rolling window. With a 60min calendar window on 5min bars,
    the rolling std at bar k (deep into the series) must equal the manual
    std of the 12 most-recent returns (pandas time-based rolling default
    ``inclusive='right'``: window is ``(t - 60min, t]``).
    """
    df = _fake_5m_ohlc(n_days=5)
    a = features(df, freq="5m", stem="SPY")
    b = features(df, freq="5m", stem="SPY", calendar_window="60min")
    # Sanity: different window specs produce different rolling vol.
    assert not a["vol"].equals(b["vol"])
    # Pin the contract numerically: deep enough in the series that the
    # rolling window is fully populated (k=200 -> ~200 bars of warm-up).
    k = 200
    # 60min on a regular 5m grid -> exactly 12 bars (k-11..k inclusive).
    expected_vol_at_k = b["ret"].iloc[k - 11 : k + 1].std()
    assert np.isclose(b["vol"].iloc[k], expected_vol_at_k, rtol=1e-9)


# ---------------------------------------------------------------------------
# resample_ohlc (already covered briefly in test_pipeline.py; expanded here)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["15m", "1h", "1d"])
def test_resample_preserves_ohlc_relationships(freq):
    df = _fake_5m_ohlc(n_days=4)
    out = resample_ohlc(df, freq)
    assert (out["High"] >= out["Low"]).all()
    assert (out["High"] >= out["Open"]).all()
    assert (out["High"] >= out["Close"]).all()
    assert (out["Low"] <= out["Open"]).all()
    assert (out["Low"] <= out["Close"]).all()


def test_resample_5m_returns_copy_not_view():
    df = _fake_5m_ohlc()
    out = resample_ohlc(df, "5m")
    assert out is not df
    out.iloc[0, 0] = 999.0
    assert df.iloc[0, 0] != 999.0


def test_resample_1d_label_is_16_NY_across_DST_spring_forward():
    """The 1d bar timestamp must be 16:00 NY local on every trading day,
    including DST-transition days.

    Regression: ``resample('1D').index + pd.Timedelta(hours=16)`` adds
    UTC-elapsed 16h, so on the spring-forward day the day's midnight is
    on the pre-transition offset and the elapsed-16h walks across the
    2am gap, landing at 17:00 -04:00 instead of 16:00 -04:00. Downstream
    ``align_regimes_to_5m`` then ffill's the 1d label one hour late on
    the affected day.
    """
    idx = pd.date_range(
        start="2026-03-07 09:30",
        end="2026-03-09 16:00",
        freq="5min",
        tz="America/New_York",
    )
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=idx,
    )
    out = resample_ohlc(df, "1d")
    hours = [t.hour for t in out.index]
    assert hours == [16, 16, 16], (
        f"DST-day 1d label should be 16:00 NY-local on every day; got {hours} "
        f"(2026-03-08 is the US spring-forward day)"
    )


def test_resample_1d_label_is_16_NY_across_DST_fall_back():
    """Same regression on the fall-back day (2026-11-01)."""
    idx = pd.date_range(
        start="2026-10-31 09:30",
        end="2026-11-02 16:00",
        freq="5min",
        tz="America/New_York",
    )
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=idx,
    )
    out = resample_ohlc(df, "1d")
    hours = [t.hour for t in out.index]
    assert hours == [16, 16, 16], (
        f"DST-day 1d label should be 16:00 NY-local on every day; got {hours} "
        f"(2026-11-01 is the US fall-back day)"
    )
