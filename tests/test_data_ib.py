"""Smoke tests for src.data.data_ib pure helpers (no IB connection needed)."""

import numpy as np
import pandas as pd

from src.data.data_ib import canonical_stem, validate_5m_ohlc


def test_canonical_stem_strips_yahoo_suffixes():
    # Old behaviour silently mapped CL=F -> CLF -> CL and USDJPY=X -> USDJPYX
    # -> USDJPY via two hard-coded special cases. CLF collides with the
    # Cleveland-Cliffs equity ticker, so the special cases were removed;
    # canonical_stem now strips only "=" / "/" / "^" / "." punctuation.
    # Yahoo-suffix awareness lives upstream in build_contract.
    assert canonical_stem("CL=F") == "CLF"
    assert canonical_stem("USDJPY=X") == "USDJPYX"


def test_canonical_stem_idempotent_on_clean_tickers():
    for sym in ("SPY", "CL", "USDJPY"):
        assert canonical_stem(sym) == sym


def test_canonical_stem_uppercases_and_strips_caret():
    assert canonical_stem("^GSPC") == "GSPC"
    assert canonical_stem("spy") == "SPY"


def test_canonical_stem_handles_dots_and_slashes():
    assert canonical_stem("BRK.B") == "BRK_B"
    assert canonical_stem("USD/JPY") == "USD_JPY"


def _make_ohlc_with_flat_hours(flat_hours: set[int], days: int = 5) -> pd.DataFrame:
    """5m OHLC frame where bars whose hour-of-day is in ``flat_hours`` are
    forward-fill placeholders (O=H=L=C), and other hours move."""
    rng = np.random.default_rng(0)
    bars_per_day = 192  # 16h * 12 (matches extended-hours density)
    idx = pd.date_range(
        "2026-04-01 00:00", periods=days * bars_per_day,
        freq="5min", tz="America/New_York",
    )
    base = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    df = pd.DataFrame(
        {"Open": base, "High": base + 0.05, "Low": base - 0.05,
         "Close": base + rng.normal(0, 0.02, len(idx)), "Volume": 100},
        index=idx,
    )
    flat_mask = np.isin(idx.hour, list(flat_hours))
    flat_vals = base[flat_mask]
    for col in ("Open", "High", "Low", "Close"):
        df.loc[flat_mask, col] = flat_vals
    return df


def test_validate_passes_when_only_one_hour_is_flat_below_threshold():
    """A single hour with 100% flat is *worth* flagging -- it produces
    pct_fallback at fine frequencies. The check kicks in at >70% with
    >=78 bars of evidence (one trading day). This test pins the
    threshold."""
    df = _make_ohlc_with_flat_hours({4}, days=5)  # ~60 bars in hour 04
    issues = validate_5m_ohlc(df, "TEST")
    # 5 days * 12 bars = 60 bars at hour 04, which is below the 78-bar
    # evidence floor -- check should not yet fire.
    flat_msgs = [m for m in issues if "flat O=H=L=C" in m]
    assert flat_msgs == []


def test_validate_flags_extended_hours_forward_fill():
    """The 2022 GLD pattern: pre-market and post-market 5m bars are
    forward-fill placeholders (>70% flat) while RTH bars move. Panel
    flat share stays well under the 50% gate but the hour-bucket
    diagnostic must still fire so reviewers know the fine-freq labels
    will be a percentile threshold, not GMM.
    """
    flat_hours = {4, 5, 6, 17, 18, 19}  # mimic GLD22 fingerprint
    # 14 days * 12 bars/hour = 168 bars per hour, comfortably above the
    # 78-bar evidence floor for every flagged hour.
    df = _make_ohlc_with_flat_hours(flat_hours, days=14)
    issues = validate_5m_ohlc(df, "TEST")
    flat_msgs = [m for m in issues if "flat O=H=L=C" in m]
    assert len(flat_msgs) == 1
    msg = flat_msgs[0]
    # Each named hour should appear in the offender list.
    for h in flat_hours:
        assert f"{h:02d}:00" in msg
    # The panel-wide >50% flat warning should NOT fire (only ~37% flat overall).
    panel_msgs = [m for m in issues if "% of bars have O=H=L=C" in m]
    assert panel_msgs == []


def test_validate_clean_data_emits_no_flat_warnings():
    """No flat bars at all -> no flat-related issues."""
    df = _make_ohlc_with_flat_hours(set(), days=5)
    issues = validate_5m_ohlc(df, "TEST")
    assert not any("flat" in m or "O=H=L=C" in m for m in issues)
