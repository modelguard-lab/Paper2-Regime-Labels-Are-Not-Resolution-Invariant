"""Smoke tests for src.workflows.pipeline pure helpers."""

import numpy as np
import pandas as pd
import pytest

from src.workflows.pipeline import resample_ohlc, window_spec


def _fake_5m_ohlc(n_days: int = 3) -> pd.DataFrame:
    """Build a tiny synthetic 5m OHLC frame in America/New_York timezone."""
    rng = pd.date_range(
        start="2026-04-01 09:30",
        periods=n_days * 78,
        freq="5min",
        tz="America/New_York",
    )
    rs = np.random.default_rng(0)
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


def test_resample_5m_passthrough_returns_copy():
    df = _fake_5m_ohlc()
    out = resample_ohlc(df, "5m")
    assert out is not df
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns)


@pytest.mark.parametrize("freq", ["15m", "1h", "1d"])
def test_resample_aggregates_to_coarser_grid(freq):
    df = _fake_5m_ohlc(n_days=3)
    out = resample_ohlc(df, freq)
    assert len(out) > 0
    assert len(out) <= len(df)
    assert {"Open", "High", "Low", "Close"}.issubset(out.columns)
    assert (out["High"] >= out["Low"]).all()


def test_resample_unsupported_freq_raises():
    df = _fake_5m_ohlc()
    with pytest.raises(ValueError):
        resample_ohlc(df, "30m")


def test_window_spec_returns_string_for_intraday():
    for freq in ("5m", "15m"):
        spec = window_spec(freq)
        assert isinstance(spec, str)
        assert "min" in spec


def test_window_spec_returns_int_for_hourly_daily():
    for freq in ("1h", "1d"):
        spec = window_spec(freq)
        assert isinstance(spec, int)
        assert spec >= 2
