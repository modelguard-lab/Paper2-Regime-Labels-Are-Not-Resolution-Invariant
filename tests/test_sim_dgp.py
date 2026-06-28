"""Backfill tests for ``src.core.sim_dgp`` (P2-C: missing coverage).

Targets the small-fixture, deterministic-seed contracts of:
- ``simulate_ms_returns_5m``: shape, dtype, stationary state distribution.
- ``simulate_ms_garch_returns_5m``: shape and finite-output stability.
- ``make_rth_5m_index``: 78 bars per RTH day, NY tz, no weekends.
- ``synthetic_ohlc_5m``: produces the OHLC schema downstream pipelines need.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.core.calibration import (
    CalibratedGarchParams,
    CalibratedMSParams,
    CalibrationAt5m,
)
from src.core.sim_dgp import (
    make_rth_5m_index,
    simulate_ms_returns_5m,
    simulate_ms_garch_returns_5m,
    synthetic_ohlc_5m,
)


def _make_calibration(P_12: float = 0.01, P_21: float = 0.01) -> CalibrationAt5m:
    raw = CalibratedMSParams(
        mu_0=0.0, mu_1=0.0,
        sigma2_0=(1e-3) ** 2, sigma2_1=(5e-3) ** 2,
        P_12=P_12, P_21=P_21,
        data_source="test", sample_start="2024-01-01", sample_end="2025-01-01",
        n_bars=1000, fit_freq="5m",
        fit_date_utc=datetime.now(timezone.utc).isoformat(),
        log_likelihood=0.0,
    )
    return CalibrationAt5m(
        mu_0=0.0, mu_1=0.0,
        sigma_0=1e-3, sigma_1=5e-3,
        P_12=P_12, P_21=P_21,
        raw=raw,
    )


def test_simulate_ms_returns_5m_shape_and_state_distribution() -> None:
    """At ``P_12 = P_21`` the chain is symmetric so the stationary share of
    state 1 is 0.5; on n=10000 bars the empirical share must lie within
    ±5% of 0.5 even for a pessimistic seed (binomial 95% CI ~ ±0.01)."""
    rng = np.random.default_rng(42)
    cal = _make_calibration(P_12=0.05, P_21=0.05)
    rets, states = simulate_ms_returns_5m(cal, n_bars=10_000, rng=rng)

    assert rets.shape == (10_000,)
    assert states.shape == (10_000,)
    assert states.dtype == np.int_  or states.dtype.kind in {"i", "u"}
    # Symmetric chain: stationary share = 0.5; ±0.05 is generous.
    share_state_1 = states.mean()
    assert 0.45 < share_state_1 < 0.55, share_state_1
    # Returns are finite floats.
    assert np.isfinite(rets).all()


def test_simulate_ms_returns_5m_asymmetric_stationary() -> None:
    """For an asymmetric chain the stationary share of state 0 is
    ``P_21 / (P_12 + P_21)``.  Use n=20000 for a tight empirical estimate."""
    rng = np.random.default_rng(0)
    P_12, P_21 = 0.02, 0.04
    cal = _make_calibration(P_12=P_12, P_21=P_21)
    _, states = simulate_ms_returns_5m(cal, n_bars=20_000, rng=rng)
    expected_share_0 = P_21 / (P_12 + P_21)  # = 0.04 / 0.06 = 0.667
    actual_share_0 = (states == 0).mean()
    assert abs(actual_share_0 - expected_share_0) < 0.03


def test_simulate_ms_garch_returns_5m_shape_and_finite() -> None:
    """MS-GARCH simulator must return finite returns under stationary
    ``alpha + beta < 0.999``.  Output shape and state schema match the
    pure-MS simulator."""
    rng = np.random.default_rng(7)
    cal = _make_calibration(P_12=0.02, P_21=0.03)
    garch = CalibratedGarchParams(
        alpha=0.10, beta=0.85,
        omega_uncond_pct=1e-3,
        data_source="test", sample_start="2024", sample_end="2025",
        n_bars=1000, fit_freq="5m",
        fit_date_utc=datetime.now(timezone.utc).isoformat(),
        log_likelihood=0.0,
    )
    rets, states = simulate_ms_garch_returns_5m(cal, garch, n_bars=2_000, rng=rng)
    assert rets.shape == (2_000,)
    assert states.shape == (2_000,)
    assert np.isfinite(rets).all()
    # The simulator clips at ±10 sigma_max; verify nothing exploded past that.
    sigma_max = max(cal.sigma_0, cal.sigma_1)
    assert np.abs(rets).max() <= 10.0 * sigma_max + 1e-12


def test_simulate_ms_garch_rejects_non_stationary() -> None:
    """``alpha + beta >= 0.999`` must raise (numerical-stationarity guard)."""
    cal = _make_calibration()
    bad = CalibratedGarchParams(
        alpha=0.5, beta=0.5,  # exactly 1.0 -> non-stationary
        omega_uncond_pct=1e-3,
        data_source="", sample_start="", sample_end="",
        n_bars=100, fit_freq="5m", fit_date_utc="now",
        log_likelihood=0.0,
    )
    with pytest.raises(ValueError, match="stationar"):
        simulate_ms_garch_returns_5m(cal, bad, n_bars=100, rng=np.random.default_rng(0))


def test_make_rth_5m_index_session_structure() -> None:
    """Each RTH session has 78 5m bars (09:35 to 16:00 ET).  Build a 5-day
    index and verify (i) length, (ii) NY tz, (iii) all weekdays, (iv) no
    duplicate timestamps."""
    idx = make_rth_5m_index(n_days=5, start="2024-01-02")
    assert len(idx) == 5 * 78
    assert idx.tz is not None
    assert "New_York" in str(idx.tz)
    # Mondays = 0 ... Fridays = 4; never weekend.
    weekdays = idx.weekday
    assert ((weekdays >= 0) & (weekdays <= 4)).all()
    assert idx.is_unique


def test_synthetic_ohlc_5m_schema() -> None:
    """``synthetic_ohlc_5m`` must produce an OHLC frame with strictly positive
    prices and OHLC inequalities (degenerate O=H=L=C is fine)."""
    rng = np.random.default_rng(0)
    n = 200
    rets = rng.normal(0.0, 1e-3, n)
    idx = pd.date_range("2024-01-02 09:35", periods=n, freq="5min", tz="America/New_York")
    df = synthetic_ohlc_5m(rets, idx, start_price=100.0)
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
    assert (df["Close"] > 0).all()
    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df["Open"]).all()
    assert (df["Low"] <= df["Close"]).all()
    assert len(df) == n
