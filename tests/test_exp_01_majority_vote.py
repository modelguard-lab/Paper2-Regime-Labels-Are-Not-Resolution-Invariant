"""Pin the bin-alignment fix in exp_01_majority_vote.

The previous implementation used ``floor`` to bin fine 5m timestamps into
coarse 15m / 1h windows, but ``resample_ohlc`` builds those windows with
``closed='right', label='right'``: the bar at timestamp T aggregates fine
timestamps in ``(T - bar, T]``. ``floor`` therefore put the wrong fine
bars into each coarse bin (a 5m bar at 10:05 was assigned to bin 10:00,
which is built from 5m bars 09:50-10:00). The fix replaces ``floor``
with ``ceil``; this module pins the corrected mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.features import resample_ohlc
from src.experiments.exp_01_majority_vote import _coarse_bin_keys


def _synthetic_5m(periods: int = 19) -> pd.DataFrame:
    idx = pd.date_range(
        "2026-02-02 09:30", periods=periods, freq="5min", tz="America/New_York"
    )
    s = pd.Series(np.arange(len(idx)) * 1.0 + 100.0, index=idx)
    return pd.DataFrame({
        "Open": s, "High": s, "Low": s, "Close": s, "Volume": 1
    })


def test_ceil_bins_match_resample_15m():
    """Each fine 5m timestamp must map to the 15m coarse bar that actually
    aggregates it. With ``ceil``, fine 09:50/09:55/10:00 all bin to 10:00,
    and 10:05/10:10/10:15 all bin to 10:15 - matching the OHLC produced by
    ``resample_ohlc`` exactly.
    """
    df5 = _synthetic_5m(periods=19)
    res15 = resample_ohlc(df5, "15m")
    bins = _coarse_bin_keys(df5.index, "15m")

    # Build the inverse: which fine timestamps got assigned to bin 10:00?
    bin_to_fine = pd.Series(df5.index, index=bins).groupby(level=0).agg(list)

    # Coarse 15m bar at 10:00 has Close = value at 5m 10:00 (=106 in the
    # synthetic series); the bin assigned to 10:00 must contain exactly
    # the three 5m timestamps 09:50, 09:55, 10:00.
    expected_at_10 = [
        pd.Timestamp("2026-02-02 09:50", tz="America/New_York"),
        pd.Timestamp("2026-02-02 09:55", tz="America/New_York"),
        pd.Timestamp("2026-02-02 10:00", tz="America/New_York"),
    ]
    bin_10 = pd.Timestamp("2026-02-02 10:00", tz="America/New_York")
    assert sorted(bin_to_fine.loc[bin_10]) == sorted(expected_at_10)

    # Spot-check Close at coarse 10:00 matches the 5m at 10:00 (=106) -
    # i.e., the resample bar at label 10:00 indeed represents 09:50-10:00.
    assert res15.loc[bin_10, "Close"] == 106.0


def test_ceil_bins_match_resample_1h():
    """Same invariant at 1h granularity: 5m at 09:05-10:00 all bin to 10:00."""
    # Need enough bars to span an hour boundary; build 09:00-10:00 inclusive.
    idx = pd.date_range(
        "2026-02-02 09:05", periods=12, freq="5min", tz="America/New_York"
    )
    bins = _coarse_bin_keys(idx, "1h")
    expected_bin = pd.Timestamp("2026-02-02 10:00", tz="America/New_York")
    # All 12 5m bars (09:05 through 10:00) should bin to 10:00.
    assert (bins == expected_bin).all()


def test_ceil_does_not_shift_boundary_timestamps():
    """5m bars exactly on a boundary (10:00, 10:15) must stay on that bin,
    not jump forward by a full bin width."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-02-02 10:00", tz="America/New_York"),
         pd.Timestamp("2026-02-02 10:15", tz="America/New_York"),
         pd.Timestamp("2026-02-02 11:00", tz="America/New_York")]
    )
    assert list(_coarse_bin_keys(idx, "15m")) == list(idx)
    assert list(_coarse_bin_keys(idx, "1h")) == [
        pd.Timestamp("2026-02-02 10:00", tz="America/New_York"),
        pd.Timestamp("2026-02-02 11:00", tz="America/New_York"),
        pd.Timestamp("2026-02-02 11:00", tz="America/New_York"),
    ]


def test_5m_freq_is_identity():
    """5m -> 5m bin map is the identity (no aggregation)."""
    idx = pd.date_range(
        "2026-02-02 09:30", periods=5, freq="5min", tz="America/New_York"
    )
    out = _coarse_bin_keys(idx, "5m")
    assert list(out) == list(idx)


def test_1d_normalize():
    """1d binning normalizes to midnight of the calendar day; matches the
    resample_ohlc 1d label after the +16h shift via .normalize().
    """
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-02-02 09:30", tz="America/New_York"),
         pd.Timestamp("2026-02-02 16:00", tz="America/New_York"),
         pd.Timestamp("2026-02-03 09:30", tz="America/New_York")]
    )
    out = _coarse_bin_keys(idx, "1d")
    assert list(out) == [
        pd.Timestamp("2026-02-02 00:00", tz="America/New_York"),
        pd.Timestamp("2026-02-02 00:00", tz="America/New_York"),
        pd.Timestamp("2026-02-03 00:00", tz="America/New_York"),
    ]
