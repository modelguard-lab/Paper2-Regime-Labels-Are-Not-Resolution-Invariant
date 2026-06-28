"""Backfill tests for ``src.core.time_utils`` (P2-C: missing coverage).

Covers the DST + window-inclusivity contracts of:
- ``ensure_ny_tz``: tz-naive -> NY-localised, tz-aware -> NY-converted.
- ``subset_index_by_dates``: inclusive endpoints, DST spring-forward
  handling without raising, returns the original-tz index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import TZ
from src.core.time_utils import ensure_ny_tz, subset_index_by_dates


def test_subset_index_by_dates_inclusive_endpoints() -> None:
    """Both ``start_date`` and ``end_date`` are inclusive: every 5m bar
    falling on the boundary calendar dates must be retained."""
    idx = pd.date_range(
        "2026-01-01 09:30", "2026-01-05 16:00", freq="5min", tz=TZ
    )
    sub = subset_index_by_dates(idx, "2026-01-02", "2026-01-04")
    assert len(sub) > 0
    # Min on 2026-01-02 (the lower boundary).
    assert sub.min().date() == pd.Timestamp("2026-01-02").date()
    # Max on 2026-01-04 (the upper boundary, inclusive).
    assert sub.max().date() == pd.Timestamp("2026-01-04").date()
    # No bars from 2026-01-01 or 2026-01-05.
    dates = pd.DatetimeIndex(sub).normalize().unique()
    assert pd.Timestamp("2026-01-01", tz=TZ) not in dates
    assert pd.Timestamp("2026-01-05", tz=TZ) not in dates


def test_subset_index_by_dates_handles_dst_spring_forward() -> None:
    """2026-03-08 is the US DST spring-forward day: 02:00 -> 03:00 ET.  The
    helper must not raise when the window straddles the gap.
    """
    # 5m index spanning 2026-03-07 through 2026-03-09 (DST transition Sun
    # 2026-03-08).  Localise via shift_forward to skip the nonexistent
    # 02:00-03:00 hour cleanly.
    naive = pd.date_range("2026-03-07 00:00", "2026-03-09 23:55", freq="5min")
    idx = naive.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    sub = subset_index_by_dates(idx, "2026-03-07", "2026-03-09")
    assert len(sub) > 0
    # Boundary dates are present.
    dates = sub.normalize().unique()
    for d in ("2026-03-07", "2026-03-08", "2026-03-09"):
        assert pd.Timestamp(d, tz=TZ) in dates


def test_subset_index_by_dates_preserves_input_tz() -> None:
    """The helper compares dates in NY time but returns a slice of the input
    index, so the returned tz matches the caller's tz (UTC stays UTC)."""
    idx = pd.date_range("2026-01-01 14:30", "2026-01-05 21:00", freq="5min", tz="UTC")
    sub = subset_index_by_dates(idx, "2026-01-02", "2026-01-04")
    assert str(sub.tz) == "UTC"


def test_subset_index_by_dates_empty_window_returns_empty() -> None:
    """A range with end before start yields an empty index without raising."""
    idx = pd.date_range("2026-01-01 09:30", "2026-01-05 16:00", freq="5min", tz=TZ)
    sub = subset_index_by_dates(idx, "2026-02-10", "2026-02-05")
    assert len(sub) == 0


def test_ensure_ny_tz_localises_naive_index() -> None:
    """tz-naive index -> localised to America/New_York."""
    naive = pd.date_range("2024-06-03 09:30", periods=10, freq="5min")
    out = ensure_ny_tz(naive)
    assert out.tz is not None
    assert "New_York" in str(out.tz)


def test_ensure_ny_tz_converts_other_tz_to_ny() -> None:
    """tz-aware UTC index -> converted (not re-localised) to NY.  The
    underlying timestamps must remain the same instants in time."""
    utc = pd.date_range("2024-06-03 13:30", periods=10, freq="5min", tz="UTC")
    out = ensure_ny_tz(utc)
    assert "New_York" in str(out.tz)
    # Same instants: comparing UTC values must round-trip.
    np.testing.assert_array_equal(
        utc.tz_convert(TZ).values, out.values
    )
