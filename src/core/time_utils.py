"""Date / time helpers used across the pipeline and extended-analyses experiments.

Theme-scoped: only TZ-aware datetime / DatetimeIndex helpers belong here.
Other shared helpers should live in their own theme module
(e.g., ``core.array_utils``) rather than landing in a generic ``utils.py``.
"""

from __future__ import annotations

import pandas as pd

from .config import TZ


def ensure_ny_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``index`` localised / converted to NY time.

    The whole core layer assumes its input ``DatetimeIndex`` is already
    tz-aware NY time. This helper makes that invariant explicit for the
    handful of entry points (``aggregation.compute_daily_outputs``,
    ``features.resample_ohlc`` for 1d, ``cl_roll_week_analysis``) that
    used to call ``.tz_convert`` blindly and would crash with
    ``AttributeError`` on a tz-naive caller.
    """
    if index.tz is None:
        return index.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    return index.tz_convert(TZ)


def _localise_endpoint(date_like: object) -> pd.Timestamp:
    """Coerce a date-string or Timestamp to a NY-localised Timestamp.

    DST transitions (e.g., 2026-03-08 in NY) are handled by
    ``nonexistent="shift_forward"`` and ``ambiguous="NaT"`` so a window
    boundary that lands on the spring-forward gap does not raise. Already
    tz-aware Timestamps are converted rather than re-localised.
    """
    ts = pd.Timestamp(date_like)
    if ts.tz is None:
        return ts.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    return ts.tz_convert(TZ)


def subset_index_by_dates(
    index_5m: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    """Select 5m timestamps whose NY calendar date is in ``[start_date, end_date]``.

    Both endpoints are inclusive. The input index is converted (or localised)
    to NY time before the comparison so the date arithmetic happens in the
    same calendar a desk reads off a screen. The returned index preserves
    the caller's original tz (callers downstream of this helper continue to
    use whatever tz they passed in).
    """
    if index_5m.tz is None:
        idx = index_5m.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    else:
        idx = index_5m.tz_convert(TZ)
    d = idx.normalize()
    start_ts = _localise_endpoint(start_date).normalize()
    end_ts = _localise_endpoint(end_date).normalize()
    mask = (d >= start_ts) & (d <= end_ts)
    return index_5m[mask]
