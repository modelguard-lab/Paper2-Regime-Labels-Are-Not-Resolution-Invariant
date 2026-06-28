"""Daily and rolling aggregations of cross-frequency ARI.

Reduces the per-bar aligned-label dict to four DataFrames the pipeline
writes as CSVs:

- ``daily_summary``: per-trading-day mean off-diagonal ARI plus crisis share
  per frequency
- ``daily_pairwise_ari``: long-format per-day pairwise ARI per frequency pair
- ``rolling_summary``: same as daily_summary but on a rolling N-day window
- ``rolling_pairwise_ari``: long-format pairwise ARI per rolling window
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_BLOCK_SIZE, DEFAULT_ROLLING_DAYS, TZ
from .metrics import cross_freq_ari_matrix, freq_pairs, mean_offdiag_ari
from .time_utils import ensure_ny_tz


def compute_daily_outputs(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
    rolling_days: int = DEFAULT_ROLLING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the four standard daily / rolling aggregation frames.

    Parameters
    ----------
    aligned : dict of pd.Series
        Per-frequency label series aligned to a common 5m index.
    freqs : tuple of str
        Frequency tuple in canonical order. Used both for the column ladder
        and as the loop ordering in the pairwise frames.
    rolling_days : int, default ``DEFAULT_ROLLING_DAYS``
        Window length of the rolling tables, in calendar trading days.

    Returns
    -------
    (daily, daily_pairs, rolling, rolling_pairs) : tuple of pd.DataFrame
    """
    base_freq = freqs[0]
    s_base = aligned.get(base_freq)
    if s_base is None or s_base.empty:
        # Build four independent empty frames so a downstream caller that
        # mutates one (e.g., ``df.attrs[...] = ...``) cannot inadvertently
        # affect the other three.
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    index_base = s_base.index
    day_labels = ensure_ny_tz(index_base).normalize()
    days = day_labels.unique().sort_values()

    daily_rows: list[dict[str, Any]] = []
    daily_pair_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    rolling_pair_rows: list[dict[str, Any]] = []

    for day in days:
        mask = day_labels == day
        day_index = index_base[mask]
        ari_df = cross_freq_ari_matrix(aligned, freqs, day_index)
        row: dict[str, Any] = {
            "date": day.strftime("%Y-%m-%d"),
            f"bars_{base_freq}": int(len(day_index)),
            "mean_offdiag_ari": mean_offdiag_ari(ari_df),
        }
        for freq in freqs:
            if not len(day_index):
                row[f"crisis_share_{freq}"] = np.nan
            else:
                day_vals = aligned[freq].reindex(day_index).dropna()
                row[f"crisis_share_{freq}"] = (
                    float(100.0 * (day_vals == 1).mean()) if not day_vals.empty else np.nan
                )
        daily_rows.append(row)
        for fa, fb in freq_pairs(freqs):
            daily_pair_rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "freq_a": fa,
                    "freq_b": fb,
                    "ari": ari_df.loc[fa, fb],
                }
            )

    # ``day_labels`` and ``index_base`` are aligned by construction (the
    # former is the NY-normalised view of the latter). Use pandas-native
    # ``isin`` so tz-aware comparisons are preserved end-to-end without
    # forcing a tz-naive ``datetime64[ns]`` cast on tz-aware indices.
    for end_idx in range(rolling_days - 1, len(days)):
        window_days = days[end_idx - rolling_days + 1: end_idx + 1]
        mask = day_labels.isin(window_days)
        window_index = index_base[mask]
        ari_df = cross_freq_ari_matrix(aligned, freqs, window_index)
        rolling_rows.append(
            {
                "window_start": window_days[0].strftime("%Y-%m-%d"),
                "window_end": window_days[-1].strftime("%Y-%m-%d"),
                "days_in_window": len(window_days),
                f"bars_{base_freq}": int(len(window_index)),
                "mean_offdiag_ari": mean_offdiag_ari(ari_df),
            }
        )
        for fa, fb in freq_pairs(freqs):
            rolling_pair_rows.append(
                {
                    "window_start": window_days[0].strftime("%Y-%m-%d"),
                    "window_end": window_days[-1].strftime("%Y-%m-%d"),
                    "freq_a": fa,
                    "freq_b": fb,
                    "ari": ari_df.loc[fa, fb],
                }
            )

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(daily_pair_rows),
        pd.DataFrame(rolling_rows),
        pd.DataFrame(rolling_pair_rows),
    )


def cl_roll_week_analysis(
    aligned: dict[str, pd.Series],
    index_base: pd.DatetimeIndex,
    freqs: tuple[str, ...],
    roll_dates: list[str] | None = None,
) -> dict[str, Any]:
    """Compare cross-frequency ARI on roll-week vs non-roll-week sub-windows.

    Specific to the WTI futures roll for the CL feed. If ``roll_dates`` is
    omitted, the function approximates roll weeks as **calendar days 18--22**
    of each month. This is a heuristic: the CME WTI roll is on the third
    business day before the 25th of the prior month, which falls between
    the 18th and 22nd and shifts with weekends. For the published
    diagnostic the calendar-day approximation is sufficient; this matches
    ``paper/supplement.tex`` (~line 119) after the round-2 fix. Pass an explicit
    ``roll_dates`` list to use exact CME roll dates.

    When ``roll_dates`` is provided, each roll date is widened to a +/- 2
    **business day** window via ``pd.tseries.offsets.BDay(2)`` so that the
    effective coverage is a uniform 5 trading days regardless of where the
    roll date falls in the week.
    """
    idx_ny = ensure_ny_tz(index_base)
    if roll_dates is None:
        days = idx_ny.day
        roll_mask = (days >= 18) & (days <= 22)
    else:
        # Vectorised replacement for the previous O(R x N) double loop:
        # build a boolean mask over the entire 5m index in one pass per
        # roll date using ``Series.between`` with NY-normalised endpoints.
        ts_norm = pd.Series(idx_ny.normalize(), index=idx_ny)
        roll_mask = np.zeros(len(index_base), dtype=bool)
        bday2 = pd.tseries.offsets.BDay(2)
        for rd in roll_dates:
            rd_ts = pd.Timestamp(rd)
            if rd_ts.tz is None:
                rd_ts = rd_ts.tz_localize(TZ)
            # +/- 2 business days each side gives uniform 5-trading-day
            # coverage; calendar days +/- 2 shrinks by ~40% when the roll
            # date abuts a weekend.
            start = (rd_ts - bday2).normalize()
            end = (rd_ts + bday2).normalize()
            roll_mask |= ts_norm.between(start, end).values

    roll_index = index_base[roll_mask]
    nonroll_index = index_base[~roll_mask]

    roll_ari = (
        cross_freq_ari_matrix(aligned, freqs, roll_index)
        if len(roll_index) >= DEFAULT_BLOCK_SIZE
        else pd.DataFrame()
    )
    nonroll_ari = (
        cross_freq_ari_matrix(aligned, freqs, nonroll_index)
        if len(nonroll_index) >= DEFAULT_BLOCK_SIZE
        else pd.DataFrame()
    )

    return {
        "roll_week_mean_ari": mean_offdiag_ari(roll_ari),
        "nonroll_week_mean_ari": mean_offdiag_ari(nonroll_ari),
        "roll_week_bars": len(roll_index),
        "nonroll_week_bars": len(nonroll_index),
        "roll_week_ari_matrix": roll_ari,
        "nonroll_week_ari_matrix": nonroll_ari,
    }


def native_day_label(native_series: pd.Series) -> pd.Series:
    """Aggregate native-resolution regime labels to one binary label per calendar day.

    For 1d input this is effectively the identity (one bar per trading day
    already, timestamped at 16:00); ``normalize()`` drops the time of day
    and the within-day mean is just that single label.

    For 1h input it is a within-day majority vote: the labels span trading
    hours of the day, and we flag the day as crisis when the within-day
    mean is >= 0.5.

    Why native, not aligned: ``align_regimes_to_5m`` ffills the 1d label
    timestamped at 16:00 of date X onto the 5m grid, so all 5m bars from
    04:00 of date X to 15:55 of date X carry yesterday's 1d label. A
    per-day groupby on the aligned series therefore produces a 1-day
    phase-shifted day-level signal. This helper bypasses that ffill.

    Promoted from a private helper in ``exp_06_var_uplift`` to a public
    utility so that ``exp_15_disagree_config`` and ``exp_17_em_restart_placebo``
    can import it without crossing the leading-underscore boundary.
    """
    s = native_series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    idx_ny = s.index.tz_convert(TZ) if s.index.tz is not None else s.index
    day = idx_ny.normalize()
    means = s.groupby(day).mean()
    return (means >= 0.5).astype(float)
