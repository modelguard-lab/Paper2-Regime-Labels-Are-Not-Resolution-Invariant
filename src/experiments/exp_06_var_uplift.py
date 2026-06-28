"""
Resolution-conditional VaR uplift on 1h-vs-1d disagreement days
(referee Q9: economic value of the cross-frequency dissonance signal).

For each calendar day, looks up the regime-conditional 1% VaR under the
1h and the 1d classifier; reports the population statistics of the
max/min ratio on disagreement days, the always-conservative average
uplift relative to the 1d single-resolution baseline, and the
disagreement-day-only uplift.

Public entry point: ``run_var_uplift(raw_dir, outputs_dir, assets, var_alpha)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import project_layout
from ..data.data_ib import iter_loaded_assets
from ..core.models import (
    align_regimes_to_5m,
    fit_regimes_per_frequency,
)
from ..core.time_utils import ensure_ny_tz
from ..core.config import FREQS
from ..core.aggregation import native_day_label as _native_day_label

logger = logging.getLogger(__name__)


def var_uplift_1h_vs_1d(
    df_5m: pd.DataFrame,
    stem: str,
    var_alpha: float = 0.01,
    min_regime_bars: int = 50,
) -> dict[str, object]:
    """Resolution-conditional VaR uplift on 1h-vs-1d disagreement days.

    Method:
      1. Fit GMM regimes per frequency on the FULL sample.
      2. Aggregate NATIVE 1h / 1d label streams to per-day labels (majority
         vote within calendar day, threshold 0.5).
      3. Pool 5m returns within each regime (under the 1h classifier and the
         1d classifier separately, both ffill'd onto the 5m grid for the
         pooling step) and compute the empirical ``var_alpha``-quantile
         per regime.
      4. Compute population statistics of max/min ratio on disagreement days,
         the always-conservative average uplift relative to the 1d baseline,
         and the disagreement-day-only uplift.

    Per-day label derivation uses the **native**-resolution series rather
    than the 5m-aligned ffill, because the 1d label timestamp sits at
    16:00 of the trading day; a naive ffill onto the 5m grid leaves the
    entire 04:00-15:55 portion of date X carrying yesterday's 1d label,
    so a per-day groupby on the aligned series would produce a 1-day
    phase-shifted day-level signal. Native 1d labels are
    one-bar-per-trading-day with timestamp 16:00 of that day; native 1h
    labels span trading hours with 1h spacing. Both are normalised to
    the calendar day directly and aggregated to a single binary label.

    Session-boundary cleanup (P1-14): the first 5m return of each session
    is computed off the previous session's close and therefore includes
    the overnight gap. Those returns are detected by gaps > 1 hour in the
    5m DatetimeIndex and set to NaN before regime aggregation, so the
    empirical 99% quantile is not inflated by overnight gaps.

    Degenerate regimes (fewer than ``min_regime_bars`` bars after pooling) are
    flagged and the result is marked as not-trustworthy in the output.
    """
    # P0-S2 (exp_06): manually replicate the prep that ``fit_aligned_regimes``
    # used to do internally so we can keep the NATIVE per-frequency series
    # for per-day label derivation while still building the 5m-aligned
    # version for the VaR-pooling step below.
    coerced_index = ensure_ny_tz(df_5m.index)
    df_local = df_5m.set_axis(coerced_index, axis=0, copy=False)
    native = fit_regimes_per_frequency(df_local, stem, FREQS)
    aligned = align_regimes_to_5m(native, coerced_index)

    rets = np.log(df_5m["Close"] / df_5m["Close"].shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan)
    # P1-14: drop the overnight return at each session boundary. Detect
    # session breaks by gaps in the 5m index larger than 1 hour (the
    # nominal intraday spacing is 5 minutes; an inter-session gap is at
    # minimum the overnight close). The first bar of the entire series
    # is already NaN from the .shift(1).
    bar_gaps = df_5m.index.to_series().diff()
    session_break = (bar_gaps > pd.Timedelta(hours=1)).fillna(False)
    if session_break.any():
        rets = rets.where(~session_break.values, np.nan)
    df_bars = pd.DataFrame({"r": rets})
    df_bars["lab1h"] = aligned["1h"].reindex(df_bars.index).astype(float)
    df_bars["lab1d"] = aligned["1d"].reindex(df_bars.index).astype(float)
    df_bars = df_bars.dropna()

    var_by_regime: dict[str, dict[int, float]] = {"1h": {}, "1d": {}}
    counts_by_regime: dict[str, dict[int, int]] = {"1h": {}, "1d": {}}
    degenerate_regimes: list[str] = []
    for label_col, key in (("lab1h", "1h"), ("lab1d", "1d")):
        for g in sorted(df_bars[label_col].unique()):
            sub = df_bars.loc[df_bars[label_col] == g, "r"].values
            counts_by_regime[key][int(g)] = int(len(sub))
            if len(sub) < min_regime_bars:
                degenerate_regimes.append(f"{key}={int(g)}({len(sub)} bars)")
                var_by_regime[key][int(g)] = float("nan")
            else:
                var_by_regime[key][int(g)] = float(np.quantile(sub, var_alpha))

    expected_regimes = {0, 1}
    present_regimes_1h = set(var_by_regime["1h"].keys())
    present_regimes_1d = set(var_by_regime["1d"].keys())
    missing_1h = expected_regimes - present_regimes_1h
    missing_1d = expected_regimes - present_regimes_1d
    if missing_1h:
        degenerate_regimes.append(f"1h_missing={sorted(missing_1h)}")
    if missing_1d:
        degenerate_regimes.append(f"1d_missing={sorted(missing_1d)}")

    day_lab_1h = _native_day_label(native["1h"])
    day_lab_1d = _native_day_label(native["1d"])
    valid_days = day_lab_1h.index.intersection(day_lab_1d.index)
    day_lab_1h = day_lab_1h.loc[valid_days]
    day_lab_1d = day_lab_1d.loc[valid_days]

    var_per_day_1h = day_lab_1h.map(var_by_regime["1h"])
    var_per_day_1d = day_lab_1d.map(var_by_regime["1d"])
    valid_mask = var_per_day_1h.notna() & var_per_day_1d.notna()
    var_per_day_1h = var_per_day_1h[valid_mask]
    var_per_day_1d = var_per_day_1d[valid_mask]
    day_lab_1h = day_lab_1h[valid_mask]
    day_lab_1d = day_lab_1d[valid_mask]

    if len(var_per_day_1h) == 0:
        return {"symbol": stem, "n_days_valid": 0, "trustworthy": False,
                "degenerate_regimes": ";".join(degenerate_regimes) or ""}

    v1h = var_per_day_1h.abs()
    v1d = var_per_day_1d.abs()
    eps = 1e-9
    ratio = pd.Series(
        np.maximum(v1h, v1d) / np.maximum(np.minimum(v1h, v1d), eps),
        index=v1h.index,
    )
    max_post = np.maximum(v1h, v1d)
    uplift_vs_1d = (max_post / np.maximum(v1d, eps)) - 1.0

    disagree = (day_lab_1h != day_lab_1d)
    n_total = int(len(disagree))
    n_dis = int(disagree.sum())
    ratio_dis = ratio[disagree]
    uplift_dis = uplift_vs_1d[disagree]

    trustworthy = (
        len(degenerate_regimes) == 0
        and n_dis >= 5
        and float(uplift_vs_1d.mean()) < 100.0
    )

    return {
        "symbol": stem,
        "n_days_valid": n_total,
        "n_disagree": n_dis,
        "pct_disagree": 100.0 * n_dis / n_total if n_total else float("nan"),
        "ratio_median": float(ratio_dis.median()) if len(ratio_dis) else float("nan"),
        "ratio_q75": float(ratio_dis.quantile(0.75)) if len(ratio_dis) else float("nan"),
        "ratio_q90": float(ratio_dis.quantile(0.90)) if len(ratio_dis) else float("nan"),
        "avg_uplift_pct_full_sample": float(uplift_vs_1d.mean() * 100),
        "avg_uplift_pct_disagree_only": float(uplift_dis.mean() * 100) if len(uplift_dis) else float("nan"),
        "trustworthy": bool(trustworthy),
        "degenerate_regimes": ";".join(degenerate_regimes) or "",
        "n_bars_1h_calm": counts_by_regime["1h"].get(0, 0),
        "n_bars_1h_crisis": counts_by_regime["1h"].get(1, 0),
        "n_bars_1d_calm": counts_by_regime["1d"].get(0, 0),
        "n_bars_1d_crisis": counts_by_regime["1d"].get(1, 0),
    }


def run_var_uplift(
    raw_dir: Path,
    outputs_dir: Path,
    assets: Iterable[str],
    var_alpha: float = 0.01,
) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _symbol, stem, df_5m in iter_loaded_assets(raw_dir, list(assets)):
        try:
            res = var_uplift_1h_vs_1d(df_5m, stem, var_alpha=var_alpha)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.warning("var_uplift failed for %s: %s", stem, e)
            rows.append({"symbol": stem, "trustworthy": False, "error": str(e)})
            continue
        rows.append(res)
    summary = pd.DataFrame(rows)
    summary.to_csv(outputs_dir / "var_uplift_by_resolution.csv", index=False)
    logger.info("Saved VaR-uplift table with %d rows", len(summary))
    return summary


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Resolution-conditional VaR uplift (Q9, 2026)")
    run_var_uplift(layout.raw_dir, layout.outputs_dir, layout.assets)
    if layout.raw_dir_2022.exists():
        logger.info("Resolution-conditional VaR uplift (2022)")
        run_var_uplift(layout.raw_dir_2022, layout.outputs_dir_2022, layout.assets_2022)


if __name__ == "__main__":
    main()
