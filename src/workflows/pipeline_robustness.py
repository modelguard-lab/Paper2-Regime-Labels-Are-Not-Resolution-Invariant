"""Pure-data robustness processing helpers for the K x window_scale sweep.

These functions operate only on DataFrames produced by the sweep; they have no
dependency on pipeline fitting internals and can be tested in isolation.
The orchestration entry point (``run_robustness``) stays in ``pipeline.py``
because it calls fitting helpers that live there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import DEFAULT_GMM_K, DEFAULT_WINDOW_SCALE

logger = logging.getLogger(__name__)


def _attach_robustness_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Annotate the sweep summary with baseline values and delta columns."""
    baseline_mask = (
        (summary["k"] == DEFAULT_GMM_K)
        & (summary["window_scale"] == DEFAULT_WINDOW_SCALE)
    )
    if not baseline_mask.any():
        # Fail loud: every downstream column (delta_*, ranges report) is
        # meaningless without the baseline anchor. Silent NaN-fill would let
        # a misconfigured sweep ship outputs that look correct.
        raise ValueError(
            f"robustness baseline (K={DEFAULT_GMM_K}, "
            f"window_scale={DEFAULT_WINDOW_SCALE}) has no matching row in "
            f"the sweep summary; cannot compute baseline deltas. "
            f"Sweep contained K values {sorted(summary['k'].unique())} and "
            f"window_scales {sorted(summary['window_scale'].unique())}."
        )
    baseline = (
        summary[baseline_mask]
        .set_index("symbol")
        .rename(columns={
            "overall_mean_ari": "baseline_overall_mean_ari",
            "latest_rolling_mean_ari": "baseline_latest_rolling_mean_ari",
        })[["baseline_overall_mean_ari", "baseline_latest_rolling_mean_ari"]]
    )
    summary = summary.join(baseline, on="symbol")
    # Symbols with no baseline row at all
    sweep_symbols = set(summary["symbol"].unique())
    baseline_symbols = set(baseline.index)
    no_baseline_row = sweep_symbols - baseline_symbols
    if no_baseline_row:
        raise ValueError(f"pipeline_robustness: symbols missing baseline rows entirely: {sorted(no_baseline_row)}")

    # Symbols where the baseline row exists but ARI is NaN - warn, don't raise
    nan_baseline = summary[summary["baseline_overall_mean_ari"].isna()]["symbol"].unique()
    if len(nan_baseline) > 0:
        import warnings as _w
        _w.warn(f"pipeline_robustness: symbols with NaN baseline ARI: {sorted(nan_baseline)}", RuntimeWarning)
    summary["delta_overall_mean_ari"] = (
        summary["overall_mean_ari"] - summary["baseline_overall_mean_ari"]
    )
    summary["delta_latest_rolling_mean_ari"] = (
        summary["latest_rolling_mean_ari"] - summary["baseline_latest_rolling_mean_ari"]
    )
    return summary.sort_values(["symbol", "k", "window_scale"]).reset_index(drop=True)


def _compute_robustness_ranges(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, grp in summary.groupby("symbol"):
        baseline_rows = grp[
            (grp["k"] == DEFAULT_GMM_K)
            & (grp["window_scale"] == DEFAULT_WINDOW_SCALE)
        ]
        if baseline_rows.empty:
            logger.warning(
                "robustness ranges: no baseline (K=%d, ws=%s) row for %s; skipping",
                DEFAULT_GMM_K, DEFAULT_WINDOW_SCALE, symbol,
            )
            continue
        base = baseline_rows.iloc[0]
        baseline_ari = base["overall_mean_ari"]
        if pd.isna(baseline_ari):
            continue
        rows.append({
            "symbol": symbol,
            "baseline_overall_mean_ari": float(base["overall_mean_ari"]),
            "overall_mean_ari_min": float(grp["overall_mean_ari"].min()),
            "overall_mean_ari_max": float(grp["overall_mean_ari"].max()),
            "baseline_ari_15m_1h": float(base["ari_15m_1h"]),
            "ari_15m_1h_min": float(grp["ari_15m_1h"].min()),
            "ari_15m_1h_max": float(grp["ari_15m_1h"].max()),
            "baseline_latest_rolling_mean_ari": float(base["latest_rolling_mean_ari"]),
            "latest_rolling_mean_ari_min": float(grp["latest_rolling_mean_ari"].min()),
            "latest_rolling_mean_ari_max": float(grp["latest_rolling_mean_ari"].max()),
        })
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _fmt_or_nd(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def _write_robustness_report(ranges: pd.DataFrame, outputs_dir: Path) -> None:
    lines = [
        "# Robustness sweep",
        "",
        "Baseline is `K=2`, `window_scale=1.0` (2h for 5m/15m, 24 bars for 1h, 5 bars for 1d).",
        "",
        "## ARI ranges by asset",
        "",
    ]
    for _, row in ranges.iterrows():
        lines.append(
            f"- `{row['symbol']}`: baseline overall ARI "
            f"{_fmt_or_nd(row['baseline_overall_mean_ari'])}; sweep range "
            f"{_fmt_or_nd(row['overall_mean_ari_min'])} to "
            f"{_fmt_or_nd(row['overall_mean_ari_max'])}. "
            f"Latest rolling mean baseline "
            f"{_fmt_or_nd(row['baseline_latest_rolling_mean_ari'])}; sweep range "
            f"{_fmt_or_nd(row['latest_rolling_mean_ari_min'])} to "
            f"{_fmt_or_nd(row['latest_rolling_mean_ari_max'])}."
        )
    (outputs_dir / "robustness_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
