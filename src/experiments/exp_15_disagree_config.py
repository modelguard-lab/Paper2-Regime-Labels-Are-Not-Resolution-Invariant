"""
Disagreement-day configuration breakdown (Suppl Table tab:var_uplift caption).

For each (asset, episode), counts how many disagreement days fall into each
of the two disagreement configurations (1h-crisis/1d-calm vs 1h-calm/1d-crisis).
Resolves whether the "single-configuration concentration" framing is exact or
approximate (e.g. >=98% in one config + remainder in the other).

Public entry point: ``run_disagree_config(raw_dirs, outputs_dir, assets)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from scipy import stats

from . import project_layout
from ..core.models import fit_regimes_per_frequency
from ..core.time_utils import ensure_ny_tz
from ..data.data_ib import iter_loaded_assets
from ..core.aggregation import native_day_label as _native_day_label
from ..core.config import FREQS, TZ

logger = logging.getLogger(__name__)


def breakdown_for_asset(df_5m: pd.DataFrame, stem: str, episode: str) -> dict[str, object]:
    """Per-asset 1h-vs-1d disagreement-day configuration breakdown.

    Per-day labels are derived from **native**-resolution label series, not
    from the 5m-aligned (ffill'd) versions. ``align_regimes_to_5m`` ffills
    the 1d label timestamped at 16:00 of date X onto the entire 5m grid,
    so groupby-on-calendar-day on the aligned series yields a 1-day
    phase-shifted day-level signal. The disagreement-day count and the
    50/50 binomial concentration test below would then be reporting
    "today's 1h regime vs yesterday's 1d regime", not "today vs today".
    Same fix as exp_06 (P0-S2).
    """
    coerced_index = ensure_ny_tz(df_5m.index)
    df_local = df_5m.set_axis(coerced_index, axis=0, copy=False)
    native = fit_regimes_per_frequency(df_local, stem, FREQS)

    day_lab_1h = _native_day_label(native["1h"])
    day_lab_1d = _native_day_label(native["1d"])
    valid = day_lab_1h.index.intersection(day_lab_1d.index)
    day_lab_1h = day_lab_1h.loc[valid]
    day_lab_1d = day_lab_1d.loc[valid]

    n_total = len(day_lab_1h)
    n_agree_calm = int(((day_lab_1h == 0) & (day_lab_1d == 0)).sum())
    n_agree_crisis = int(((day_lab_1h == 1) & (day_lab_1d == 1)).sum())
    n_dis_1hcrisis = int(((day_lab_1h == 1) & (day_lab_1d == 0)).sum())
    n_dis_1hcalm = int(((day_lab_1h == 0) & (day_lab_1d == 1)).sum())
    n_dis = n_dis_1hcrisis + n_dis_1hcalm

    # Two-sided exact binomial test of H0: 50/50 split between the two
    # disagreement configurations. A small p-value supports the
    # "single-configuration concentration" framing. Reports nan when n_dis=0.
    if n_dis > 0:
        binom = stats.binomtest(
            max(n_dis_1hcrisis, n_dis_1hcalm), n_dis, p=0.5, alternative="two-sided",
        )
        p_concentration = float(binom.pvalue)
    else:
        p_concentration = float("nan")

    return {
        "episode": episode,
        "symbol": stem,
        "n_total": n_total,
        "n_agree_calm": n_agree_calm,
        "n_agree_crisis": n_agree_crisis,
        "n_dis_total": n_dis,
        "n_dis_1hcrisis_1dcalm": n_dis_1hcrisis,
        "n_dis_1hcalm_1dcrisis": n_dis_1hcalm,
        "pct_disagree": 100.0 * n_dis / n_total if n_total else float("nan"),
        "pct_dominant_config": (
            100.0 * max(n_dis_1hcrisis, n_dis_1hcalm) / n_dis if n_dis else float("nan")
        ),
        "dominant_config": (
            "1h-crisis/1d-calm" if n_dis_1hcrisis >= n_dis_1hcalm
            else "1h-calm/1d-crisis"
        ) if n_dis else "n/a",
        "p_concentration_50_50": p_concentration,
    }


def run_disagree_config(
    raw_dirs: Sequence[tuple[str, Path]],
    outputs_dir: Path,
    assets: Iterable[str],
) -> pd.DataFrame:
    """Run the breakdown across one or more (episode, raw_dir) pairs."""
    rows: list[dict[str, object]] = []
    for episode, raw_dir in raw_dirs:
        if not raw_dir.exists():
            logger.info("Skip %s: %s missing", episode, raw_dir)
            continue
        def _warn_missing(sym, path):
            logger.warning("exp_15: skipping %s (no data file at %s)", sym, path)

        for _symbol, stem, df_5m in iter_loaded_assets(raw_dir, list(assets), on_missing=_warn_missing):
            try:
                row = breakdown_for_asset(df_5m, stem, episode)
                rows.append(row)
                logger.info(
                    "%s %s  total=%d  dis=%d (%.1f%%)  split=(%d, %d)  dominant=%.1f%% in %s",
                    episode, stem, row["n_total"], row["n_dis_total"],
                    row["pct_disagree"], row["n_dis_1hcrisis_1dcalm"],
                    row["n_dis_1hcalm_1dcrisis"], row["pct_dominant_config"],
                    row["dominant_config"],
                )
            except Exception as e:
                logger.warning("%s %s: failed: %s", episode, stem, e)

    df = pd.DataFrame(rows)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "disagree_config_breakdown.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved disagree-config breakdown: %s", out_path)
    return df


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_dirs: list[tuple[str, Path]] = [("2026", layout.raw_dir)]
    if layout.raw_dir_2022.exists():
        raw_dirs.append(("2022", layout.raw_dir_2022))
    logger.info("Disagreement-day configuration breakdown across episodes")
    run_disagree_config(raw_dirs, layout.outputs_dir, layout.assets)


if __name__ == "__main__":
    main()
