"""
Calm-day-only subsample ARI (referee Q7: do results hold under non-stress
conditions, or are they an artefact of the geopolitical episodes?).

Recomputes mean off-diagonal ARI on a calm-day subsample (days with
realised vol below the median, additionally excluding the peak-stress
window) while holding the full-sample GMM regime boundaries fixed.

Public entry point: ``run_calm_day_subsample(raw_dir, outputs_dir, assets, exclude_window, quantile)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import project_layout
from ..core.config import EPISODES
from ..data.data_ib import iter_loaded_assets
from ..core.features import daily_realised_vol
from ..core.models import fit_aligned_regimes
from ..core.config import FREQS, TZ
from ..core.metrics import cross_freq_ari_matrix, freq_pairs, mean_offdiag_ari

logger = logging.getLogger(__name__)


def _count_valid_offdiag_pairs(ari_mat: pd.DataFrame, freqs: tuple[str, ...]) -> int:
    """Count off-diagonal cells of ``ari_mat`` that are non-NaN.

    Used to expose whether the calm-day subsample's headline mean was
    computed over the full 6-pair set or degraded to fewer pairs because
    the 1d frequency had < 10 jointly-valid bars (P1-11).
    """
    n_valid = 0
    for fa, fb in freq_pairs(freqs):
        if pd.notna(ari_mat.loc[fa, fb]):
            n_valid += 1
    return n_valid


def calm_day_subsample_ari(
    df_5m: pd.DataFrame,
    stem: str,
    exclude_window: tuple[str, str] | None = None,
    quantile: float = 0.5,
) -> dict[str, object]:
    """Recompute mean off-diagonal ARI on the calm-day subsample.

    Procedure:
      1. Fit GMM regimes per frequency on the FULL sample so the decision
         boundary is held fixed.
      2. Compute daily realised volatility from 5m returns.
      3. Define the calm-day set as days with daily RV below ``quantile``,
         additionally excluding any day in ``exclude_window``.
      4. Restrict the 5m index to calm days and compute the ARI matrix.
    """
    template: dict[str, object] = {
        "symbol": stem,
        "quantile": quantile,
        "calm_n_days": 0,
        "total_n_days": 0,
        "rv_cutoff": float("nan"),
        "rv_median": float("nan"),
        "full_sample_ari": float("nan"),
        "calm_subsample_ari": float("nan"),
        "n_valid_pairs": 0,
        "ari_5m_15m": float("nan"),
        "ari_15m_1h": float("nan"),
        "ari_1h_1d": float("nan"),
    }

    aligned = fit_aligned_regimes(df_5m, stem, FREQS)

    # Use the canonical per-day RV helper (drops O=H=L=C placeholder
    # bars before summing squared returns). For 2022 GLD this shifts
    # ~12/62 calm vs non-calm classifications relative to the legacy
    # all-bars RV, because the placeholder share differs day-to-day
    # and tilts the quantile cutoff.
    rv_per_day = daily_realised_vol(df_5m)
    if rv_per_day.empty:
        return {**template, "calm_n_days": 0}
    idx_ny = df_5m.index.tz_convert(TZ)
    day_key = pd.Series(idx_ny.normalize(), index=df_5m.index)

    cutoff = float(rv_per_day.quantile(quantile))
    calm_days = set(rv_per_day[rv_per_day < cutoff].index)
    if exclude_window is not None:
        def _to_ny(ts):
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize(TZ)
            else:
                t = t.tz_convert(TZ)
            return t.normalize()

        ex_start = _to_ny(exclude_window[0])
        ex_end = _to_ny(exclude_window[1])
        calm_days = {d for d in calm_days if not (ex_start <= d <= ex_end)}
    if not calm_days:
        return {
            **template,
            "calm_n_days": 0,
            "total_n_days": int(rv_per_day.size),
            "rv_cutoff": cutoff,
            "rv_median": float(rv_per_day.median()),
        }

    mask = day_key.isin(calm_days).values
    calm_idx = df_5m.index[mask]
    full_ari = mean_offdiag_ari(cross_freq_ari_matrix(aligned, FREQS))
    calm_ari_mat = cross_freq_ari_matrix(aligned, FREQS, calm_idx)
    calm_ari = mean_offdiag_ari(calm_ari_mat)
    n_valid_pairs = _count_valid_offdiag_pairs(calm_ari_mat, FREQS)
    if n_valid_pairs < 6:
        logger.warning(
            "calm-subsample %s: headline mean ARI is over %d/6 off-diagonal "
            "pairs (the 1d-involving pairs probably failed the min_valid=10 "
            "guard because calm_n_days=%d gives < 10 daily bars).",
            stem, n_valid_pairs, len(calm_days),
        )

    def _safe_ari(mat, fa, fb):
        if fa in mat.index and fb in mat.columns:
            v = mat.loc[fa, fb]
            return float(v) if not pd.isna(v) else float("nan")
        return float("nan")

    return {
        **template,
        "calm_n_days": int(len(calm_days)),
        "total_n_days": int(rv_per_day.size),
        "rv_cutoff": cutoff,
        "rv_median": float(rv_per_day.median()),
        "full_sample_ari": full_ari,
        "calm_subsample_ari": calm_ari,
        "n_valid_pairs": int(n_valid_pairs),
        "ari_5m_15m": _safe_ari(calm_ari_mat, "5m", "15m"),
        "ari_15m_1h": _safe_ari(calm_ari_mat, "15m", "1h"),
        "ari_1h_1d": _safe_ari(calm_ari_mat, "1h", "1d"),
    }


def run_calm_day_subsample(
    raw_dir: Path,
    outputs_dir: Path,
    assets: Iterable[str],
    exclude_window: tuple[str, str] | None,
    quantile: float = 0.5,
) -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _symbol, stem, df_5m in iter_loaded_assets(raw_dir, list(assets)):
        try:
            res = calm_day_subsample_ari(
                df_5m, stem,
                exclude_window=exclude_window,
                quantile=quantile,
            )
        except (ValueError, KeyError, RuntimeError) as e:
            logger.warning("calm_day_subsample failed for %s: %s", stem, e)
            continue
        rows.append(res)
    summary = pd.DataFrame(rows)
    summary.to_csv(outputs_dir / "calm_day_subsample_ari.csv", index=False)
    logger.info("Saved calm-day subsample ARI with %d rows", len(summary))
    return summary


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Each entry: (episode_label, raw_dir, outputs_dir, EPISODES key, assets).
    episodes = [
        ("2026", layout.raw_dir, layout.outputs_dir, "2026_iran", layout.assets),
        ("2022", layout.raw_dir_2022, layout.outputs_dir_2022, "2022_ukraine", layout.assets_2022),
    ]
    for label, raw_dir, outputs_dir, episode_key, assets in episodes:
        if not raw_dir.exists():
            logger.info("Skip calm-day subsample %s: raw dir missing", label)
            continue
        event_window, _ = EPISODES[episode_key]
        logger.info("Calm-day subsample ARI (%s)", label)
        run_calm_day_subsample(
            raw_dir, outputs_dir, assets,
            exclude_window=event_window, quantile=0.5,
        )


if __name__ == "__main__":
    main()
