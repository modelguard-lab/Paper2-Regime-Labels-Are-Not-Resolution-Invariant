"""EM-restart placebo for the conservative-resolution VaR uplift.

Isolates the algebraic-floor component of the headline VaR-uplift figures
in :mod:`exp_06_var_uplift` by applying the same ``max(|VaR_A|, |VaR_B|)``
rule to two EM-restart variants of *one* classifier on identical data.

Because both variants share the same sampling resolution, the resulting
uplift carries no resolution information; any non-zero value bounds the
algebraic-floor / classifier-noise component of the headline 1h-vs-1d
uplift.

Public entry point: :func:`run_em_restart_placebo`. CLI hook:
``python run.py extended_em_restart_placebo``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from . import project_layout
from ..data.data_ib import iter_loaded_assets
from ..core.config import TZ
from ..core.features import features, resample_ohlc
from ..core.models import (
    DEFAULT_GMM_N_INIT,
    is_degenerate_log_vol,
    prepare_log_vol,
    align_regimes_to_5m,
)
from ..core.time_utils import ensure_ny_tz
from ..core.aggregation import native_day_label as _native_day_label

logger = logging.getLogger(__name__)


PLACEBO_SEEDS: tuple[int, int] = (42, 2024)


def _fit_gmm_at_freq(
    df_5m: pd.DataFrame,
    stem: str,
    freq: str,
    seed: int,
    n_init: int,
) -> pd.Series | None:
    """Fit a 2-component GMM at one frequency with a chosen EM seed.

    Returns the native-frequency 0/1 label series, or ``None`` if the
    feature matrix is degenerate (one bar / constant log-vol).
    """
    ohlc = resample_ohlc(df_5m, freq)
    feats = features(ohlc, freq, stem=stem)
    X, log_vol, _finite = prepare_log_vol(feats, impute_with_median=True)
    if is_degenerate_log_vol(X, n_components=2, freq=freq):
        return None
    gmm = GaussianMixture(n_components=2, random_state=seed, n_init=n_init)
    try:
        gmm.fit(X)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning("GMM fit failed (%s, seed=%d): %s", freq, seed, e)
        return None
    crisis_cluster = int(np.argmax(gmm.means_.ravel()))
    pred = (gmm.predict(log_vol.reshape(-1, 1)) == crisis_cluster).astype(int)
    return pd.Series(pred, index=feats.index, name=f"{freq}_seed{seed}")


def _regime_var(rets: pd.Series, labels_5m: pd.Series, var_alpha: float, min_bars: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for g in (0, 1):
        sub = rets[labels_5m == g].dropna().values
        if len(sub) < min_bars:
            out[g] = float("nan")
        else:
            out[g] = float(np.quantile(sub, var_alpha))
    return out


def em_restart_placebo_one_freq(
    df_5m: pd.DataFrame,
    stem: str,
    placebo_freq: str,
    seeds: tuple[int, int] = PLACEBO_SEEDS,
    n_init: int = DEFAULT_GMM_N_INIT,
    var_alpha: float = 0.01,
    min_regime_bars: int = 50,
) -> dict[str, object]:
    """Apply max(|VaR_A|, |VaR_B|) where A/B are two EM-restart variants
    of the GMM at ``placebo_freq``.
    """
    coerced_index = ensure_ny_tz(df_5m.index)
    df_local = df_5m.set_axis(coerced_index, axis=0, copy=False)

    labels_A_native = _fit_gmm_at_freq(df_local, stem, placebo_freq, seeds[0], n_init)
    labels_B_native = _fit_gmm_at_freq(df_local, stem, placebo_freq, seeds[1], n_init)
    if labels_A_native is None or labels_B_native is None:
        return {
            "symbol": stem, "placebo_freq": placebo_freq,
            "n_init": n_init, "trustworthy": False,
            "note": "degenerate_at_placebo_freq",
        }

    aligned = align_regimes_to_5m(
        {"A": labels_A_native, "B": labels_B_native}, coerced_index,
    )
    aligned_A = aligned["A"]
    aligned_B = aligned["B"]

    rets = np.log(df_local["Close"] / df_local["Close"].shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan)
    bar_gaps = df_local.index.to_series().diff()
    session_break = (bar_gaps > pd.Timedelta(hours=1)).fillna(False)
    rets = rets.where(~session_break.values, np.nan)

    var_A = _regime_var(rets, aligned_A, var_alpha, min_regime_bars)
    var_B = _regime_var(rets, aligned_B, var_alpha, min_regime_bars)
    if any(np.isnan(v) for v in var_A.values()) or any(np.isnan(v) for v in var_B.values()):
        return {
            "symbol": stem, "placebo_freq": placebo_freq,
            "n_init": n_init, "trustworthy": False,
            "note": "thin_regime_pool",
        }

    day_A = _native_day_label(labels_A_native)
    day_B = _native_day_label(labels_B_native)
    days = day_A.index.intersection(day_B.index)
    day_A = day_A.loc[days]
    day_B = day_B.loc[days]

    var_per_day_A = day_A.map(var_A).astype(float)
    var_per_day_B = day_B.map(var_B).astype(float)
    valid = var_per_day_A.notna() & var_per_day_B.notna()
    var_per_day_A = var_per_day_A[valid]
    var_per_day_B = var_per_day_B[valid]
    day_A = day_A[valid]
    day_B = day_B[valid]
    if len(var_per_day_A) == 0:
        return {
            "symbol": stem, "placebo_freq": placebo_freq,
            "n_init": n_init, "trustworthy": False,
            "note": "no_valid_days",
        }

    v_A = var_per_day_A.abs()
    v_B = var_per_day_B.abs()
    eps = 1e-9
    max_v = np.maximum(v_A, v_B)
    uplift_vs_A = (max_v / np.maximum(v_A, eps)) - 1.0
    disagree = (day_A != day_B)
    n_total = int(len(disagree))
    n_dis = int(disagree.sum())
    bar_disagree = (aligned_A != aligned_B).reindex(coerced_index).fillna(False)
    return {
        "symbol": stem,
        "placebo_freq": placebo_freq,
        "n_init": n_init,
        "seed_A": seeds[0],
        "seed_B": seeds[1],
        "n_days_valid": n_total,
        "n_disagree_days": n_dis,
        "pct_disagree_days": 100.0 * n_dis / n_total if n_total else float("nan"),
        "pct_disagree_bars": float(bar_disagree.mean() * 100),
        "avg_uplift_pct_full_sample": float(uplift_vs_A.mean() * 100),
        "avg_uplift_pct_disagree_only": (
            float(uplift_vs_A[disagree].mean() * 100) if n_dis else 0.0
        ),
        "trustworthy": True,
        "note": "",
    }


def run_em_restart_placebo(
    raw_dir: Path,
    outputs_dir: Path,
    assets: Iterable[str],
    placebo_freqs: tuple[str, ...] = ("1h", "1d"),
    seeds: tuple[int, int] = PLACEBO_SEEDS,
    n_init_settings: tuple[int, ...] = (DEFAULT_GMM_N_INIT, 1),
    var_alpha: float = 0.01,
) -> pd.DataFrame:
    """Run the EM-restart placebo for every (asset, placebo_freq, n_init) cell.

    ``n_init_settings`` defaults to ``(10, 1)``: ``n_init=10`` matches the
    canonical pipeline (the placebo the paper headline must beat) and
    ``n_init=1`` is the worst-case classifier-noise sensitivity (the
    upper bound on the algebraic-floor contribution).
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _symbol, stem, df_5m in iter_loaded_assets(raw_dir, list(assets)):
        for freq in placebo_freqs:
            for n_init in n_init_settings:
                try:
                    res = em_restart_placebo_one_freq(
                        df_5m, stem, placebo_freq=freq,
                        seeds=seeds, n_init=n_init, var_alpha=var_alpha,
                    )
                except (ValueError, KeyError, RuntimeError) as e:
                    logger.warning(
                        "placebo failed for %s %s n_init=%d: %s",
                        stem, freq, n_init, e,
                    )
                    res = {
                        "symbol": stem, "placebo_freq": freq, "n_init": n_init,
                        "trustworthy": False, "note": f"error:{e}",
                    }
                rows.append(res)
                logger.info(
                    "%s %s n_init=%d: pct_disagree_days=%.1f, uplift_full=%.2f%%",
                    stem, freq, n_init,
                    res.get("pct_disagree_days", float("nan")),
                    res.get("avg_uplift_pct_full_sample", float("nan")),
                )
    summary = pd.DataFrame(rows)
    out_path = outputs_dir / "var_uplift_em_restart_placebo.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Saved EM-restart placebo: %s", out_path)
    return summary


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("EM-restart VaR-uplift placebo (2026)")
    run_em_restart_placebo(layout.raw_dir, layout.outputs_dir, layout.assets)
    if layout.raw_dir_2022.exists():
        logger.info("EM-restart VaR-uplift placebo (2022)")
        run_em_restart_placebo(
            layout.raw_dir_2022, layout.outputs_dir_2022, layout.assets_2022,
        )


if __name__ == "__main__":
    main()
