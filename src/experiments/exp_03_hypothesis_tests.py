"""
Kruskal-Wallis + Mann-Whitney U on cross-resolution ARI groups.

Pools off-diagonal ARI values from per-asset cross-frequency matrices
(produced by the main pipeline) into three frequency-pair categories:
adjacent intraday, non-adjacent intraday, and intraday-daily. Tests
whether these categories differ in mean ARI.

Two flavours of the omnibus KW test are reported:

* ``kruskal_p`` - pooled across episodes (the "pooled (correlated) version").
  Treats every (episode, asset, freq-pair) tuple as an independent
  observation. Because the same 4 assets contribute to two episodes, the
  24 intraday-daily values are non-independent, so this p-value is biased
  low. Kept for backward compatibility.
* ``kruskal_p_<episode>`` and ``kruskal_p_fisher`` - KW run separately
  per episode and combined via Fisher's method
  (``scipy.stats.combine_pvalues(method="fisher")``). This is the
  recommended omnibus statistic since each episode contributes
  independent draws.

Public entry point: ``run_hypothesis_tests(outputs_dirs, target_dir, assets)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from . import project_layout
from ..core.metrics import bh_fdr
from ..data.data_ib import canonical_stem
from ..core.config import FREQS

logger = logging.getLogger(__name__)


def _pair_type(fa: str, fb: str) -> str:
    """Classify a (fa, fb) frequency pair into one of three buckets.

    The bucketing is **by the coarsest frequency involved**, NOT by
    distance on the resolution ladder 5m < 15m < 1h < 1d:

    * ``adjacent_intraday`` -- (5m, 15m) only.
    * ``nonadjacent_intraday`` -- (5m, 1h) AND (15m, 1h). Note that
      (15m, 1h) is *adjacent* on the ladder; it is grouped here because
      the bucket name reflects "intraday pairs that involve 1h".
    * ``intraday_daily`` -- any pair involving 1d.

    The headline KW + MWU monotonicity test conditions on this
    "coarsest-freq" bucketing, not on a ladder-distance bucketing. If
    you want a strict ladder-distance test, build a separate three-bucket
    function: {(5m,15m), (15m,1h), (1h,1d)} as adjacent, etc. Changing
    the bucketing here regenerates every supplement-table p-value.
    """
    pair = tuple(sorted((fa, fb), key=FREQS.index))
    if pair == ("5m", "15m"):
        return "adjacent_intraday"
    if pair in (("5m", "1h"), ("15m", "1h")):
        return "nonadjacent_intraday"
    if pair in (("5m", "1d"), ("15m", "1d"), ("1h", "1d")):
        return "intraday_daily"
    return "other"


def _episode_key(outputs_dir: Path) -> str:
    """Derive a short episode label from the outputs directory name.

    ``outputs/`` -> "2026"; ``outputs_2022/`` -> "2022". Falls back to the
    directory's leaf name for any other layout.
    """
    name = outputs_dir.name
    if name.endswith("_2022"):
        return "2022"
    if name == "outputs":
        return "2026"
    return name


def per_episode_ari_groups(
    outputs_dirs: list[Path],
    assets: Iterable[str],
) -> dict[str, dict[str, list[float]]]:
    """Per-episode breakdown of off-diagonal ARI by frequency-pair category.

    Returns a dict keyed by episode label (e.g., ``"2026"``, ``"2022"``)
    whose values are the same three-key ``adjacent / nonadjacent /
    intraday_daily`` group dicts as :func:`pooled_ari_groups`. Used by
    :func:`run_hypothesis_tests` to run KW separately per episode and
    combine via Fisher's method.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for outputs_dir in outputs_dirs:
        ep = _episode_key(outputs_dir)
        groups: dict[str, list[float]] = {
            "adjacent_intraday": [],
            "nonadjacent_intraday": [],
            "intraday_daily": [],
        }
        for symbol in assets:
            stem = canonical_stem(symbol)
            path = outputs_dir / f"{stem}_cross_freq_ari.csv"
            if not path.exists():
                continue
            mat = pd.read_csv(path, index_col=0)
            for i, fa in enumerate(FREQS):
                for j, fb in enumerate(FREQS):
                    if j <= i:
                        continue
                    group = _pair_type(fa, fb)
                    if group not in groups:
                        continue
                    cell = mat.loc[fa, fb]
                    # NaN ARI cells (freq pair failed the ``min_valid=10``
                    # gate inside ``cross_freq_ari_matrix`` due to short
                    # subwindows / sparse data) MUST be filtered out:
                    # ``stats.kruskal`` and ``stats.mannwhitneyu`` propagate
                    # any NaN input to a NaN p-value with a RuntimeWarning,
                    # which silently invalidates the entire episode's test.
                    if pd.notna(cell):
                        groups[group].append(float(cell))
        out[ep] = groups
    return out


def pooled_ari_groups(outputs_dirs: list[Path], assets: Iterable[str]) -> dict[str, list[float]]:
    """Pool off-diagonal ARI values into three groups across outputs directories
    (typically outputs/ and outputs_2022/).

    NOTE: pooling treats each (episode, asset, pair) value as independent.
    For the recommended per-episode + Fisher's combination, see
    :func:`per_episode_ari_groups`.
    """
    pooled: dict[str, list[float]] = {
        "adjacent_intraday": [],
        "nonadjacent_intraday": [],
        "intraday_daily": [],
    }
    per_ep = per_episode_ari_groups(outputs_dirs, assets)
    for groups in per_ep.values():
        for k, v in groups.items():
            pooled[k].extend(v)
    return pooled


def _kw_three_groups(groups: dict[str, list[float]]) -> tuple[float, float] | tuple[float, float]:
    """KW H-statistic and p-value over the three pair categories.

    Returns ``(nan, nan)`` when any group is empty (cannot run KW).
    """
    g1 = np.asarray(groups["adjacent_intraday"], dtype=float)
    g2 = np.asarray(groups["nonadjacent_intraday"], dtype=float)
    g3 = np.asarray(groups["intraday_daily"], dtype=float)
    if g1.size == 0 or g2.size == 0 or g3.size == 0:
        return float("nan"), float("nan")
    h, p = stats.kruskal(g1, g2, g3)
    return float(h), float(p)


def run_hypothesis_tests(
    outputs_dirs: list[Path],
    target_dir: Path,
    assets: Iterable[str],
) -> dict[str, object]:
    """Pool off-diagonal ARI by frequency-pair category and run KW + Mann-Whitney.

    The two Mann-Whitney tests use ``alternative="greater"`` because the
    pre-registered direction (motivated by the cross-frequency-dissonance
    hypothesis) is that ARI decreases monotonically as the resolution gap
    widens: adjacent_intraday > nonadjacent_intraday > intraday_daily.
    p-values therefore correspond to a one-sided test of the predicted
    ordering, not an exploratory two-sided comparison.

    Reports both the pooled KW p-value (``kruskal_p`` - biased low because
    episodes contribute correlated samples) and per-episode KW p-values
    combined via Fisher's method (``kruskal_p_fisher`` - recommended).
    """
    per_ep = per_episode_ari_groups(outputs_dirs, assets)
    groups = pooled_ari_groups(outputs_dirs, assets)
    g1 = np.asarray(groups["adjacent_intraday"], dtype=float)
    g2 = np.asarray(groups["nonadjacent_intraday"], dtype=float)
    g3 = np.asarray(groups["intraday_daily"], dtype=float)
    # Pooled (correlated) version - kept for back-compat.
    if g1.size == 0 or g2.size == 0 or g3.size == 0:
        kw_stat, kw_p = float("nan"), float("nan")
    else:
        kw_stat, kw_p = stats.kruskal(g1, g2, g3)
    # MWU empty-group guard: scipy raises on a 0-sample group; without this
    # the whole hypothesis-tests CSV fails to write when one of the
    # frequency-pair categories has no contributing assets (e.g., a
    # single-episode subset where the daily-pair files are all missing).
    if g1.size == 0 or g2.size == 0:
        u12, p12 = float("nan"), float("nan")
    else:
        u12, p12 = stats.mannwhitneyu(g1, g2, alternative="greater")
    if g2.size == 0 or g3.size == 0:
        u23, p23 = float("nan"), float("nan")
    else:
        u23, p23 = stats.mannwhitneyu(g2, g3, alternative="greater")
    # BH-FDR over the two pre-registered one-sided MWU comparisons (the KW
    # omnibus is reported for context but not part of the multiplicity family
    # since it tests a different null).  With m=2 the BH q-values are simply
    # max(p_adj_at_step) of the ranked vector; we still go through the helper
    # so the column semantics match the rest of the codebase.
    _, qvals = bh_fdr([p12, p23], alpha=0.05)
    q12, q23 = float(qvals[0]), float(qvals[1])

    # Per-episode KW + Fisher's method (P1-2). Each episode runs the same
    # three-group KW; the resulting per-episode p-values are combined via
    # ``scipy.stats.combine_pvalues(method="fisher")``. This treats episodes
    # as independent, removing the within-asset cross-episode correlation
    # that biases the pooled ``kruskal_p`` low.
    # Note: Fisher's combination assumes independent per-episode KW p-values, but
    # the same 4 assets contribute to both episodes - there is within-asset
    # correlation across episodes that this method does not address. The
    # headline `kruskal_p_fisher` should be interpreted with that caveat. A more
    # rigorous alternative (paired permutation over (asset, episode) pairs) is
    # left for a future revision.
    out: dict[str, object] = {
        "mean_adjacent_intraday": float(g1.mean()) if g1.size else float("nan"),
        "median_adjacent_intraday": float(np.median(g1)) if g1.size else float("nan"),
        "n_adjacent_intraday": int(g1.size),
        "mean_nonadjacent_intraday": float(g2.mean()) if g2.size else float("nan"),
        "median_nonadjacent_intraday": float(np.median(g2)) if g2.size else float("nan"),
        "n_nonadjacent_intraday": int(g2.size),
        "mean_intraday_daily": float(g3.mean()) if g3.size else float("nan"),
        "median_intraday_daily": float(np.median(g3)) if g3.size else float("nan"),
        "n_intraday_daily": int(g3.size),
        "kruskal_H": float(kw_stat),
        "kruskal_p": float(kw_p),  # pooled (correlated) version - back-compat only
        "mannwhitney_adj_vs_nonadj_U": float(u12),
        "mannwhitney_adj_vs_nonadj_p": float(p12),
        "mannwhitney_adj_vs_nonadj_q_bh": q12,
        "mannwhitney_nonadj_vs_daily_U": float(u23),
        "mannwhitney_nonadj_vs_daily_p": float(p23),
        "mannwhitney_nonadj_vs_daily_q_bh": q23,
    }

    valid_pvals: list[float] = []
    for ep_label, ep_groups in per_ep.items():
        h_ep, p_ep = _kw_three_groups(ep_groups)
        out[f"kruskal_H_{ep_label}"] = h_ep
        out[f"kruskal_p_{ep_label}"] = p_ep
        if np.isfinite(p_ep):
            valid_pvals.append(p_ep)

    if len(valid_pvals) >= 2:
        fisher_stat, fisher_p = stats.combine_pvalues(valid_pvals, method="fisher")
        out["kruskal_p_fisher"] = float(fisher_p)
        out["kruskal_fisher_chi2"] = float(fisher_stat)
    elif len(valid_pvals) == 1:
        # Only one episode available; Fisher reduces to that p-value.
        out["kruskal_p_fisher"] = float(valid_pvals[0])
        out["kruskal_fisher_chi2"] = float("nan")
    else:
        out["kruskal_p_fisher"] = float("nan")
        out["kruskal_fisher_chi2"] = float("nan")

    target_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(target_dir / "hypothesis_tests.csv", index=False)
    logger.info("Saved hypothesis tests: %s", out)
    return out


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    outputs_dirs = [layout.outputs_dir]
    if layout.outputs_dir_2022.exists():
        outputs_dirs.append(layout.outputs_dir_2022)
    logger.info("Hypothesis tests on pooled ARI across %d output dirs", len(outputs_dirs))
    run_hypothesis_tests(outputs_dirs, layout.outputs_dir, layout.assets)


if __name__ == "__main__":
    main()
