"""
Formal statistical test: is cross-frequency ARI different between stress
(wartime) and calm (peacetime) windows?

Consumes existing outputs:
  - outputs/bootstrap_five_day_windows.csv         (2026 US-Iran)
  - outputs_2022/bootstrap_five_day_windows.csv    (2022 Russia-Ukraine)

Each row already contains calm_ari, stress_ari, and bootstrap percentiles
for one asset. We pool the 6 asset-by-episode pairs and test whether the
stress and calm ARIs differ.

Output:
  - outputs/stress_vs_calm_test.csv     summary stats
  - outputs/stress_vs_calm_test.txt     human-readable report

Answers the question: does the US-Iran episode produce statistically
different cross-frequency ARI than calm periods?
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import project_layout

logger = logging.getLogger(__name__)


def load_pairs(
    out_dirs: list[tuple[Path, str]],
) -> pd.DataFrame:
    """Load bootstrap CSVs from one or more (outputs_dir, episode_label) pairs.

    Skips any directory whose ``bootstrap_five_day_windows.csv`` is missing
    so the test can run on a single-episode subset (e.g., 2026 only).
    """
    rows = []
    for out_dir, episode in out_dirs:
        csv = out_dir / "bootstrap_five_day_windows.csv"
        if not csv.exists():
            logger.warning("stress_vs_calm: skip %s, missing %s", episode, csv)
            continue
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            # P1-13: prefer the renamed tail-rank keys, fall back to the
            # deprecated ``p_*_vs_boot`` aliases for backward compatibility
            # with bootstrap CSVs produced by older runs.
            tail_calm = r.get("tail_rank_calm_vs_boot",
                              r.get("p_calm_vs_boot", float("nan")))
            tail_stress = r.get("tail_rank_stress_vs_boot",
                                r.get("p_stress_vs_boot", float("nan")))
            rows.append({
                "episode": episode,
                "asset": r["symbol"],
                "calm_ARI": r["calm_ari"],
                "stress_ARI": r["stress_ari"],
                "diff_stress_minus_calm": r["stress_ari"] - r["calm_ari"],
                "boot_median": r["boot_median"],
                "boot_q025": r["boot_q025"],
                "boot_q975": r["boot_q975"],
                "tail_rank_calm_vs_boot": tail_calm,
                "tail_rank_stress_vs_boot": tail_stress,
                # Keep the legacy column names too so consumers that read
                # this CSV directly still see them (deprecated aliases).
                "p_calm_vs_boot": tail_calm,
                "p_stress_vs_boot": tail_stress,
            })
    return pd.DataFrame(rows)


def _paired_tests(stress: np.ndarray, calm: np.ndarray) -> dict[str, float]:
    """Run paired-t / Wilcoxon / sign test on a single (stress, calm) sample.

    Used both for the pooled (cross-episode) test and per-episode tests.
    Returns NaNs for sample sizes < 2 rather than raising.
    """
    diffs = stress - calm
    n = len(diffs)
    if n < 2:
        return {
            "paired_t_stat": float("nan"), "paired_t_p": float("nan"),
            "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
            "sign_test_p": float("nan"),
            "n_pairs": int(n), "n_pairs_nonzero": 0,
            "mean_diff": float(diffs.mean()) if n else float("nan"),
            "std_diff": float(diffs.std(ddof=1)) if n > 1 else float("nan"),
            "median_diff": float(np.median(diffs)) if n else float("nan"),
            "stress_gt_calm_count": int((diffs > 0).sum()),
            "stress_gt_calm_nonzero_count": int(((diffs != 0) & (diffs > 0)).sum()),
        }

    t_stat, p_t = stats.ttest_rel(stress, calm)
    try:
        # Pratt drops zero-difference pairs from the rank computation but
        # keeps n_eff = full n for the variance term; exact distribution is
        # used for n <= ~25 to avoid the asymptotic approximation that the
        # default ("auto") would silently switch to on borderline n.
        w_stat, p_w = stats.wilcoxon(
            stress, calm, zero_method="pratt", method="exact",
        )
    except Exception:
        # Fallback for scipy versions where method="exact" rejects the input
        # (e.g., excessive ties relative to n).
        try:
            w_stat, p_w = stats.wilcoxon(stress, calm, zero_method="pratt")
        except Exception:
            w_stat, p_w = float("nan"), float("nan")

    # Sign test drops zero diffs from both numerator and denominator.
    nonzero_mask = diffs != 0
    n_eff = int(nonzero_mask.sum())
    n_gt = int((diffs > 0).sum())
    n_gt_nonzero = int((diffs[nonzero_mask] > 0).sum())
    if n_eff > 0:
        binom = stats.binomtest(n_gt_nonzero, n_eff, p=0.5, alternative="two-sided")
        p_sign = float(binom.pvalue)
    else:
        p_sign = float("nan")

    return {
        "n_pairs": int(n),
        "n_pairs_nonzero": n_eff,
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)),
        "median_diff": float(np.median(diffs)),
        "stress_gt_calm_count": n_gt,
        "stress_gt_calm_nonzero_count": n_gt_nonzero,
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(p_t),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(p_w),
        "sign_test_p": float(p_sign),
    }


def run_tests(full: pd.DataFrame) -> dict:
    """Paired stress-vs-calm tests, pooled and per-episode.

    Reports both the pooled paired tests (n = 4 assets x n_episodes) - which
    ignore episode clustering and treat the same 4 assets as independent
    across episodes - and per-episode paired tests (n = 4 each), which
    respect the clustering. The per-episode versions are the recommended
    headline numbers (P1-4); the pooled version is kept for back-compat
    only and labelled as such.
    """
    calm_raw = full["calm_ARI"].values
    stress_raw = full["stress_ARI"].values
    mask = ~(np.isnan(calm_raw) | np.isnan(stress_raw))
    calm = calm_raw[mask]
    stress = stress_raw[mask]
    if len(calm) < 2:
        pooled = {
            "paired_t_stat": float("nan"), "paired_t_p": float("nan"),
            "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
            "sign_test_p": float("nan"),
            "n_pairs": int(len(calm)), "n_pairs_nonzero": 0,
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "median_diff": float("nan"),
            "stress_gt_calm_count": 0,
            "stress_gt_calm_nonzero_count": 0,
        }
    else:
        pooled = _paired_tests(stress, calm)

    # Per-episode tests (P1-4): n = 4 assets each, no pooling.
    per_episode_results: dict[str, dict[str, float]] = {}
    for ep, sub in full.groupby("episode"):
        calm_raw_ep = sub["calm_ARI"].values
        stress_raw_ep = sub["stress_ARI"].values
        mask_ep = ~(np.isnan(calm_raw_ep) | np.isnan(stress_raw_ep))
        calm_ep = calm_raw_ep[mask_ep]
        stress_ep = stress_raw_ep[mask_ep]
        if len(calm_ep) < 2:
            per_episode_results[str(ep)] = {
                "paired_t_stat": float("nan"), "paired_t_p": float("nan"),
                "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
                "sign_test_p": float("nan"),
                "n_pairs": int(len(calm_ep)), "n_pairs_nonzero": 0,
                "mean_diff": float("nan"),
                "std_diff": float("nan"),
                "median_diff": float("nan"),
                "stress_gt_calm_count": 0,
                "stress_gt_calm_nonzero_count": 0,
            }
        else:
            per_episode_results[str(ep)] = _paired_tests(stress_ep, calm_ep)

    # Episode-key normalisation: paper text uses bare years.
    def _short(ep: str) -> str:
        if ep.startswith("2026"):
            return "2026"
        if ep.startswith("2022"):
            return "2022"
        return ep

    out: dict[str, object] = {
        # Pooled (back-compat). Same keys as before for any downstream
        # consumer; "ignores episode clustering" is the documented caveat.
        "n_pairs": pooled["n_pairs"],
        "n_pairs_nonzero": pooled["n_pairs_nonzero"],
        "mean_diff": pooled["mean_diff"],
        "std_diff": pooled["std_diff"],
        "median_diff": pooled["median_diff"],
        "stress_gt_calm_count": pooled["stress_gt_calm_count"],
        "stress_gt_calm_nonzero_count": pooled["stress_gt_calm_nonzero_count"],
        "paired_t_stat": pooled["paired_t_stat"],
        "paired_t_p_pooled": pooled["paired_t_p"],   # ignores episode clustering
        "wilcoxon_stat": pooled["wilcoxon_stat"],
        "wilcoxon_p_pooled": pooled["wilcoxon_p"],   # ignores episode clustering
        "sign_test_p_pooled": pooled["sign_test_p"], # ignores episode clustering
        # Back-compat aliases - same value as *_pooled.
        "paired_t_p": pooled["paired_t_p"],
        "wilcoxon_p": pooled["wilcoxon_p"],
        "sign_test_p": pooled["sign_test_p"],
    }

    # Per-episode columns (recommended).
    for ep_label, res in per_episode_results.items():
        short = _short(ep_label)
        out[f"paired_t_p_{short}"] = res["paired_t_p"]
        out[f"wilcoxon_p_{short}"] = res["wilcoxon_p"]
        out[f"sign_test_p_{short}"] = res["sign_test_p"]
        out[f"n_pairs_{short}"] = res["n_pairs"]

    boot_cols = ["tail_rank_calm_vs_boot", "tail_rank_stress_vs_boot"]
    for c in boot_cols:
        if c not in full.columns:
            # Older bootstrap CSVs only carry the deprecated key; load_pairs
            # already aliases it but be defensive in case run_tests is
            # called on a hand-built DataFrame.
            full[c] = full.get(c.replace("tail_rank_", "p_"), float("nan"))
    out["all_boot_p_gt_0_3"] = bool(np.all(np.nan_to_num(full[boot_cols].values, nan=1.0) > 0.3))
    if np.all(np.isnan(full[boot_cols].values)):
        out["min_boot_p"] = float("nan")
    else:
        out["min_boot_p"] = float(np.nanmin(full[boot_cols].values))
    return out


def write_report(full: pd.DataFrame, results: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("Formal test: does wartime differ from peacetime in cross-frequency ARI?")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Paired data ({results['n_pairs']} asset-by-episode pairs):")
    lines.append("  " + "-" * 70)
    lines.append(f"  {'episode':<22}{'asset':<8}{'calm':>9}{'stress':>9}{'diff':>10}")
    lines.append("  " + "-" * 70)
    for _, r in full.iterrows():
        lines.append(
            f"  {r['episode']:<22}{r['asset']:<8}"
            f"{r['calm_ARI']:>9.4f}{r['stress_ARI']:>9.4f}"
            f"{r['diff_stress_minus_calm']:>+10.4f}"
        )
    lines.append("")
    lines.append(f"n = {results['n_pairs']}")
    lines.append(f"mean(stress - calm) = {results['mean_diff']:+.4f}")
    lines.append(f"std(stress - calm)  = {results['std_diff']:.4f}")
    lines.append(f"median diff         = {results['median_diff']:+.4f}")
    lines.append(f"stress > calm in {results['stress_gt_calm_count']}/{results['n_pairs']} pairs")
    lines.append("")
    lines.append("Hypothesis tests (H0: stress ARI = calm ARI):")
    lines.append("  Pooled (ignores episode clustering - back-compat only):")
    lines.append(f"    Paired t-test :  t = {results['paired_t_stat']:+.3f}, p = {results['paired_t_p_pooled']:.3f}")
    lines.append(f"    Wilcoxon      :  W = {results['wilcoxon_stat']:.2f}, p = {results['wilcoxon_p_pooled']:.3f}")
    lines.append(f"    Exact sign    :  p = {results['sign_test_p_pooled']:.3f}")
    lines.append("  Per-episode (recommended; n=4 each, no pooling):")
    for ep in ("2026", "2022"):
        if f"paired_t_p_{ep}" in results:
            lines.append(
                f"    {ep}: paired-t p = {results[f'paired_t_p_{ep}']:.3f}, "
                f"Wilcoxon p = {results[f'wilcoxon_p_{ep}']:.3f}, "
                f"sign p = {results[f'sign_test_p_{ep}']:.3f} "
                f"(n = {results[f'n_pairs_{ep}']})"
            )
    lines.append("")
    lines.append("Bootstrap check (are calm/stress 5-day windows unusual vs random 5-day draws?):")
    lines.append(f"  All bootstrap p-values > 0.3?  {results['all_boot_p_gt_0_3']}")
    lines.append(f"  Minimum bootstrap p-value     = {results['min_boot_p']:.3f}")
    lines.append("")
    lines.append("Conclusion:")
    p_2026 = results.get("paired_t_p_2026", float("nan"))
    p_2022 = results.get("paired_t_p_2022", float("nan"))
    w_2026 = results.get("wilcoxon_p_2026", float("nan"))
    w_2022 = results.get("wilcoxon_p_2022", float("nan"))
    if not pd.isna(w_2026) and not pd.isna(w_2022):
        indistinguishable = w_2026 > 0.1 and w_2022 > 0.1
    elif not pd.isna(p_2026) and not pd.isna(p_2022):
        indistinguishable = p_2026 > 0.1 and p_2022 > 0.1  # fallback
    else:
        indistinguishable = results["paired_t_p_pooled"] > 0.1  # final fallback
    if indistinguishable and results["all_boot_p_gt_0_3"]:
        lines.append("  Stress and calm ARIs are statistically INDISTINGUISHABLE. The US-Iran")
        lines.append("  episode does NOT produce different cross-frequency label agreement than")
        lines.append("  peacetime windows. This is consistent with the calm-day subsample")
        lines.append("  robustness result in section 3.4: resolution dissonance is a structural")
        lines.append("  property of the cross-frequency decomposition, not a stress-specific")
        lines.append("  phenomenon.")
        lines.append("")
        lines.append("  Implication for Paper 2 framing: the 2026 US-Iran window is the empirical")
        lines.append("  laboratory on which quantitative evidence was computed, not a necessary")
        lines.append("  condition for the phenomenon. The title's 'Evidence from the 2026 US-Iran")
        lines.append("  Escalation' describes the data window, not a claim of episode-specificity.")
    else:
        lines.append("  Evidence of stress-vs-calm difference. Paper 2's episode-specific framing")
        lines.append("  is supported. Review individual asset differences below.")
    lines.append("")
    return "\n".join(lines)


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dirs = [(layout.outputs_dir, "2026_US_Iran")]
    if layout.outputs_dir_2022.exists():
        out_dirs.append((layout.outputs_dir_2022, "2022_Russia_Ukraine"))

    full = load_pairs(out_dirs)
    if full.empty:
        logger.warning("stress_vs_calm: no bootstrap CSVs found in any episode dir; nothing to do")
        return
    results = run_tests(full)

    layout.outputs_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(layout.outputs_dir / "stress_vs_calm_test.csv", index=False)
    pd.DataFrame([results]).to_csv(
        layout.outputs_dir / "stress_vs_calm_test_summary.csv", index=False
    )
    report = write_report(full, results)
    (layout.outputs_dir / "stress_vs_calm_test.txt").write_text(report, encoding="utf-8")
    logger.info("\n%s", report)


if __name__ == "__main__":
    main()
