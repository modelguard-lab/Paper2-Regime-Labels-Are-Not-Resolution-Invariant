"""
Multi-frequency regime pipeline for Paper 2: top-level orchestrator.

Loads 5m OHLC, resamples to 15m/1h/1d, builds risk features, fits GMM per
frequency, aligns regime labels to the 5m time axis, and dispatches to
:func:`pipeline_asset.analyze_asset` for the per-asset analytic recipe.
This module owns the I/O glue:

- :func:`run_asset` runs one asset (GMM + HMM + calendar-window robustness)
  and persists outputs through ``pipeline_io``.
- :func:`run` loops over a list of assets, builds the panel-level
  ``pipeline_summary.csv`` with BH-FDR-adjusted permutation p-values.

Shared helpers (``_load_asset_5m``, ``_build_features_cache``,
``_cleanup_asset_outputs``) live in :mod:`pipeline_asset`. Pure-data
robustness helpers (``_attach_robustness_baseline_deltas``,
``_compute_robustness_ranges``, ``_write_robustness_report``) live in
:mod:`pipeline_robustness`; ``run_robustness`` orchestrates them from here.
Re-exports below preserve the public surface that tests, notebooks, and
downstream experiments import from ``src.workflows.pipeline``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import (
    DEFAULT_WINDOW_SCALE,
    EPISODES,
    FREQS,
    MODEL_GMM,
    MODEL_HMM,
)
from ..core.features import features, resample_ohlc, robust_filter_returns, window_spec
from ..core.metrics import bh_fdr, mean_offdiag_ari
from ..core.models import (
    fit_regime,
    fit_regime_hmm,
    fit_regime_model,
    gmm_fit_diagnostics,
)
from ..visualization.timeline import plot_timeline
from .pipeline_asset import (
    AssetAnalysis,
    _build_features_cache,
    _cleanup_asset_outputs,
    _load_asset_5m,
    analyze_asset,
)
from .pipeline_calendar import (
    _build_calendar_window_features,
    _run_calendar_window_robustness,
)
from .pipeline_io import (
    _save_baseline_csvs,
    _save_calendar_window_csv,
    _save_cl_roll_csv,
    _save_daily_rolling_csvs,
    _save_expanding_csv,
    _save_fallback_triggers,
    _save_fit_diagnostics_csv,
    _save_results_json,
    _save_source_snapshot,
)
from .pipeline_kxw import run_robustness

logger = logging.getLogger(__name__)


# Re-exports kept for tests / notebooks / downstream experiments that import
# these from ``src.workflows.pipeline``. Bind them to module-level names so
# ``from src.workflows.pipeline import X`` keeps working.
__all__ = [
    "AssetAnalysis",
    "FREQS",
    "analyze_asset",
    "features",
    "fit_regime",
    "fit_regime_hmm",
    "fit_regime_model",
    "gmm_fit_diagnostics",
    "mean_offdiag_ari",
    "resample_ohlc",
    "robust_filter_returns",
    "run",
    "run_asset",
    "run_robustness",
    "window_spec",
]


def run_asset(
    symbol: str,
    raw_dir: Path,
    outputs_dir: Path,
    event_window: tuple[str, str] | None = None,
    calm_window: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Run full-sample multi-frequency analysis for one asset."""
    loaded = _load_asset_5m(symbol, raw_dir, outputs_dir)
    if loaded is None:
        return {}
    stem, df_5m = loaded
    # Build the per-asset features cache once and share between the GMM
    # baseline and the HMM comparison; both fit on the same window_scale=1.0
    # log-vol features. The 6h calendar-window robustness uses a different
    # feature recipe so does not share the cache.
    features_by_freq = _build_features_cache(df_5m, stem, DEFAULT_WINDOW_SCALE)

    analysis = analyze_asset(
        symbol, df_5m, model=MODEL_GMM,
        event_window=event_window, calm_window=calm_window,
        features_by_freq=features_by_freq,
    )
    hmm_analysis = analyze_asset(
        symbol, df_5m, model=MODEL_HMM,
        event_window=event_window, calm_window=calm_window,
        features_by_freq=features_by_freq,
    )
    # Calendar-window features depend on (calendar_window, stem) only, not on
    # the model; build them once and share across the GMM and HMM calls.
    cal_window_features = _build_calendar_window_features(df_5m, stem)
    cal_window_gmm = _run_calendar_window_robustness(
        symbol, df_5m, model=MODEL_GMM, features_by_freq=cal_window_features,
    )
    cal_window_hmm = _run_calendar_window_robustness(
        symbol, df_5m, model=MODEL_HMM, features_by_freq=cal_window_features,
    )

    # Clean stale artefacts before writing fresh ones; ensures empty-result
    # matrices don't leave behind a previous run's CSVs.
    _cleanup_asset_outputs(outputs_dir, stem)
    # Snapshot the loaded source data into outputs/ as the anchor that
    # _save_results_json points back to via metadata.source_file. Done AFTER
    # both analyze_asset calls succeed so a partial failure (e.g., HMM
    # raising while GMM succeeded) does not leave a stale snapshot on disk.
    snapshot_path = _save_source_snapshot(df_5m, outputs_dir, stem)

    # Wrap the entire save block: if any artefact fails to write, remove the
    # source snapshot before re-raising so outputs/ ends up either with all
    # artefacts for this asset or none.
    try:
        ari_path = _save_baseline_csvs(analysis, outputs_dir, stem)
        timeline_path = outputs_dir / f"{stem}_timeline.png"
        plot_timeline(analysis["regimes_aligned"], stem, timeline_path)
        _save_fallback_triggers(analysis, hmm_analysis, outputs_dir, stem)
        _save_fit_diagnostics_csv(analysis, outputs_dir, stem, model_tag="gmm")
        _save_expanding_csv(analysis, outputs_dir, stem)
        _save_calendar_window_csv(cal_window_gmm, outputs_dir, stem)
        _save_cl_roll_csv(analysis, outputs_dir, stem)
        daily_summary_path, rolling_path = _save_daily_rolling_csvs(analysis, outputs_dir, stem)

        # HMM mirror: full battery of analyses persisted under {stem}_hmm_*.csv.
        hmm_ari_path = _save_baseline_csvs(hmm_analysis, outputs_dir, stem, prefix="hmm_")
        plot_timeline(
            hmm_analysis["regimes_aligned"], stem,
            outputs_dir / f"{stem}_hmm_timeline.png",
        )
        _save_fit_diagnostics_csv(hmm_analysis, outputs_dir, stem, model_tag="hmm")
        _save_expanding_csv(hmm_analysis, outputs_dir, stem, prefix="hmm_")
        _save_calendar_window_csv(cal_window_hmm, outputs_dir, stem, prefix="hmm_")
        _save_cl_roll_csv(hmm_analysis, outputs_dir, stem, prefix="hmm_")
        _save_daily_rolling_csvs(hmm_analysis, outputs_dir, stem, prefix="hmm_")

        # Single self-contained JSON: all GMM + HMM + both calendar-window
        # results in one analysable file paired to the {stem}_5m.csv source.
        _save_results_json(
            analysis, hmm_analysis, cal_window_gmm, cal_window_hmm,
            outputs_dir, stem,
        )
    except Exception:
        _cleanup_asset_outputs(outputs_dir, stem)
        raise

    logger.info("Saved %s baseline outputs (ARI=%s, daily=%s, rolling=%s)",
                symbol, ari_path, daily_summary_path, rolling_path)

    return {
        **analysis,
        "ari_path": ari_path,
        "timeline_path": timeline_path,
        "daily_summary_path": daily_summary_path,
        "rolling_7d_path": rolling_path,
        "hmm_ari_path": hmm_ari_path,
        "hmm_overall_mean_ari": hmm_analysis.get("overall_mean_ari_matrix"),
        "hmm_overall_mean_ari_pvalue_perm": hmm_analysis.get("overall_mean_ari_pvalue_perm"),
        "hmm_block_perm_pvalue": hmm_analysis.get("block_perm_pvalue"),
        "hmm_event_mean_ari": hmm_analysis.get("event_mean_ari"),
        "hmm_calm_mean_ari": hmm_analysis.get("calm_mean_ari"),
        "hmm_expanding_mean_ari": hmm_analysis.get("expanding_mean_ari"),
        "hmm_latest_rolling_7d_mean_ari": hmm_analysis.get("latest_rolling_7d_mean_ari"),
        "calendar_window_mean_ari": cal_window_gmm.get("mean_offdiag_ari"),
        "hmm_calendar_window_mean_ari": cal_window_hmm.get("mean_offdiag_ari"),
        # Surfaced so the panel-level pipeline_summary.csv writer can flag
        # per-(asset, freq) cells that fell back to the 80th-percentile
        # threshold.  The per-asset fallback_triggers.csv already records
        # this; the panel surface is what audit and reviewers read first.
        "hmm_fit_status": hmm_analysis.get("fit_status", {}),
        "hmm_fallback_flags": hmm_analysis.get("fallback_flags", {}),
    }


def run(
    raw_dir: Path | str,
    outputs_dir: Path | str,
    assets: list[str],
    episode: str | None = None,
) -> list[dict[str, Any]]:
    """Run pipeline for all assets and save summary outputs.

    If *episode* is provided (e.g. "2022_ukraine"), the corresponding
    event/calm windows from EPISODES are used instead of the defaults.
    Unknown episode names raise ValueError; pass ``None`` to use defaults.
    """
    event_window = None
    calm_window = None
    if episode is not None:
        if episode not in EPISODES:
            raise ValueError(
                f"Unknown episode {episode!r}; "
                f"registered episodes: {sorted(EPISODES.keys())}"
            )
        event_window, calm_window = EPISODES[episode]
        logger.info("Using episode %s: event=%s calm=%s", episode, event_window, calm_window)

    raw_dir = Path(raw_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for symbol in assets:
        try:
            result = run_asset(symbol, raw_dir, outputs_dir,
                               event_window=event_window, calm_window=calm_window)
        except Exception:
            # Fail loud: log full traceback for the failing asset (and how
            # many assets had succeeded so far so the operator can see how
            # much progress is on disk), then propagate to abort the run.
            logger.exception(
                "Asset %s failed after %d assets succeeded; aborting pipeline run.",
                symbol, len(results),
            )
            raise
        if result:
            results.append(result)

    rows = []
    kept = {r["symbol"] for r in results}
    results_by_symbol = {r["symbol"]: r for r in results}
    for symbol in assets:
        result = results_by_symbol.get(symbol)

        def _f(key: str) -> Any:
            return None if result is None else result.get(key)

        def _ci(key: str, idx: int) -> Any:
            ci = _f(key)
            return None if ci is None else ci[idx]

        # Per-frequency fit_status surfacing. ``fit_status`` distinguishes
        # ``normal`` (GMM/HMM fit succeeded) from ``pct_fallback`` (model
        # split was trivial; labels replaced by 80th-percentile log-vol
        # threshold) and ``degenerate_skipped`` (input too short or
        # constant; all-calm series). Reviewers reading the headline
        # cross-frequency ARI need to know which (asset, freq) cells
        # were not actually GMM-derived. ``any_pct_fallback`` is a
        # quick boolean flag for filter / sort.
        gmm_status = _f("fit_status") or {}
        hmm_status = _f("hmm_fit_status") or {}
        gmm_fb = _f("fallback_flags") or {}
        hmm_fb = _f("hmm_fallback_flags") or {}
        per_freq_status: dict[str, Any] = {}
        for freq in FREQS:
            per_freq_status[f"fit_status_{freq}"] = gmm_status.get(freq)
            per_freq_status[f"hmm_fit_status_{freq}"] = hmm_status.get(freq)

        rows.append(
            {
                "symbol": symbol,
                "included": symbol in kept,
                "overall_mean_ari_matrix": _f("overall_mean_ari_matrix"),
                "overall_n_valid_pairs": _f("overall_n_valid_pairs"),
                "overall_n_total_pairs": _f("overall_n_total_pairs"),
                "overall_mean_ari_perm_stat": _f("overall_mean_ari_perm_stat"),
                "overall_mean_ari_pvalue_perm": _f("overall_mean_ari_pvalue_perm"),
                "overall_mean_ari_null_ci_low": _ci("overall_mean_ari_null_ci", 0),
                "overall_mean_ari_null_ci_high": _ci("overall_mean_ari_null_ci", 1),
                "latest_rolling_7d_mean_ari": _f("latest_rolling_7d_mean_ari"),
                "rolling_ari_median": _f("rolling_ari_median"),
                "rolling_ari_q25": _f("rolling_ari_q25"),
                "rolling_ari_q75": _f("rolling_ari_q75"),
                "block_perm_pvalue": _f("block_perm_pvalue"),
                "block_perm_observed_stat": _f("block_perm_observed_stat"),
                "block_perm_null_ci_low": _ci("block_perm_null_ci", 0),
                "block_perm_null_ci_high": _ci("block_perm_null_ci", 1),
                "expanding_mean_ari": _f("expanding_mean_ari"),
                "calendar_window_mean_ari": _f("calendar_window_mean_ari"),
                "hmm_overall_mean_ari": _f("hmm_overall_mean_ari"),
                "hmm_overall_mean_ari_pvalue_perm": _f("hmm_overall_mean_ari_pvalue_perm"),
                "hmm_block_perm_pvalue": _f("hmm_block_perm_pvalue"),
                "hmm_event_mean_ari": _f("hmm_event_mean_ari"),
                "hmm_calm_mean_ari": _f("hmm_calm_mean_ari"),
                "hmm_expanding_mean_ari": _f("hmm_expanding_mean_ari"),
                "hmm_latest_rolling_7d_mean_ari": _f("hmm_latest_rolling_7d_mean_ari"),
                "hmm_calendar_window_mean_ari": _f("hmm_calendar_window_mean_ari"),
                **per_freq_status,
                "any_pct_fallback": any(
                    v == "pct_fallback" for v in (
                        list(gmm_status.values()) + list(hmm_status.values())
                    )
                ) if (gmm_status or hmm_status) else None,
                "any_fallback_triggered": any(
                    bool(v) for v in (
                        list(gmm_fb.values()) + list(hmm_fb.values())
                    )
                ) if (gmm_fb or hmm_fb) else None,
            }
        )
    summary_df = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR adjustment on the per-asset permutation p-values.
    # Two perm tests per asset (overall mean off-diag ARI and block-perm
    # variant) are exposed in the summary; we q-adjust each family separately
    # so the FDR controls for the per-asset multiplicity of the headline
    # statistic and its block-resampled robustness sibling, not across them.
    if not summary_df.empty:
        for col in (
            "overall_mean_ari_pvalue_perm",
            "block_perm_pvalue",
            "hmm_overall_mean_ari_pvalue_perm",
            "hmm_block_perm_pvalue",
        ):
            if col in summary_df.columns:
                _, q = bh_fdr(summary_df[col].values, alpha=0.05)
                summary_df[col + "_qvalue_bh"] = q
    summary_df.to_csv(outputs_dir / "pipeline_summary.csv", index=False)
    return results
