"""CSV / plot persistence helpers for the multi-frequency pipeline.

All functions accept the analysis dict returned by ``analyze_asset`` and a
target ``outputs_dir`` + ``stem`` string; they have no dependency on other
pipeline internals and can be tested or reused independently.

Most save helpers accept a ``prefix`` kwarg for the model tag in filenames:
``prefix=""`` writes the GMM-default paths (e.g. ``{stem}_cross_freq_ari.csv``);
``prefix="hmm_"`` writes the HMM mirror (``{stem}_hmm_cross_freq_ari.csv``).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..visualization.rolling_ari import plot_rolling_ari as _plot_rolling_ari

logger = logging.getLogger(__name__)


def _save_source_snapshot(
    df_5m: pd.DataFrame, outputs_dir: Path, stem: str,
) -> Path:
    """Snapshot the loaded (post-validation, tz-localised) 5m OHLC frame to CSV.

    The output filename ``{stem}_5m.csv`` mirrors the raw input filename and
    serves as the source-data anchor for the per-asset JSON summary written
    by :func:`_save_analysis_summary_json`.
    """
    path = outputs_dir / f"{stem}_5m.csv"
    df_5m.to_csv(path)
    return path


def _scalar_for_json(obj: Any) -> Any:
    """Encode a single non-container value into JSON-safe form.

    Handles numpy scalars, NaN/inf (-> None), pandas Timestamps, NaT,
    and plain Python primitives. Raises ``TypeError`` on anything else
    so the caller can decide to skip or fail.
    """
    if obj is None or obj is pd.NaT:
        return None
    # IMPORTANT: ``bool`` is a subclass of ``int`` in Python and ``np.bool_``
    # would also match ``np.integer`` on some numpy versions. The bool checks
    # MUST run before the integer/float checks below or ``True`` gets coerced
    # to ``1`` (or ``1.0``) and the JSON loses semantic type fidelity.
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (pd.Timedelta, np.timedelta64)):
        return pd.Timedelta(obj).isoformat()
    logger.debug("_scalar_for_json: falling back to repr() for %s", type(obj).__name__)
    return repr(obj)


def _df_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    """Serialise a DataFrame to a JSON-safe dict (orient='split' style).

    Returns ``{"index": [...], "columns": [...], "data": [[...], ...]}``
    where index entries become ISO strings if the index is datetime-like
    and cell values pass through :func:`_scalar_for_json`.
    """
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None and len(idx) > 0:
            raise ValueError("_dataframe_to_jsonable: DatetimeIndex must be tz-aware")
        index_list = [t.isoformat() for t in idx]
    else:
        index_list = [_scalar_for_json(v) for v in idx.tolist()]
    columns_list = [str(c) for c in df.columns.tolist()]
    data: list[list[Any]] = []
    for row in df.itertuples(index=False, name=None):
        data.append([_scalar_for_json(v) for v in row])
    return {"index": index_list, "columns": columns_list, "data": data}


def _series_to_dict(s: pd.Series) -> dict[str, Any]:
    """Serialise a Series to ``{"index": [...], "data": [...]}``."""
    idx = s.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None and len(idx) > 0:
            raise ValueError("_dataframe_to_jsonable: DatetimeIndex must be tz-aware")
        index_list = [t.isoformat() for t in idx]
    else:
        index_list = [_scalar_for_json(v) for v in idx.tolist()]
    data = [_scalar_for_json(v) for v in s.tolist()]
    return {"index": index_list, "data": data}


def _coerce_for_json(obj: Any) -> Any:
    """Recursively normalise nested dicts/lists/tuples for JSON output.

    Pandas DataFrames and Series are serialised inline (orient='split')
    so the resulting JSON is self-contained; loading it back gives a
    dict with all matrices and label series intact.
    """
    if isinstance(obj, dict):
        return {str(k): _coerce_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_for_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _df_to_dict(obj)
    if isinstance(obj, pd.Series):
        return _series_to_dict(obj)
    if isinstance(obj, pd.Index):
        return [_coerce_for_json(v) for v in obj.tolist()]
    return _scalar_for_json(obj)


def _save_results_json(
    gmm_analysis: dict[str, Any],
    hmm_analysis: dict[str, Any],
    cal_window_gmm: dict[str, Any],
    cal_window_hmm: dict[str, Any],
    outputs_dir: Path,
    stem: str,
) -> Path:
    """Persist the full per-asset analysis as a single self-contained JSON.

    The file ``{stem}_5m_results.json`` carries everything ``analyze_asset``
    produced for both GMM and HMM, plus the calendar-window robustness for
    both models, plus metadata. DataFrames and Series are inlined
    (orient='split' style), so a downstream consumer can ``json.load(...)``
    and reconstruct the full object with one call.

    Existing per-artefact CSVs are unchanged; this file is the analysis
    convenience entry point, not a replacement.
    """
    # The shared scalars below are by construction identical between the GMM
    # and HMM analyses (run_asset passes the same window_scale/rolling_days/
    # event_window/calm_window/n_components to both). Assert this so a future
    # divergence is loud rather than silently writing the GMM value into a
    # field that purports to describe both.
    for shared_key in (
        "n_components", "window_scale", "rolling_days",
        "event_window", "calm_window",
    ):
        if gmm_analysis.get(shared_key) != hmm_analysis.get(shared_key):
            raise ValueError(
                f"GMM/HMM divergence on shared key {shared_key!r}: "
                f"gmm={gmm_analysis.get(shared_key)} hmm={hmm_analysis.get(shared_key)}"
            )
    metadata = {
        "symbol": gmm_analysis.get("symbol"),
        "stem": stem,
        "source_file": f"{stem}_5m.csv",
        "n_components": gmm_analysis.get("n_components"),
        "window_scale": gmm_analysis.get("window_scale"),
        "rolling_days": gmm_analysis.get("rolling_days"),
        "event_window": gmm_analysis.get("event_window"),
        "calm_window": gmm_analysis.get("calm_window"),
    }
    payload = {
        "metadata": _coerce_for_json(metadata),
        "gmm": _coerce_for_json(gmm_analysis),
        "hmm": _coerce_for_json(hmm_analysis),
        "calendar_window_gmm": _coerce_for_json(cal_window_gmm),
        "calendar_window_hmm": _coerce_for_json(cal_window_hmm),
    }
    payload["metadata"]["schema_version"] = "1.0"
    path = outputs_dir / f"{stem}_5m_results.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(path)
    return path


def _save_baseline_csvs(
    analysis: dict[str, Any], outputs_dir: Path, stem: str, prefix: str = "",
) -> Path:
    """Save baseline ARI / AMI / VI / TOD / event / calm CSVs. Returns the ARI path."""
    ari_path = outputs_dir / f"{stem}_{prefix}cross_freq_ari.csv"
    if not analysis["ari_matrix"].empty:
        analysis["ari_matrix"].to_csv(ari_path)
    if not analysis["ami_matrix"].empty:
        analysis["ami_matrix"].to_csv(outputs_dir / f"{stem}_{prefix}cross_freq_ami.csv")
    if not analysis["vi_matrix"].empty:
        analysis["vi_matrix"].to_csv(outputs_dir / f"{stem}_{prefix}cross_freq_vi.csv")
    if (
        isinstance(analysis.get("tod_crisis_distribution"), pd.DataFrame)
        and not analysis["tod_crisis_distribution"].empty
    ):
        analysis["tod_crisis_distribution"].to_csv(
            outputs_dir / f"{stem}_{prefix}tod_crisis_distribution.csv", index=False
        )
    if (
        isinstance(analysis.get("tod_adjusted_ari_matrix"), pd.DataFrame)
        and not analysis["tod_adjusted_ari_matrix"].empty
    ):
        analysis["tod_adjusted_ari_matrix"].to_csv(
            outputs_dir / f"{stem}_{prefix}tod_adjusted_cross_freq_ari.csv"
        )
    if (
        isinstance(analysis.get("event_ari_matrix"), pd.DataFrame)
        and not analysis["event_ari_matrix"].empty
    ):
        analysis["event_ari_matrix"].to_csv(
            outputs_dir / f"{stem}_{prefix}event_cross_freq_ari.csv"
        )
    if (
        isinstance(analysis.get("calm_ari_matrix"), pd.DataFrame)
        and not analysis["calm_ari_matrix"].empty
    ):
        analysis["calm_ari_matrix"].to_csv(
            outputs_dir / f"{stem}_{prefix}calm_cross_freq_ari.csv"
        )
    return ari_path


def _save_fallback_triggers(
    analysis: dict[str, Any],
    hmm_analysis: dict[str, Any],
    outputs_dir: Path,
    stem: str,
) -> None:
    """Persist per-frequency fit status. ``status`` distinguishes the three
    states ``normal / degenerate_skipped / pct_fallback`` so the CSV is not
    ambiguous about whether the model ran at all."""
    rows: list[dict[str, Any]] = []
    for kind, src in (("gmm", analysis), ("hmm", hmm_analysis)):
        flags = src.get("fallback_flags", {})
        status = src.get("fit_status", {})
        for freq, fb in flags.items():
            rows.append({
                "freq": freq,
                "model": kind,
                "fallback_triggered": fb,
                "status": status.get(freq, "normal"),
            })
    if rows:
        pd.DataFrame(rows).to_csv(
            outputs_dir / f"{stem}_fallback_triggers.csv", index=False
        )


def _save_fit_diagnostics_csv(
    analysis: dict[str, Any], outputs_dir: Path, stem: str, model_tag: str = "gmm",
) -> None:
    """Flatten ``fit_diagnostics`` dict to a per-frequency long table.

    ``model_tag`` ("gmm" or "hmm") appears in the filename:
    ``{stem}_{model_tag}_diagnostics.csv``.
    """
    diag_dict = analysis.get("fit_diagnostics") or {}
    if not diag_dict:
        return
    rows: list[dict[str, Any]] = []
    for freq, diag in diag_dict.items():
        scalar = {k: v for k, v in diag.items() if k not in ("means", "stds", "weights")}
        row: dict[str, Any] = {"freq": freq, **scalar}
        SINGULAR = {"means": "mean", "stds": "std", "weights": "weight"}
        for key in ("means", "stds", "weights"):
            singular = SINGULAR[key]
            for ci, val in enumerate(diag.get(key, [])):
                row[f"{singular}_{ci}"] = val
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        outputs_dir / f"{stem}_{model_tag}_diagnostics.csv", index=False,
    )


def _save_expanding_csv(
    analysis: dict[str, Any], outputs_dir: Path, stem: str, prefix: str = "",
) -> None:
    """Persist the expanding-window cross-frequency ARI matrix when available."""
    matrix = analysis.get("expanding_ari_matrix")
    if isinstance(matrix, pd.DataFrame) and not matrix.empty:
        matrix.to_csv(outputs_dir / f"{stem}_{prefix}expanding_cross_freq_ari.csv")


def _save_calendar_window_csv(
    cal_window_result: dict[str, Any], outputs_dir: Path, stem: str, prefix: str = "",
) -> None:
    """Persist the calendar-window (intraday-only) ARI matrix."""
    matrix = cal_window_result.get("ari_matrix")
    if isinstance(matrix, pd.DataFrame) and not matrix.empty:
        matrix.to_csv(outputs_dir / f"{stem}_{prefix}calendar_window_cross_freq_ari.csv")


def _save_cl_roll_csv(
    analysis: dict[str, Any], outputs_dir: Path, stem: str, prefix: str = "",
) -> None:
    roll = analysis.get("cl_roll_analysis")
    if not roll:
        return
    pd.DataFrame([{
        "roll_week_mean_ari": roll["roll_week_mean_ari"],
        "nonroll_week_mean_ari": roll["nonroll_week_mean_ari"],
        "roll_week_bars": roll["roll_week_bars"],
        "nonroll_week_bars": roll["nonroll_week_bars"],
    }]).to_csv(outputs_dir / f"{stem}_{prefix}roll_week_ari.csv", index=False)


def _save_daily_rolling_csvs(
    analysis: dict[str, Any], outputs_dir: Path, stem: str, prefix: str = "",
) -> tuple[Path, Path]:
    """Save daily / rolling CSVs and the rolling-ARI plot. Returns (daily, rolling) paths."""
    daily_summary_path = outputs_dir / f"{stem}_{prefix}daily_summary.csv"
    rolling_path = outputs_dir / f"{stem}_{prefix}rolling_7d_ari.csv"
    if isinstance(analysis.get("daily_df"), pd.DataFrame) and not analysis["daily_df"].empty:
        analysis["daily_df"].to_csv(daily_summary_path, index=False)
    if isinstance(analysis.get("daily_pair_df"), pd.DataFrame) and not analysis["daily_pair_df"].empty:
        analysis["daily_pair_df"].to_csv(
            outputs_dir / f"{stem}_{prefix}daily_pairwise_ari.csv", index=False,
        )
    if isinstance(analysis.get("rolling_df"), pd.DataFrame) and not analysis["rolling_df"].empty:
        analysis["rolling_df"].to_csv(rolling_path, index=False)
    if isinstance(analysis.get("rolling_pair_df"), pd.DataFrame) and not analysis["rolling_pair_df"].empty:
        analysis["rolling_pair_df"].to_csv(
            outputs_dir / f"{stem}_{prefix}rolling_7d_pairwise_ari.csv", index=False,
        )
    _plot_rolling_ari(
        analysis["rolling_df"],
        analysis["rolling_pair_df"],
        stem,
        outputs_dir / f"{stem}_{prefix}rolling_7d_ari.png",
    )
    return daily_summary_path, rolling_path
