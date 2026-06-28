"""
CLI dispatcher for the ``pipeline`` command.

Responsibilities are limited to: load YAML config, ensure 5m data is present
(IB download / freshness check / validation), then delegate to the actual
multi-frequency pipeline in :mod:`src.workflows.pipeline`. This module is
*not* the pipeline itself; it is the CLI entry point that wraps it.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..data.data_ib import canonical_stem, download_tickers_ib_5m, load_5m_ohlc, validate_raw_5m_dir
from ..core.runtime import configure_global_file_logging
from ..workflows.pipeline import run as run_pipeline

logger = logging.getLogger(__name__)

def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_date_like(s: Any) -> pd.Timestamp | None:
    if s is None:
        return None
    try:
        ts = pd.Timestamp(s)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    return ts


def _target_end_ts(ib_cfg: dict[str, Any]) -> pd.Timestamp:
    """
    Resolve the effective end timestamp we consider "up to date".
    If ib.end_date is null, use today (NY time), end-of-day.
    """
    end_cfg = ib_cfg.get("end_date")
    ts = _parse_date_like(end_cfg) if end_cfg is not None else None
    if ts is None:
        now = pd.Timestamp.now(tz="America/New_York")
        ts = now.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts


def _file_last_ts(path: Path) -> pd.Timestamp | None:
    """Return the most recent timestamp in the 5m CSV at ``path``.

    Delegates to :func:`load_5m_ohlc` so freshness checks see the same
    parsing rules (UTF-8 BOM stripping, mixed-DST offset coercion via
    ``utc=True``) as the pipeline; otherwise the parallel reader could
    silently disagree on which file is "stale".
    """
    try:
        df = load_5m_ohlc(path)
    except Exception:
        return None
    if df.empty:
        return None
    ts = pd.Timestamp(df.index.max())
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    return ts


def run(config_path: Path, overrides: dict[str, Any] | None = None) -> None:
    t0 = time.perf_counter()
    config_path = Path(config_path).resolve()
    project_dir = config_path.parent
    with open(config_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    overrides = overrides or {}

    # raw_dir / outputs_dir are anchored to the config's directory when given
    # as relative paths, so the pipeline behaves the same regardless of the
    # caller's cwd (e.g., debugger launches from a parent directory).
    def _anchor(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (project_dir / path)
    raw_dir = _anchor(overrides.get("raw_dir") or cfg.get("raw_dir", "data"))
    outputs_dir = _anchor(overrides.get("outputs_dir") or cfg.get("outputs_dir", "outputs"))
    # `--symbols` (CLI) and `assets`/`symbols` overrides replace the config
    # asset list wholesale rather than merging: the user's intent in passing
    # `--symbols SPY` is "run only SPY", not "add SPY on top of the config
    # default". Merging would surprise CLI callers who expect their explicit
    # list to be authoritative.
    assets: list[str] = overrides.get("assets") or overrides.get("symbols") or cfg.get("assets", ["SPY", "USDJPY", "CL"])
    ib_cfg: dict[str, Any] = {**(cfg.get("ib") or {}), **{k: v for k, v in overrides.items() if v is not None and k in ("host", "port", "client_id", "start_date", "end_date", "future_expiry_by_symbol")}}
    ensure_cfg: dict[str, Any] = cfg.get("ensure_data") or {}

    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    configure_global_file_logging(outputs_dir / "run.log")
    logger.info(
        "Run started; config=%s raw_dir=%s outputs_dir=%s assets=%s",
        config_path.name,
        raw_dir,
        outputs_dir,
        assets,
    )

    mode_cfg = str(ensure_cfg.get("mode") or "auto").strip().lower()
    download_only = bool(overrides.get("download_only", False)) or mode_cfg in {"download_only", "download-only", "download"}
    validate_only = bool(overrides.get("validate_only", False)) or mode_cfg in {"validate_only", "validate-only", "validate"}
    if download_only and validate_only:
        raise ValueError("cli_main: --download and --validate are mutually exclusive")
    to_download = overrides.get("download_symbols")
    if to_download is not None:
        missing = list(to_download)
    elif download_only:
        missing = list(assets)
    else:
        download_missing = _as_bool(ensure_cfg.get("download_missing"), default=True)
        refresh_if_outdated = _as_bool(ensure_cfg.get("refresh_if_outdated"), default=True)
        slack_days = int(ensure_cfg.get("outdated_slack_days", 1) or 0)
        slack = pd.Timedelta(days=max(0, slack_days))
        end_target = _target_end_ts(ib_cfg)

        candidates: list[str] = []
        for a in assets:
            path = raw_dir / f"{canonical_stem(a)}_5m.csv"
            if not path.exists():
                if download_missing:
                    candidates.append(a)
                continue
            if not refresh_if_outdated:
                continue
            last_ts = _file_last_ts(path)
            if last_ts is None:
                candidates.append(a)
                continue
            # Compare on date level so end-of-day vs midnight differences
            # between end_target and the file's last-bar timestamp do not
            # spuriously mark up-to-date files as stale.
            last_date = last_ts.normalize()
            target_date = end_target.normalize() - slack
            if last_date < target_date:
                candidates.append(a)
        missing = candidates

    if missing and not validate_only:
        client_id_cfg = ib_cfg.get("client_id")
        # PID-based default: every running process has a distinct PID, so
        # parallel pipeline runs cannot collide on clientId. IB caps clientId
        # at 2^31-1, which comfortably fits any OS-level PID.
        client_id = int(client_id_cfg) if client_id_cfg is not None else os.getpid()
        logger.info(
            "Download phase: tickers=%s clientId=%s start=%s end=%s",
            missing,
            client_id,
            ib_cfg.get("start_date"),
            ib_cfg.get("end_date"),
        )
        download_failures: list[str] = []
        try:
            download_tickers_ib_5m(
                tickers=missing,
                output_dir=raw_dir,
                start_date=ib_cfg.get("start_date", "2026-01-01"),
                end_date=ib_cfg.get("end_date"),
                host=ib_cfg.get("host", "127.0.0.1"),
                port=int(ib_cfg.get("port", 4002)),
                client_id=client_id,
                future_expiry_by_symbol=ib_cfg.get("future_expiry_by_symbol") or {},
            )
            logger.info("Download phase complete.")
        except Exception as e:
            download_failures = list(missing)
            logger.error("Download failed: %s. Continuing to validate existing files.", e)
        if download_failures:
            logger.error(
                "DOWNLOAD FAILURES for: %s. Pipeline will use whatever data is on disk.",
                download_failures,
            )
    elif not missing:
        logger.info("All assets have 5m data in %s; skipping download.", raw_dir)

    logger.info("Validation phase: checking 5m data in %s", raw_dir)
    validation = validate_raw_5m_dir(raw_dir, assets)
    has_issues = False
    for symbol, issues in validation.items():
        if issues:
            has_issues = True
            for msg in issues:
                logger.warning("  %s", msg)
    if not has_issues:
        logger.info("Validation passed for all assets.")
    else:
        logger.warning("Validation reported issues for one or more assets.")

    if download_only or validate_only:
        elapsed = time.perf_counter() - t0
        logger.info("Run complete (download_only=%s validate_only=%s); elapsed_s=%.1f", download_only, validate_only, elapsed)
        return

    # Multi-frequency pipeline (Paper 2: cross-freq ARI + timeline).
    # Use explicit episode selection from config/overrides.
    # Avoid implicit inference from dates to prevent window mismatch.
    episode = overrides.get("episode") or cfg.get("episode")
    if episode is not None:
        episode = str(episode).strip()
        if not episode:
            episode = None
    run_pipeline(
        raw_dir=raw_dir,
        outputs_dir=outputs_dir,
        assets=assets,
        episode=episode,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Run complete; total_elapsed_s=%.1f", elapsed)
