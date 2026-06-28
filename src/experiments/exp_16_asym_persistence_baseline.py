"""Asymmetric-persistence calibrated baseline (referee M2).

The default Markov-Gaussian baseline in ``exp_04`` uses the calibrated
transition probabilities from a real ML fit on SPY 1h returns. Empirical
crisis shares span ~16% (USD/JPY 1h) to ~91% (GLD 1d), so a calibration
mismatch on the stationary crisis share is a plausible source of the
empirical-vs-baseline gap.

This experiment recalibrates the DGP per asset so that the simulation's
stationary crisis share matches the empirical 1h crisis share, while
keeping the total transition rate ``tau = P_12 + P_21`` fixed at the
calibrated tau (read from ``outputs/calibrated_ms_params.json`` at the
5m scale). It then reports the asymmetric-baseline mean off-diagonal ARI
through the same canonical pipeline as ``exp_04``, so the resulting ARI
is directly comparable to the empirical and symmetric-baseline numbers.

Public entry point: ``run_asym_baseline(target_dir, ...)``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import project_layout
from ..core.config import FREQS
from ..core.models import fit_regimes_per_frequency
from ..core.sim_dgp import aggregate_reps
from ..data.data_ib import canonical_stem, load_5m_ohlc
from .exp_04_simulation_calibration import ensure_calibration as _ensure_calibration

logger = logging.getLogger(__name__)


# Per-asset stationary crisis shares are computed at runtime from raw
# data via :func:`compute_panel_crisis_shares` rather than hardcoded.
# Replication-affecting fix (P0-S3): the previous implementation pinned
# constants from one specific pipeline run ("ASSET_PI_1H_2026 = {SPY:
# 0.705, ...}"), so any subsequent change to features.py / clip_pct /
# the regime fitter silently broke the calibration target without
# touching this file. The new asymmetric DGP shares now track whatever
# the canonical fitter produces on the current raw data.
#
# Caching: the native-bar shares used here are not the same as the
# 5m-bar-aligned crisis shares persisted in ``<stem>_5m_results.json``
# (the alignment forward-fills coarser-freq labels onto the 5m grid,
# producing slightly different bar-weighted means; SPY 1h native 70.04%
# vs aligned 70.64% in the 2026 panel). Reading the aligned values
# would silently shift the calibration target by up to ~0.6 percentage
# points, so we instead cache the native fit result on disk and
# invalidate by raw-data mtime. See ``_NATIVE_PI_CACHE_NAME``.


_NATIVE_PI_CACHE_NAME = "asym_baseline_native_pi.json"


def _native_pi_cache_path(outputs_dir: Path, freq: str) -> Path:
    return Path(outputs_dir) / f"{_NATIVE_PI_CACHE_NAME}.{freq}"


def _load_cached_native_pi(
    cache_path: Path,
    raw_dir: Path,
    assets: Iterable[str],
) -> dict[str, float] | None:
    """Return cached shares iff cache is fresher than every required raw CSV."""
    if not cache_path.exists():
        return None
    cache_mtime = cache_path.stat().st_mtime
    needed_stems: list[str] = []
    for symbol in assets:
        stem = canonical_stem(symbol)
        needed_stems.append(stem)
        raw = Path(raw_dir) / f"{stem}_5m.csv"
        if raw.exists() and raw.stat().st_mtime > cache_mtime:
            logger.info(
                "compute_panel_crisis_shares: cache %s is stale vs %s; refitting",
                cache_path.name, raw.name,
            )
            return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("compute_panel_crisis_shares: cache %s unreadable (%s); refitting", cache_path, exc)
        return None
    out: dict[str, float] = {}
    for stem in needed_stems:
        if stem not in cached:
            logger.info(
                "compute_panel_crisis_shares: cache %s missing stem %s; refitting",
                cache_path.name, stem,
            )
            return None
        out[stem] = float(cached[stem])
    return out


def compute_panel_crisis_shares(
    raw_dir: Path,
    assets: Iterable[str],
    freq: str,
    outputs_dir: Path | None = None,
) -> dict[str, float]:
    """Per-asset stationary crisis share at ``freq`` on the current raw data.

    Loads each asset's 5m CSV, runs the canonical
    :func:`fit_regimes_per_frequency` recipe (matching what the main
    pipeline does), and returns the empirical share of crisis-labelled
    bars at the requested frequency. Used by :func:`run_asym_baseline`
    as the calibration target for the asymmetric-persistence DGP.

    When ``outputs_dir`` is supplied, results are cached on disk in
    ``<outputs_dir>/asym_baseline_native_pi.json.<freq>`` and reused on
    subsequent calls iff every requested asset's raw CSV is older than
    the cache file. The cache stores native-bar shares (matching the
    fit performed here), so it is bit-stable with the no-cache path.
    Caching avoids the ~3-second-per-asset refit cost in the main
    ``extended_asym_baseline`` workflow without changing the
    calibration target.

    Each share is clipped to ``(0.01, 0.99)`` so that the downstream
    :func:`_pi_to_transitions` conversion never receives a degenerate
    boundary value (which would make the implied chain absorbing).
    """
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {FREQS}, got {freq!r}")
    asset_list = list(assets)

    if outputs_dir is not None:
        cache_path = _native_pi_cache_path(outputs_dir, freq)
        cached = _load_cached_native_pi(cache_path, raw_dir, asset_list)
        if cached is not None:
            logger.info(
                "compute_panel_crisis_shares: loaded %d cached %s shares from %s",
                len(cached), freq, cache_path.name,
            )
            return cached

    out: dict[str, float] = {}
    for symbol in asset_list:
        stem = canonical_stem(symbol)
        path = Path(raw_dir) / f"{stem}_5m.csv"
        if not path.exists():
            logger.warning("compute_panel_crisis_shares: %s missing; skipping", path)
            continue
        df_5m = load_5m_ohlc(path)
        regimes = fit_regimes_per_frequency(df_5m, stem, FREQS)
        labels = regimes.get(freq, pd.Series(dtype=float)).dropna()
        if labels.empty:
            logger.warning(
                "compute_panel_crisis_shares: no %s labels for %s; skipping",
                freq, stem,
            )
            continue
        share = float((labels == 1).mean())
        share = max(min(share, 0.99), 0.01)
        out[stem] = share

    if outputs_dir is not None and out:
        cache_path = _native_pi_cache_path(outputs_dir, freq)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, sort_keys=True)
            logger.info(
                "compute_panel_crisis_shares: wrote %d %s shares to %s",
                len(out), freq, cache_path.name,
            )
        except OSError as exc:
            logger.warning("compute_panel_crisis_shares: failed to write cache %s (%s)", cache_path, exc)

    return out


def _pi_to_transitions(pi_crisis: float, tau: float) -> tuple[float, float]:
    """Convert (stationary crisis share, total transition rate) to (P_12, P_21).

    Stationary distribution of a 2-state Markov chain:
        pi_crisis = P_12 / (P_12 + P_21) = P_12 / tau
    so P_12 = pi * tau, P_21 = (1 - pi) * tau.
    """
    if not (0 < pi_crisis < 1):
        raise ValueError(f"pi_crisis must be in (0, 1), got {pi_crisis}")
    max_tau = min(1.0 / pi_crisis, 1.0 / (1.0 - pi_crisis))
    if not (0 < tau < max_tau):
        raise ValueError(
            f"tau must be in (0, {max_tau:.4f}) for pi_crisis={pi_crisis}, got {tau}"
        )
    p12 = pi_crisis * tau
    p21 = (1.0 - pi_crisis) * tau
    return float(p12), float(p21)


def run_asym_baseline(
    target_dir: Path,
    out_filename: str,
    n_reps: int = 200,
    asset_pi: dict[str, float] | None = None,
    tau: float | None = None,
    seed: int = 42,
    raw_dir: Path | None = None,
    pi_freq: str = "1h",
    assets: Iterable[str] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the per-asset asymmetric-persistence calibrated baseline.

    For each asset, ``P_12`` and ``P_21`` are set so the stationary
    distribution matches the supplied per-asset crisis share at total
    transition rate ``tau``. When ``asset_pi`` is ``None`` the shares
    are computed at runtime from ``raw_dir`` at the ``pi_freq``
    frequency (canonical pipeline) so the calibration target tracks
    whatever the current fitter produces on the current data. Pass an
    explicit dict to pin shares to a specific previously-published
    table.

    ``tau`` defaults to the calibrated 5m tau ``P_12 + P_21`` from
    ``outputs/calibrated_ms_params.json``.

    All assets share the same ``seed``, using Common Random Numbers
    (CRN) for variance-reduced cross-asset comparison. Vary ``seed``
    per call for statistically independent per-asset simulations.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    cal = _ensure_calibration(target_dir, raw_dir=raw_dir)
    if asset_pi is None:
        if raw_dir is None or assets is None:
            raise ValueError(
                "run_asym_baseline: when asset_pi is None, both raw_dir "
                "and assets must be supplied so per-asset crisis shares "
                "can be computed from the current data."
            )
        asset_pi = compute_panel_crisis_shares(
            raw_dir, assets, pi_freq, outputs_dir=cache_dir,
        )
        logger.info(
            "exp_16: per-asset %s crisis shares (computed from %s): %s",
            pi_freq, raw_dir,
            {k: round(v, 3) for k, v in asset_pi.items()},
        )
    if tau is None:
        tau = float(cal.P_12 + cal.P_21)
        logger.info("Using calibrated 5m tau = P_12 + P_21 = %.4f", tau)

    rows: list[dict[str, object]] = []
    for asset, pi in asset_pi.items():
        p12, p21 = _pi_to_transitions(pi, tau)
        agg = aggregate_reps(
            cal, n_reps=n_reps, n_components=2,
            P_12_override=p12, P_21_override=p21, seed=seed,
        )
        rows.append({
            "asset": asset,
            "pi_crisis": pi,
            "tau": tau,
            "P_12": p12,
            "P_21": p21,
            **agg,
        })
        logger.info(
            "%s: pi=%.3f, P_12=%.5f P_21=%.5f -> alt_mean_all4=%.4f "
            "[q25 %.4f, q75 %.4f]",
            asset, pi, p12, p21,
            rows[-1]["alt_mean_all4"], rows[-1]["alt_q25_all4"],
            rows[-1]["alt_q75_all4"],
        )

    df = pd.DataFrame(rows)
    out_path = target_dir / out_filename
    df.to_csv(out_path, index=False)
    logger.info("Saved asymmetric-persistence baseline: %s", out_path)
    return df


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Asymmetric-persistence calibrated baseline (canonical pipeline)")

    logger.info("Sweep 1/2: pi calibrated to 1h crisis shares (computed at runtime)")
    run_asym_baseline(
        layout.outputs_dir,
        n_reps=200,
        seed=42,
        raw_dir=layout.raw_dir,
        pi_freq="1h",
        assets=layout.assets,
        cache_dir=layout.outputs_dir,
        out_filename="simulation_rss_asym_persistence_1h_anchor.csv",
    )

    logger.info("Sweep 2/2: pi calibrated to 1d crisis shares (computed at runtime)")
    run_asym_baseline(
        layout.outputs_dir,
        n_reps=200,
        seed=43,
        raw_dir=layout.raw_dir,
        pi_freq="1d",
        assets=layout.assets,
        cache_dir=layout.outputs_dir,
        out_filename="simulation_rss_asym_persistence_1d_anchor.csv",
    )


if __name__ == "__main__":
    main()
