"""Shared constants for the multi-frequency regime pipeline.

Centralised here so that experiment modules and notebooks can import the
canonical values without dragging the full pipeline import surface, and so
that future experiments can override per-call (e.g., a sweep over alternative
event windows) without monkey-patching ``workflows.pipeline``.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Time and resolution
# ---------------------------------------------------------------------------

TZ: Final[str] = "America/New_York"
FREQS: Final[tuple[str, str, str, str]] = ("5m", "15m", "1h", "1d")

# Approximate bar count per trading day per asset, used as a sanity guide
# when validating downloaded 5m data (not as a hard threshold).
BARS_PER_DAY: Final[dict[str, int]] = {
    "SPY": 78,
    "SPX": 78,
    "CL": 276,
    "USDJPY": 288,
}

# Asset stems for which the contract-roll-week robustness analysis runs.
# Currently CL only (continuous front-month future with monthly roll); add
# stems here when extending the roll-week diagnostic to other futures.
CL_ROLL_ASSETS: Final[frozenset[str]] = frozenset({"CL"})


# ---------------------------------------------------------------------------
# GMM / HMM defaults
# ---------------------------------------------------------------------------

DEFAULT_GMM_K: Final[int] = 2
MIN_TRADING_DAYS: Final[int] = 5
DEFAULT_ROLLING_DAYS: Final[int] = 7
DEFAULT_WINDOW_SCALE: Final[float] = 1.0

MODEL_GMM: Final[str] = "gmm"
MODEL_HMM: Final[str] = "hmm"


# ---------------------------------------------------------------------------
# Permutation / bootstrap protocol
# ---------------------------------------------------------------------------
# Centralised so the pipeline, exp_14_block_sweep_gld, and any future
# perm-test consumers cite the same (n_perm, block_size, seed). Changing
# these is a paper-replication-affecting decision; keep them here.

DEFAULT_PERM_N: Final[int] = 500
DEFAULT_PERM_SEED: Final[int] = 42
DEFAULT_BLOCK_SIZE: Final[int] = 50  # 50 5m bars ~= 4h


# ---------------------------------------------------------------------------
# Expanding-window warm-up
# ---------------------------------------------------------------------------
# Adaptive minimum-training-bars for the no-look-ahead expanding GMM/HMM fit.
# Floor 20 / ceiling 200 / asset-fraction 0.35 / per-state-floor 6 are
# chosen so the 1d series (n=124) does not collapse to all-NaN expanding
# labels while still leaving enough OOS bars for ARI computation; values
# determined empirically on the 2026 four-asset panel.

EXPANDING_MIN_TRAIN_FLOOR: Final[int] = 20
EXPANDING_MIN_TRAIN_CEIL: Final[int] = 200
EXPANDING_MIN_TRAIN_FRACTION: Final[float] = 0.35
EXPANDING_MIN_TRAIN_PER_STATE: Final[int] = 6
# Refit cadence: target ~EXPANDING_REFIT_DENOM evenly-spaced refits per series.
EXPANDING_REFIT_DENOM: Final[int] = 50


# ---------------------------------------------------------------------------
# Regime-fit fallback thresholds
# ---------------------------------------------------------------------------
# Replication-affecting: when the GMM/HMM split is trivial (one component
# absorbs <TRIVIAL_SPLIT_LOWER_PCT% or >TRIVIAL_SPLIT_UPPER_PCT% of bars),
# regime labels fall back to a hard PCT_FALLBACK_PERCENTILE-th percentile
# log-vol threshold. Centralised here because changing any of the three
# alters which cells are flagged as degenerate in Suppl Table tab:gmm_diag.

PCT_FALLBACK_PERCENTILE: Final[float] = 80.0
TRIVIAL_SPLIT_LOWER_PCT: Final[float] = 1.0
TRIVIAL_SPLIT_UPPER_PCT: Final[float] = 99.0


# ---------------------------------------------------------------------------
# Episode windows
# ---------------------------------------------------------------------------

# 2026 US-Iran escalation windows (NY calendar dates, inclusive).
# Real event timeline:
#   * Jan 8  Iran internet cutoff during domestic-protest crackdown
#   * Jan 20-24  diplomatic-tension period (low realised vol)
#   * Jan 28  "massive Armada heading to Iran" announcement, military buildup
#   * Feb 3  IRGC intercepts US tankers in the Strait of Hormuz; F-35 shootdown
#   * Feb 28  joint US + Israeli strike on Iran (the actual stress event)
#   * Mar 3   VIX intraday +31% to 28.15 ("close +10%"), oil +13%
#   * post-Mar  VIX peak >35, KOSPI single-day -12% circuit-breaker
#   * Late Mar / Apr  ceasefire, VIX returns to pre-event levels
#
# EVENT_WINDOW captures the actual stress arc end-to-end: from the joint
# US-Israeli strike on 28 Feb through the early-April ceasefire, ~6.5 trading
# weeks.  CALM_WINDOW is the pre-escalation baseline 1 Jan -- 24 Jan: data
# is in-sample (avoids the gap that a Nov-Dec 2025 reference would create),
# but precedes the 28 Jan "Armada" announcement that kicked off the visible
# market response, so realised vol is low (the 8 Jan internet cutoff and
# 20-24 Jan diplomatic-tension period did not materially move markets).
# This is a descriptive event/reference contrast, not a formal event-study
# abnormal-return calculation; for an FRL letter the cross-frequency ARI
# panel + permutation null + rolling-7d trace are the inferential statistics.
# Episode registry: maps episode name -> (event_window, calm_window).
# Tuple-of-tuples shape preserved for backward compatibility with the
# existing pipeline orchestration code.  2022 windows unchanged: stress is
# the invasion week, calm is mid-Jan ~6 weeks ahead of invasion.
EPISODES: Final[dict[str, tuple[tuple[str, str], tuple[str, str]]]] = {
    "2026_iran": (("2026-02-28", "2026-04-15"), ("2026-01-01", "2026-01-24")),
    "2022_ukraine": (("2022-02-22", "2022-02-28"), ("2022-01-10", "2022-01-14")),
}

# Single source of truth: derive EVENT_WINDOW / CALM_WINDOW from the
# episode registry rather than re-declaring the same string literals.
# Editing EPISODES["2026_iran"] now propagates everywhere.
EVENT_WINDOW: Final[tuple[str, str]] = EPISODES["2026_iran"][0]
CALM_WINDOW: Final[tuple[str, str]] = EPISODES["2026_iran"][1]


# ---------------------------------------------------------------------------
# Resolution-dissonance monitoring threshold
# ---------------------------------------------------------------------------
# Indicative fail-safe ARI level used in the cross-asset resonance figure
# (src/visualization/cross_asset_resonance.py).  This is a visualisation
# guideline, not a validated decision threshold; see Suppl S.3 (practical
# protocol) for the recommended asset-specific calibration approach.
CROSS_ASSET_RESONANCE_FAILSAFE_ARI: Final[float] = 0.1
