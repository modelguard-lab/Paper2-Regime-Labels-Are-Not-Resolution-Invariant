"""Canonical-pipeline simulator for the calibrated MS-Gaussian DGP.

Replaces the previously-bespoke ``np.repeat``-upsampled GMM-on-RSS
pipeline used by ``exp_04 / exp_11 / exp_13 / exp_16`` with a single
function that

1. generates a synthetic 5m return path under the calibrated 2-state
   Markov-switching DGP (``simulate_ms_returns_5m``),
2. builds a synthetic OHLC frame indexed on a real RTH 5m DatetimeIndex
   (``synthetic_ohlc_5m``),
3. resamples to 5m / 15m / 1h / 1d through the canonical
   ``resample_ohlc`` and runs the canonical
   ``fit_regimes_per_frequency`` + ``cross_freq_ari_matrix`` to compute
   the same ``overall_mean_ari`` field that the empirical pipeline
   produces for SPY, USDJPY, CL, and GLD.

The shared helper ``run_one_sim_replication`` returns two scalars per
replication (alt mean off-diag ARI and a null-distribution mean ARI),
so each calibration experiment becomes a thin wrapper that aggregates
``n_reps`` calls and writes its own CSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_BLOCK_SIZE, FREQS, MODEL_GMM, TZ
from .calibration import CalibrationAt5m, CalibratedGarchParams
from .features import resample_ohlc
from .metrics import cross_freq_ari_matrix, mean_offdiag_ari
from .models import align_regimes_to_5m, fit_regimes_per_frequency

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic 5m DGP
# ---------------------------------------------------------------------------


# Default: 80 trading days (~16 weeks) of RTH 5m bars. With 78 bars per RTH
# day this gives ~6240 bars, comfortably above the ~30-day pipeline floor at
# the daily frequency and within an order of magnitude of the 6-month
# empirical sample. Trade-off: larger n is more accurate per replication but
# scales the per-rep cost linearly through the GMM fit.
DEFAULT_N_DAYS_5M: int = 80
DEFAULT_BARS_PER_DAY_RTH: int = 78


def make_rth_5m_index(
    n_days: int = DEFAULT_N_DAYS_5M,
    bars_per_day: int = DEFAULT_BARS_PER_DAY_RTH,
    start: str = "2024-01-02",
) -> pd.DatetimeIndex:
    """Build an RTH-only NY-tz 5m DatetimeIndex spanning ``n_days`` sessions.

    Each session runs from 09:35 ET (first 5m bar end) to 16:00 ET, which
    is 78 bars at 5-minute spacing. Skips weekends and US federal market
    holidays via ``CustomBusinessDay`` with ``USFederalHolidayCalendar``;
    this aligns the synthetic index with the real RTH 5m grid the
    empirical pipeline operates on.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    start_ts = pd.Timestamp(start, tz=TZ)
    cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    days = pd.date_range(
        start=start_ts.date(), periods=n_days, freq=cbd, tz=TZ
    )
    timestamps: list[pd.Timestamp] = []
    for d in days:
        # First bar end at 09:35 (1 bar of [09:30, 09:35]); last bar end 16:00.
        session_start = d.normalize() + pd.Timedelta(hours=9, minutes=35)
        for k in range(bars_per_day):
            timestamps.append(session_start + pd.Timedelta(minutes=5 * k))
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    return idx


def simulate_ms_returns_5m(
    cal: CalibrationAt5m,
    n_bars: int,
    P_12_override: float | None = None,
    P_21_override: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2-state Markov-switching Gaussian return path at 5m scale.

    Returns
    -------
    (returns, states) : both shape (n_bars,)
        ``returns`` are raw log-returns; ``states`` are 0 / 1.
    """
    if rng is None:
        raise ValueError("rng is required for reproducibility; pass np.random.default_rng(seed)")
    P_12 = float(P_12_override) if P_12_override is not None else cal.P_12
    P_21 = float(P_21_override) if P_21_override is not None else cal.P_21
    P_12 = max(0.0, min(1.0, P_12))
    P_21 = max(0.0, min(1.0, P_21))

    states = np.empty(n_bars, dtype=int)
    # Stationary distribution of a 2-state Markov chain with transition
    # probabilities P(0->1) = P_12, P(1->0) = P_21:
    #
    #     pi_0 = P_21 / (P_12 + P_21)   pi_1 = P_12 / (P_12 + P_21)
    #
    # Drawing the initial state from this distribution removes the
    # ~150-bar transient where uniform initialisation oversamples the
    # less-stationary regime; the simulator becomes stationary from t=0.
    denom = P_12 + P_21
    if denom <= 1e-12:
        raise ValueError(
            f"simulate_ms_returns_5m: degenerate transitions (P_12={P_12}, "
            f"P_21={P_21}); chain is reducible and the simulated path locks "
            f"into its initial regime forever, producing no MS dynamics."
        )
    pi_0 = P_21 / denom
    states[0] = 0 if rng.random() < pi_0 else 1
    for t in range(1, n_bars):
        if states[t - 1] == 0:
            states[t] = 1 if rng.random() < P_12 else 0
        else:
            states[t] = 0 if rng.random() < P_21 else 1
    mu = np.where(states == 0, cal.mu_0, cal.mu_1)
    sigma = np.where(states == 0, cal.sigma_0, cal.sigma_1)
    if (sigma <= 0).any():
        raise ValueError("simulate_ms_returns_5m: sigma must be positive")
    rets = rng.normal(loc=mu, scale=sigma)
    return rets, states


def simulate_ms_garch_returns_5m(
    cal: CalibrationAt5m,
    garch: CalibratedGarchParams,
    n_bars: int,
    P_12_override: float | None = None,
    P_21_override: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 5m MS-GARCH(1,1) return path under the calibrated DGP.

    Within regime k, sigma_t^2 = omega[k] + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2,
    with eps_t = sigma_t * z_t, z_t ~ N(0,1). ``alpha`` and ``beta`` come
    from a single-regime GARCH(1,1) fit on SPY 5m returns. Per-regime
    omega is set so the within-regime unconditional variance matches the
    MS-calibrated sigma_k^2:

        omega[k] = sigma_k^2 * (1 - alpha - beta)

    Mean returns ``mu[k]`` come from the MS calibration.

    Cross-regime memory handling
    ----------------------------
    A naive recursion ``sigma2[t] = omega[k_t] + alpha*eps_{t-1}^2 +
    beta*sigma2[t-1]`` carries variance memory from the OLD regime
    across a switch. With the calibrated ``alpha+beta = 0.896`` and a
    crisis half-life of ~47 bars, this contaminates the realised
    within-regime variance and breaks the matched-moments calibration
    that motivates ``omega[k] = sigma_k^2 * (1 - alpha - beta)``.

    Fix (Option A in the review): on every regime change, reset the
    recursion's lagged values to the new regime's stationary state
    (``sigma2[t-1] = sigma2_regime[k_t]``, ``eps[t-1] = 0``) BEFORE
    advancing the GARCH update at time t. This makes the within-regime
    GARCH process draw from its own stationary distribution at each
    visit, so the cross-frequency ARI comparison against the i.i.d.
    MS-Gauss baseline becomes a matched-moments test rather than an
    upper-bound persistence test.
    """
    if rng is None:
        raise ValueError("rng is required for reproducibility; pass np.random.default_rng(seed)")
    P_12 = float(P_12_override) if P_12_override is not None else cal.P_12
    P_21 = float(P_21_override) if P_21_override is not None else cal.P_21
    P_12 = max(0.0, min(1.0, P_12))
    P_21 = max(0.0, min(1.0, P_21))

    # Stationarity guard. ``>= 0.999`` is a numerical-stability margin
    # (not just exact non-stationarity): with alpha+beta arbitrarily
    # close to 1, sigma2 can drift to overflow on long n_bars before
    # crossing the formal boundary.
    if garch.alpha < 0 or garch.beta < 0:
        raise ValueError("GARCH alpha and beta must be non-negative")
    persist = garch.alpha + garch.beta
    if persist >= 0.999:
        raise ValueError(
            f"GARCH alpha+beta = {persist:.6f} must be < 0.999 for "
            f"numerical stationarity (stricter than the formal < 1.0 "
            f"bound to prevent overflow on long simulations)"
        )

    states = np.empty(n_bars, dtype=int)
    # Stationary init (see simulate_ms_returns_5m for derivation).
    denom = P_12 + P_21
    if denom <= 1e-12:
        raise ValueError(
            f"simulate_ms_garch_returns_5m: degenerate transitions "
            f"(P_12={P_12}, P_21={P_21}); chain is reducible."
        )
    pi_0 = P_21 / denom
    states[0] = 0 if rng.random() < pi_0 else 1
    for t in range(1, n_bars):
        if states[t - 1] == 0:
            states[t] = 1 if rng.random() < P_12 else 0
        else:
            states[t] = 0 if rng.random() < P_21 else 1

    # Per-regime omega so the within-regime stationary variance matches
    # the MS-calibrated sigma_k^2 (in raw log-return units). Use the
    # ``sigma2`` fields directly to avoid sqrt-then-square round-trip.
    sigma2_regime = (float(cal.sigma2_0), float(cal.sigma2_1))
    omega_regime = (
        sigma2_regime[0] * (1.0 - persist),
        sigma2_regime[1] * (1.0 - persist),
    )
    mu_regime = (cal.mu_0, cal.mu_1)

    sigma2 = np.empty(n_bars, dtype=float)
    eps = np.empty(n_bars, dtype=float)
    rets = np.empty(n_bars, dtype=float)
    sigma2[0] = sigma2_regime[states[0]]
    eps[0] = float(np.sqrt(sigma2[0])) * rng.standard_normal()
    rets[0] = mu_regime[states[0]] + eps[0]
    for t in range(1, n_bars):
        # On a regime switch, reset the GARCH state to the new regime's
        # stationary point before advancing the recursion. Draw eps_lag
        # from the new regime's stationary distribution so that
        # E[sigma2[t]] equals sigma2_regime[k] within the new regime
        # rather than (1 - alpha) * sigma2_regime[k] (which biased
        # within-regime variance down).
        if states[t] != states[t - 1]:
            new_var = sigma2_regime[states[t]]
            sigma2_lag = new_var
            eps_lag = float(np.sqrt(new_var)) * float(rng.standard_normal())
        else:
            sigma2_lag = sigma2[t - 1]
            eps_lag = eps[t - 1]
        omega_t = omega_regime[states[t]]
        sigma2[t] = omega_t + garch.alpha * eps_lag ** 2 + garch.beta * sigma2_lag
        eps[t] = float(np.sqrt(sigma2[t])) * rng.standard_normal()
        rets[t] = mu_regime[states[t]] + eps[t]
    # Clip simulated returns per-bar to +/-10 * sigma_t to prevent
    # downstream overflow (e.g., cumsum->exp in synthetic_ohlc_5m) at
    # long n_bars without truncating high-vol tails uniformly.
    np.clip(rets, -10.0 * np.sqrt(sigma2), 10.0 * np.sqrt(sigma2), out=rets)
    return rets, states


def synthetic_ohlc_5m(
    returns_5m: np.ndarray,
    index_5m: pd.DatetimeIndex,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Build a synthetic OHLC 5m frame from a log-return path.

    Each bar is a degenerate OHLC where O=H=L=C is the cumulative-product
    price at that timestamp. The canonical pipeline reads only Close
    (via ``np.log(C_t / C_{t-1})``) for return computation, so degenerate
    HL bars are sufficient. Volume is set to 1 to keep the schema valid
    without affecting any downstream computation.
    """
    if len(returns_5m) != len(index_5m):
        raise ValueError(f"synthetic_ohlc_5m: length mismatch returns={len(returns_5m)} index={len(index_5m)}")
    n = len(returns_5m)
    log_prices = np.log(start_price) + np.cumsum(returns_5m)
    if np.max(np.abs(log_prices)) > 700:
        raise ValueError(
            f"synthetic_ohlc_5m: cumulative log-return exceeds 700 (max={np.max(np.abs(log_prices)):.1f}); "
            "exp would overflow"
        )
    prices = np.exp(log_prices)
    df = pd.DataFrame({
        "Open": prices,
        "High": prices,
        "Low": prices,
        "Close": prices,
        "Volume": np.ones(n, dtype=float),
    }, index=index_5m)
    return df


# ---------------------------------------------------------------------------
# Canonical pipeline ARI on a synthetic OHLC frame
# ---------------------------------------------------------------------------


@dataclass
class SimReplicationResult:
    """One replication's headline statistics, comparable to empirical ARI."""

    overall_mean_ari: float          # 4-freq mean off-diag ARI (5m/15m/1h/1d)
    intraday_mean_ari: float         # 3-freq intraday-only (5m/15m/1h)
    null_mean_ari: float             # null with coarse labels permuted
    n_components: int


def _block_permute_labels(
    labels: np.ndarray,
    rng: np.random.Generator,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray:
    """Circular block permutation matching the canonical pipeline null.

    Rotates the series by a uniform random offset, slices into
    fixed-length blocks, and permutes the block order. Preserves
    within-block autocorrelation while destroying cross-block label
    structure, which is the relevant null for "shared regime structure
    is real beyond block-level coincidence". Falls back to an IID
    permutation if fewer than 2 blocks exist (``n < 2 * block_size``).
    """
    n = len(labels)
    n_blocks = n // block_size
    if n_blocks < 2:
        return rng.permutation(labels)
    shift = int(rng.integers(0, n))
    rotated = np.concatenate([labels[shift:], labels[:shift]])
    blocks = [
        rotated[i * block_size : (i + 1) * block_size]
        for i in range(n_blocks)
    ]
    perm = rng.permutation(len(blocks))
    permuted = np.concatenate([blocks[i] for i in perm])
    # If n is not a multiple of block_size, the trailing remainder is
    # dropped by the block split. Pad with ROTATED (post-shift) values
    # so the tail still moves under the random circular rotation; using
    # the original ``labels[keep:]`` would freeze the last (n % block_size)
    # positions at the observed values across every permutation, biasing
    # the null toward observed at the tail.
    if len(permuted) < n:
        tail = rotated[len(permuted):]
        permuted = np.concatenate([permuted, tail])
    return permuted


def _shuffle_coarse_labels(
    aligned: dict[str, pd.Series],
    rng: np.random.Generator,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, pd.Series]:
    """Return a copy of ``aligned`` with the coarse-frequency labels permuted.

    Matches the canonical pipeline null: keep 5m unchanged, apply a
    circular block permutation of size ``DEFAULT_BLOCK_SIZE`` (=50 5m
    bars ~= 4h) to 15m / 1h / 1d. The earlier IID-permutation null
    destroyed within-regime persistence and produced an artificially
    low null mean ARI not comparable to the canonical Table A.12 null.
    """
    out: dict[str, pd.Series] = {}
    for freq, ser in aligned.items():
        if freq == "5m":
            out[freq] = ser
            continue
        arr = ser.values.copy()
        mask = ~np.isnan(arr)
        # block_size is in 5m-bar units. The aligned series are ffill'd to the
        # 5m grid so non-NaN positions are dense (median_gap == 1) and the
        # block size on the aligned grid is just ``block_size``. The earlier
        # ``block_size / median_gap`` scaling assumed sparse non-NaN positions
        # at the native frequency, but that is not what ``aligned`` carries
        # after ``align_regimes_to_5m``; the rescaled value collapsed to
        # ``block_size`` on every frequency anyway.
        finite_idx = np.flatnonzero(mask)
        if len(finite_idx) < 2:
            out[freq] = pd.Series(arr, index=ser.index, dtype=float)
            continue
        freq_block_size = int(block_size)
        finite = arr[mask].copy()
        shuffled = _block_permute_labels(finite, rng, freq_block_size)
        arr_out = arr.copy()
        arr_out[mask] = shuffled
        out[freq] = pd.Series(arr_out, index=ser.index, dtype=float)
    return out


def run_one_sim_replication(
    cal: CalibrationAt5m,
    rng: np.random.Generator,
    n_components: int = 2,
    P_12_override: float | None = None,
    P_21_override: float | None = None,
    n_days: int = DEFAULT_N_DAYS_5M,
    bars_per_day: int = DEFAULT_BARS_PER_DAY_RTH,
    stem: str = "SIM",
    master_seed: int | None = None,
    rep_idx: int | None = None,
) -> SimReplicationResult:
    """Run one synthetic-DGP replication through the canonical pipeline.

    Steps
    -----
    1. Build an RTH-only 5m DatetimeIndex of length ``n_days * bars_per_day``.
    2. Simulate a 2-state MS-Gaussian path at 5m using the calibrated DGP
       (or the supplied per-rep ``P_12_override / P_21_override``).
    3. Build OHLC at 5m via cumulative-product prices (``synthetic_ohlc_5m``).
    4. Resample to 5m / 15m / 1h / 1d via the canonical ``resample_ohlc``
       and fit regimes per frequency via ``fit_regimes_per_frequency``.
    5. Align labels to the 5m axis, compute the cross-frequency ARI
       matrix, and read the mean off-diagonal ARI.
    6. Permute the coarse labels for the null-distribution sibling.
    """
    index_5m = make_rth_5m_index(n_days=n_days, bars_per_day=bars_per_day)
    n_bars = len(index_5m)
    rets, _ = simulate_ms_returns_5m(
        cal, n_bars,
        P_12_override=P_12_override, P_21_override=P_21_override,
        rng=rng,
    )
    df_5m = synthetic_ohlc_5m(rets, index_5m)

    # Decouple GMM init seed from upstream rng draws via an independent
    # SeedSequence keyed on (master_seed, rep_idx) when those are
    # supplied; otherwise fall back to the legacy rng.integers path so
    # callers that have not adopted the new kwargs still work.
    if master_seed is not None and rep_idx is not None:
        gmm_seed = int(np.random.SeedSequence((master_seed, rep_idx)).generate_state(1)[0])
    else:
        gmm_seed = int(rng.integers(2 ** 31 - 1))
    regimes_by_freq = fit_regimes_per_frequency(
        df_5m, stem, FREQS,
        n_components=int(n_components),
        model=MODEL_GMM,
        seed=gmm_seed,
    )
    aligned = align_regimes_to_5m(regimes_by_freq, df_5m.index)
    ari_df = cross_freq_ari_matrix(aligned, FREQS)
    overall_mean_ari = mean_offdiag_ari(ari_df)

    # Intraday-only: drop 1d row/column from the aligned dict.
    intraday_freqs = ("5m", "15m", "1h")
    aligned_intraday = {f: aligned[f] for f in intraday_freqs}
    intraday_ari_df = cross_freq_ari_matrix(aligned_intraday, intraday_freqs)
    intraday_mean_ari = mean_offdiag_ari(intraday_ari_df)

    # Null distribution: permute coarse labels in the aligned dict.
    null_aligned = _shuffle_coarse_labels(aligned, rng)
    null_ari_df = cross_freq_ari_matrix(null_aligned, FREQS)
    null_mean_ari = mean_offdiag_ari(null_ari_df)

    return SimReplicationResult(
        overall_mean_ari=float(overall_mean_ari) if pd.notna(overall_mean_ari) else float("nan"),
        intraday_mean_ari=float(intraday_mean_ari) if pd.notna(intraday_mean_ari) else float("nan"),
        null_mean_ari=float(null_mean_ari) if pd.notna(null_mean_ari) else float("nan"),
        n_components=int(n_components),
    )


# ---------------------------------------------------------------------------
# Aggregation helper used by every calibration experiment
# ---------------------------------------------------------------------------


def run_one_garch_replication(
    cal: CalibrationAt5m,
    garch: CalibratedGarchParams,
    rng: np.random.Generator,
    n_components: int = 2,
    P_12_override: float | None = None,
    P_21_override: float | None = None,
    n_days: int = DEFAULT_N_DAYS_5M,
    bars_per_day: int = DEFAULT_BARS_PER_DAY_RTH,
    stem: str = "SIM",
    master_seed: int | None = None,
    rep_idx: int | None = None,
) -> SimReplicationResult:
    """MS-GARCH variant of :func:`run_one_sim_replication`.

    Identical recipe except the per-bar return is drawn from the
    calibrated MS-GARCH(1,1) DGP rather than the i.i.d. MS-Gaussian DGP.
    """
    index_5m = make_rth_5m_index(n_days=n_days, bars_per_day=bars_per_day)
    n_bars = len(index_5m)
    rets, _ = simulate_ms_garch_returns_5m(
        cal, garch, n_bars,
        P_12_override=P_12_override, P_21_override=P_21_override,
        rng=rng,
    )
    df_5m = synthetic_ohlc_5m(rets, index_5m)

    if master_seed is not None and rep_idx is not None:
        gmm_seed = int(np.random.SeedSequence((master_seed, rep_idx)).generate_state(1)[0])
    else:
        gmm_seed = int(rng.integers(2 ** 31 - 1))
    regimes_by_freq = fit_regimes_per_frequency(
        df_5m, stem, FREQS,
        n_components=int(n_components),
        model=MODEL_GMM,
        seed=gmm_seed,
    )
    aligned = align_regimes_to_5m(regimes_by_freq, df_5m.index)
    ari_df = cross_freq_ari_matrix(aligned, FREQS)
    overall_mean_ari = mean_offdiag_ari(ari_df)

    intraday_freqs = ("5m", "15m", "1h")
    aligned_intraday = {f: aligned[f] for f in intraday_freqs}
    intraday_ari_df = cross_freq_ari_matrix(aligned_intraday, intraday_freqs)
    intraday_mean_ari = mean_offdiag_ari(intraday_ari_df)

    null_aligned = _shuffle_coarse_labels(aligned, rng)
    null_ari_df = cross_freq_ari_matrix(null_aligned, FREQS)
    null_mean_ari = mean_offdiag_ari(null_ari_df)

    return SimReplicationResult(
        overall_mean_ari=float(overall_mean_ari) if pd.notna(overall_mean_ari) else float("nan"),
        intraday_mean_ari=float(intraday_mean_ari) if pd.notna(intraday_mean_ari) else float("nan"),
        null_mean_ari=float(null_mean_ari) if pd.notna(null_mean_ari) else float("nan"),
        n_components=int(n_components),
    )


def _summarise_replications(
    alt_all4: np.ndarray,
    alt_intra: np.ndarray,
    null_all4: np.ndarray,
) -> dict[str, float]:
    """Compute the summary-stat dict shared by all aggregator variants."""
    aa = alt_all4[np.isfinite(alt_all4)]
    ai = alt_intra[np.isfinite(alt_intra)]
    na = null_all4[np.isfinite(null_all4)]
    return {
        "n_reps_used": int(aa.size),
        "alt_mean_all4": float(np.mean(aa)) if aa.size else float("nan"),
        "alt_q25_all4": float(np.quantile(aa, 0.25)) if aa.size else float("nan"),
        "alt_q75_all4": float(np.quantile(aa, 0.75)) if aa.size else float("nan"),
        "alt_mean_intraday": float(np.mean(ai)) if ai.size else float("nan"),
        "alt_q25_intraday": float(np.quantile(ai, 0.25)) if ai.size else float("nan"),
        "alt_q75_intraday": float(np.quantile(ai, 0.75)) if ai.size else float("nan"),
        "alt_frac_below_0p20_all4": (
            float(np.mean(aa < 0.20)) if aa.size else float("nan")
        ),
        "null_mean_all4": float(np.mean(na)) if na.size else float("nan"),
        "null_q975_all4": (
            float(np.quantile(na, 0.975)) if na.size else float("nan")
        ),
    }


def _aggregate(
    rep_fn,
    n_reps: int,
    seed: int,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Run ``n_reps`` replications via ``rep_fn(rng, master_seed, rep_idx)`` and summarise.

    ``rep_fn`` must accept ``(rng, master_seed, rep_idx)`` and return a
    :class:`SimReplicationResult`. Child seeds are produced via
    :class:`np.random.SeedSequence.spawn` so each replication's RNG is
    statistically independent and the seed sequence is reproducible
    across NumPy versions (no reliance on the master ``integers`` draw
    order).

    Parameters
    ----------
    n_jobs : int, default 1
        Number of parallel worker processes for the replication loop.
        ``1`` runs the loop serially in the calling process (default,
        bit-identical to the legacy implementation). ``-1`` uses every
        available core; any positive integer pins to that worker count.
        Per-replication results are bit-stable across ``n_jobs`` choices
        because each replication's RNG is seeded from
        ``SeedSequence(seed).spawn(n_reps)[i]`` independently of dispatch
        order.
    """
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_reps)

    def _one(i: int):
        rng = np.random.default_rng(child_seeds[i])
        res = rep_fn(rng, seed, i)
        return res.overall_mean_ari, res.intraday_mean_ari, res.null_mean_ari

    if n_jobs == 1:
        results = [_one(i) for i in range(n_reps)]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one)(i) for i in range(n_reps)
        )

    alt_all4 = np.fromiter((r[0] for r in results), dtype=float, count=n_reps)
    alt_intra = np.fromiter((r[1] for r in results), dtype=float, count=n_reps)
    null_all4 = np.fromiter((r[2] for r in results), dtype=float, count=n_reps)
    return _summarise_replications(alt_all4, alt_intra, null_all4)


def aggregate_garch_reps(
    cal: CalibrationAt5m,
    garch: CalibratedGarchParams,
    n_reps: int,
    n_components: int = 2,
    P_12_override: float | None = None,
    P_21_override: float | None = None,
    seed: int = 42,
    n_days: int = DEFAULT_N_DAYS_5M,
    bars_per_day: int = DEFAULT_BARS_PER_DAY_RTH,
    n_jobs: int = 1,
) -> dict[str, float]:
    """MS-GARCH variant of :func:`aggregate_reps`."""
    return _aggregate(
        rep_fn=lambda rng, master_seed, rep_idx: run_one_garch_replication(
            cal, garch, rng,
            n_components=n_components,
            P_12_override=P_12_override,
            P_21_override=P_21_override,
            n_days=n_days,
            bars_per_day=bars_per_day,
            master_seed=master_seed,
            rep_idx=rep_idx,
        ),
        n_reps=n_reps,
        seed=seed,
        n_jobs=n_jobs,
    )


def aggregate_reps(
    cal: CalibrationAt5m,
    n_reps: int,
    n_components: int = 2,
    P_12_override: float | None = None,
    P_21_override: float | None = None,
    seed: int = 42,
    n_days: int = DEFAULT_N_DAYS_5M,
    bars_per_day: int = DEFAULT_BARS_PER_DAY_RTH,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Aggregate ``n_reps`` independent replications and return summary stats.

    Returns
    -------
    dict with keys: ``alt_mean_all4``, ``alt_q25_all4``, ``alt_q75_all4``,
    ``alt_mean_intraday``, ``alt_q25_intraday``, ``alt_q75_intraday``,
    ``null_mean_all4``, ``null_q975_all4``, ``alt_frac_below_0p20_all4``,
    ``n_reps_used``.
    """
    return _aggregate(
        rep_fn=lambda rng, master_seed, rep_idx: run_one_sim_replication(
            cal, rng,
            n_components=n_components,
            P_12_override=P_12_override,
            P_21_override=P_21_override,
            n_days=n_days,
            bars_per_day=bars_per_day,
            master_seed=master_seed,
            rep_idx=rep_idx,
        ),
        n_reps=n_reps,
        seed=seed,
        n_jobs=n_jobs,
    )
