"""Permutation-based stability primitives for label time-series.

Three families:

1. ``ordering_null_pvalue`` (generic): tests whether an observed statistic
   on one or more label sequences exceeds what label-frequency permutation
   alone would produce. The strict ordering null per METHODS.md.

2. ``mean_offdiag_ari_complete_case``: cross-frequency mean ARI on the
   complete-case subset (rows where every frequency has a non-null label).

3. ``permute_pvalue_mean_offdiag_ari`` and
   ``block_permute_pvalue_mean_offdiag_ari``: specialised permutation tests
   for the mean off-diagonal ARI used by Paper 2's cross-resolution
   experiments. The block variant preserves within-block autocorrelation.

All functions take ``freqs`` as an explicit parameter so the same primitive
can serve any frequency tuple, not only the ``("5m", "15m", "1h", "1d")``
default of Paper 2.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .config import DEFAULT_BLOCK_SIZE, DEFAULT_PERM_N, DEFAULT_PERM_SEED
from .metrics import freq_pairs


def ordering_null_pvalue(
    statistic_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_perm: int = 1000,
    seed: int = DEFAULT_PERM_SEED,
    one_sided: str = "greater",
) -> dict[str, object]:
    """One-sided permutation test against the ordering null.

    Parameters
    ----------
    statistic_fn : callable
        Function taking ``len(arrays)`` 1-D arrays and returning a float
        scalar. Called once on the original arrays for the observed value
        and ``n_perm`` times on independently permuted copies for the null
        distribution.
    *arrays : np.ndarray
        Label arrays. Each array is independently permuted on every
        iteration; this is the strict ordering null (no joint permutation).
    n_perm : int, default 1000
        Number of permutations.
    seed : int, default 42
        Seed for the numpy default RNG.
    one_sided : {"greater", "less"}
        Direction of the alternative hypothesis. ``"greater"`` returns the
        share of null replicates >= observed; ``"less"`` returns the share
        <= observed. Add-one smoothing is applied so the minimum reported
        p-value is ``1 / (n_perm + 1)``.

    Returns
    -------
    dict
        Keys: ``observed``, ``p_value``, ``null_mean``, ``null_ci`` (the
        2.5th and 97.5th percentiles of the null distribution).

    Raises
    ------
    ValueError
        If ``one_sided`` is not ``"greater"`` or ``"less"``.
    """
    if one_sided not in {"greater", "less"}:
        raise ValueError(f"one_sided must be 'greater' or 'less', got {one_sided!r}")

    inputs = [np.asarray(a) for a in arrays]
    if not inputs:
        raise ValueError("at least one array must be supplied")

    observed = float(statistic_fn(*inputs))

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype=float)
    for k in range(int(n_perm)):
        permuted = [rng.permutation(a) for a in inputs]
        null[k] = float(statistic_fn(*permuted))

    valid = ~np.isnan(null)
    n_valid = int(valid.sum())
    if n_valid < n_perm:
        import warnings as _w
        _w.warn(f"permute_pvalue: {n_perm - n_valid} of {n_perm} null replicates were NaN", RuntimeWarning)
    if n_valid == 0:
        # Match the early-return convention used by
        # ``permute_pvalue_mean_offdiag_ari`` /
        # ``block_permute_pvalue_mean_offdiag_ari`` below: an all-NaN
        # null distribution has no information, so return NaN p / CI
        # rather than letting ``np.nanpercentile`` emit a RuntimeWarning
        # and silently produce ``(nan, nan)``.
        return {
            "observed": observed,
            "p_value": float("nan"),
            "null_mean": float("nan"),
            "null_ci": (float("nan"), float("nan")),
        }
    if one_sided == "greater":
        n_extreme = int(np.sum(null[valid] >= observed))
    else:
        n_extreme = int(np.sum(null[valid] <= observed))

    p = (n_extreme + 1) / (n_valid + 1)

    return {
        "observed": observed,
        "p_value": float(p),
        "null_mean": float(np.nanmean(null)),
        # ``null_ci`` is the 2.5/97.5-percentile envelope of the null
        # distribution, NOT a confidence interval on the test statistic.
        # ``np.nanpercentile`` is used (matching the rest of this module)
        # so a stray NaN replicate does not poison the bound.
        "null_ci": (
            float(np.nanpercentile(null, 2.5)),
            float(np.nanpercentile(null, 97.5)),
        ),
    }


# ---------------------------------------------------------------------------
# Cross-frequency mean off-diagonal ARI (complete-case + permutation tests)
# ---------------------------------------------------------------------------


def _safe_int_labels(s: pd.Series, name: str) -> pd.Series:
    """Cast a label Series to int after asserting all finite values are integer-valued.

    ``df[freq].astype(int)`` silently floors fractional labels, masking
    upstream bugs.  This helper raises if any non-NaN label is not within
    floating-point round-off of an integer, then drops NaNs and rounds
    before casting.
    """
    finite = s.dropna()
    if len(finite) and not np.allclose(finite, finite.round()):
        raise ValueError(f"{name}: labels must be integer-valued")
    return finite.round().astype(int)


def _mean_offdiag_ari_from_aligned(
    y: dict[str, np.ndarray],
    freqs: tuple[str, ...],
) -> float:
    """Mean of pairwise ARIs across the upper-triangle of the freqs grid."""
    if not all(len(y[f]) == len(y[freqs[0]]) for f in freqs):
        raise ValueError("_mean_offdiag_ari_from_aligned: all label arrays must be same length")
    for f in freqs:
        arr = y[f]
        finite_arr = arr[np.isfinite(arr.astype(float))] if arr.dtype != object else arr
        if not np.all(np.isin(finite_arr, [0, 1])):
            raise ValueError(f"_mean_offdiag_ari_from_aligned: labels for {f} must be in {{0,1}}")
    vals = [adjusted_rand_score(y[fa], y[fb]) for fa, fb in freq_pairs(freqs)]
    return float(np.mean(vals)) if vals else float("nan")


def mean_offdiag_ari_complete_case(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
    index_subset: pd.DatetimeIndex | None = None,
) -> tuple[float | None, int]:
    """Mean off-diagonal ARI on rows where every frequency has a non-null label.

    Parameters
    ----------
    aligned : dict of pd.Series
        Per-frequency label series aligned to a common time axis.
    freqs : tuple of str
        Frequency tuple in canonical order.
    index_subset : pd.DatetimeIndex, optional
        Restrict the computation to this subset of timestamps. Defaults to the
        index of ``aligned[freqs[0]]``.

    Returns
    -------
    (mean_ari, n_complete_case) : tuple
        ``mean_ari`` is None if fewer than 10 complete-case rows are available.
    """
    base = aligned.get(freqs[0])
    if base is None or base.empty:
        return None, 0
    if index_subset is None:
        index_subset = base.index
    df = pd.DataFrame({f: aligned[f].reindex(index_subset) for f in freqs}).dropna()
    if len(df) == 0 and len(index_subset) > 0:
        import warnings as _w
        _w.warn("mean_offdiag_ari_complete_case: index_subset has no overlap with aligned series; possible tz mismatch", RuntimeWarning)
    if len(df) < 10:  # threshold differs from permute_pvalue_* (50) by design: minimum for ARI computation, not for permutation null
        return None, 0
    vals = [
        adjusted_rand_score(_safe_int_labels(df[fa], fa), _safe_int_labels(df[fb], fb))
        for fa, fb in freq_pairs(freqs)
    ]
    if not vals:
        return None, int(len(df))
    return float(np.mean(vals)), int(len(df))


def permute_pvalue_mean_offdiag_ari(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
    index_subset: pd.DatetimeIndex | None = None,
    n_perm: int = DEFAULT_PERM_N,
    seed: int = DEFAULT_PERM_SEED,
) -> tuple[float | None, tuple[float, float] | None, float | None]:
    """Permutation test for the mean off-diagonal ARI under the ordering null.

    Each frequency's label sequence is independently shuffled in time on every
    iteration; the resulting null distribution is the share of cross-frequency
    agreement attributable to marginal label frequencies alone.

    Returns
    -------
    (p_value, (ci_low, ci_high), observed_stat)
        Where the CI covers the 2.5 / 97.5 percentiles of the null distribution.
        Add-one smoothing keeps the minimum reported p-value at ``1 / (n_perm + 1)``.
    """
    base = aligned.get(freqs[0])
    if base is None or base.empty:
        return None, None, None
    if index_subset is None:
        index_subset = base.index
    if len(index_subset) < 50:
        return None, None, None

    y_df = pd.DataFrame(
        {freq: aligned[freq].reindex(index_subset) for freq in freqs}
    ).dropna()
    if len(y_df) < 50:
        return None, None, None
    y = {freq: _safe_int_labels(y_df[freq], freq).to_numpy() for freq in freqs}

    obs = _mean_offdiag_ari_from_aligned(y, freqs)
    if not np.isfinite(obs):
        return None, None, None

    rng = np.random.default_rng(seed)
    null_stats = np.empty(int(n_perm), dtype=float)
    for k in range(int(n_perm)):
        y_perm = {freq: rng.permutation(arr) for freq, arr in y.items()}
        null_stats[k] = _mean_offdiag_ari_from_aligned(y_perm, freqs)

    # Integer count form of the add-one-smoothed p-value: identical in
    # exact arithmetic to ``(np.mean(null >= obs) * n_perm + 1) / (n+1)``
    # but free of float drift across architectures.  NaN-aware: drop
    # null replicates where the statistic could not be computed and
    # warn rather than treat NaN as not-extreme.
    valid = ~np.isnan(null_stats)
    n_valid = int(valid.sum())
    if n_valid < n_perm:
        import warnings as _w
        _w.warn(f"permute_pvalue: {n_perm - n_valid} of {n_perm} null replicates were NaN", RuntimeWarning)
    if n_valid == 0:
        return None, None, float(obs)
    p = float((int(np.sum(null_stats[valid] >= obs)) + 1) / (n_valid + 1))
    ci = (
        float(np.nanpercentile(null_stats, 2.5)),
        float(np.nanpercentile(null_stats, 97.5)),
    )
    return p, ci, float(obs)


def _resolve_block_size(
    block_size: int | str,
    index_subset: pd.DatetimeIndex,
    n_rows: int,
) -> int:
    """Resolve a numeric block size from the ``"auto"`` sentinel.

    ``"auto"`` infers the median bar spacing from ``index_subset`` and picks
    a block of roughly one wall-clock day, clamped to ``[10, n_rows // 3]``
    so the block-permutation null still has at least three blocks. For 5m
    spacing this gives 288, for 1h it gives 24. The caller can always pass
    a numeric value to override.
    """
    if isinstance(block_size, int):
        return block_size
    if block_size == "auto":
        if len(index_subset) < 2:
            return max(10, n_rows // 3)
        # Filter to intraday-only diffs first: an overnight gap (~17h on
        # a futures session) skews the median away from the true bar
        # spacing whenever the index spans more than one trading day.
        diffs = pd.Series(index_subset).diff()
        intraday = diffs[diffs <= pd.Timedelta(hours=1, minutes=1)]
        spacing = intraday.median() if len(intraday) else diffs.median()
        if pd.isna(spacing) or spacing.total_seconds() <= 0:
            return max(10, min(DEFAULT_BLOCK_SIZE, n_rows // 3))
        target = int(round(86400.0 / spacing.total_seconds()))
        return max(10, min(n_rows // 3, target))
    raise ValueError(f"block_size must be int or 'auto', got {block_size!r}")


def block_permute_pvalue_mean_offdiag_ari(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
    index_subset: pd.DatetimeIndex | None = None,
    n_perm: int = DEFAULT_PERM_N,
    block_size: int | str = DEFAULT_BLOCK_SIZE,
    seed: int = DEFAULT_PERM_SEED,
) -> tuple[float | None, tuple[float, float] | None, float | None]:
    """Block-permutation variant preserving within-block autocorrelation.

    Instead of permuting individual timestamps, contiguous blocks of
    ``block_size`` bars are permuted. This preserves regime persistence
    within each block while destroying cross-resolution temporal alignment.

    NOTE: this null permutes blocks INDEPENDENTLY per frequency (each
    frequency gets its own ``rng.permutation``). This is the
    "weak independently-shuffled-labels null" documented in paper/main.tex
    near section 4 ("we treat them as sanity checks rather than primary
    evidence"). A stronger null that preserves cross-frequency block
    alignment would share one permutation across all frequencies; that
    variant is left as follow-on work and is not the published statistic.

    Parameters
    ----------
    block_size : int or ``"auto"``, default ``DEFAULT_BLOCK_SIZE``
        Block length in rows of ``index_subset``. Pass ``"auto"`` to derive
        a one-trading-day block from the index spacing (288 at 5m, 24 at
        1h, 1 at 1d clamped up to 10). Numeric default kept at the
        ``DEFAULT_BLOCK_SIZE`` constant (50) for backward compatibility
        with the published Table A.12 numbers.
    """
    base = aligned.get(freqs[0])
    if base is None or base.empty:
        return None, None, None
    if index_subset is None:
        index_subset = base.index

    y_df = pd.DataFrame(
        {freq: aligned[freq].reindex(index_subset) for freq in freqs}
    ).dropna()
    n = len(y_df)
    # Resolve ``block_size`` AFTER dropna so the auto-derived target uses
    # the post-dropna n (and the same complete-case index used to compute
    # the observed statistic).
    block_size = _resolve_block_size(block_size, y_df.index, n)
    if n < block_size * 3:
        return None, None, None
    y = {freq: _safe_int_labels(y_df[freq], freq).to_numpy() for freq in freqs}

    obs = _mean_offdiag_ari_from_aligned(y, freqs)
    if not np.isfinite(obs):
        return None, None, None

    rng = np.random.default_rng(seed)
    n_blocks = n // block_size
    # Truncate to ``n_blocks * block_size``: leftover bars beyond the last
    # full block were previously appended unshuffled, which biased the
    # null toward the observed value at the tail. Truncation is the
    # simpler fix and at most discards ``block_size - 1`` bars.
    keep = n_blocks * block_size
    null_stats = np.empty(int(n_perm), dtype=float)

    for k in range(int(n_perm)):
        y_perm: dict[str, np.ndarray] = {}
        # Match the circular-shift convention used by sim_dgp's
        # ``_block_permute_labels``: rotate by a uniform random offset
        # before splitting into blocks.  The same shift is shared across
        # all freqs in one permutation iteration so the cross-freq
        # alignment of the original block boundaries is destroyed in the
        # same way for every freq.  Half-open ``[0, keep)``: ``shift = keep``
        # is identical to ``shift = 0`` under circular rotation, so
        # ``rng.integers(0, keep + 1)`` would over-sample the no-rotation
        # case by 1/(keep+1) (negligible at keep ~= n_blocks * 50 but
        # cosmetically off-by-one relative to ``sim_dgp._block_permute_labels``).
        shift = int(rng.integers(0, n_blocks * block_size))
        for freq, arr in y.items():
            arr_keep = arr[:keep]
            rotated = np.concatenate([arr_keep[shift:], arr_keep[:shift]])
            blocks = [
                rotated[i * block_size:(i + 1) * block_size]
                for i in range(n_blocks)
            ]
            perm_order = rng.permutation(n_blocks)
            y_perm[freq] = np.concatenate([blocks[i] for i in perm_order])
        null_stats[k] = _mean_offdiag_ari_from_aligned(y_perm, freqs)

    # Integer count form of the add-one-smoothed p-value (matches the
    # plain permutation variant; see comment there).  NaN-aware: drop
    # null replicates where the statistic could not be computed and
    # warn rather than treat NaN as not-extreme.
    valid = ~np.isnan(null_stats)
    n_valid = int(valid.sum())
    if n_valid < n_perm:
        import warnings as _w
        _w.warn(f"permute_pvalue: {n_perm - n_valid} of {n_perm} null replicates were NaN", RuntimeWarning)
    if n_valid == 0:
        return None, None, float(obs)
    p = float((int(np.sum(null_stats[valid] >= obs)) + 1) / (n_valid + 1))
    ci = (
        float(np.nanpercentile(null_stats, 2.5)),
        float(np.nanpercentile(null_stats, 97.5)),
    )
    return p, ci, float(obs)
