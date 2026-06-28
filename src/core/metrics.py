"""Pure clustering / partition metrics used across the pipeline and experiments.

These functions are deliberately free of any module-level constants so they can
be reused in robustness scripts and notebooks without dragging the full
pipeline import surface. ``workflows/pipeline.py`` re-exports them under the
existing leading-underscore names for backward compatibility with the
extended-analyses experiments that import from it directly.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def freq_pairs(freqs: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    """Yield each unordered ``(fa, fb)`` frequency pair once.

    The previous codebase repeated ``for i, fa in enumerate(freqs): for j, fb
    in enumerate(freqs): if j <= i: continue`` six times across
    :mod:`metrics`, :mod:`stability`, and :mod:`aggregation`; that boilerplate
    is consolidated here.
    """
    for i, fa in enumerate(freqs):
        for fb in freqs[i + 1:]:
            yield fa, fb


def bh_fdr(pvals: np.ndarray | list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR adjustment.

    Returns ``(reject, p_adjusted)`` arrays aligned with the input order:

    * ``reject[i]``  -- True iff the i-th hypothesis is rejected at FDR ``alpha``.
    * ``p_adjusted[i]`` -- BH-step-up adjusted p-value (a.k.a. q-value),
      monotonically non-decreasing across the ranked p-vector.

    NaN inputs are passed through (rejected = False, q = NaN) and are not
    counted in the multiplicity ``m`` -- this matches the convention in
    statsmodels' :func:`multipletests` for the ``"fdr_bh"`` method.
    """
    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    finite = np.isfinite(p)
    m = int(finite.sum())
    reject = np.zeros(n, dtype=bool)
    q = np.full(n, np.nan, dtype=float)
    if m == 0:
        return reject, q

    finite_idx = np.where(finite)[0]
    finite_p = p[finite_idx]
    # P1-S2: refuse silently-invalid inputs.  BH q-values can mathematically
    # exceed 1 (we cap upward) but raw input p-values must be in [0, 1];
    # negative inputs almost always indicate an upstream sign-flip bug, not
    # a numerical edge case.  Previously these were silently clipped to 0,
    # masking the bug.
    if not ((finite_p >= 0).all() and (finite_p <= 1).all()):
        raise ValueError(
            "bh_fdr: raw p-values must lie in [0, 1]; "
            f"got min={float(finite_p.min())}, max={float(finite_p.max())}"
        )
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    # BH step-up: q_i = min_{k>=i} ranked[k] * m / (k+1), then sorted back.
    raw = ranked * m / (np.arange(m) + 1.0)
    cummin_rev = np.minimum.accumulate(raw[::-1])[::-1]
    # Upper clip only: q-values above 1 are mathematically possible but not
    # interpretable; lower clip is now redundant given the assertion above.
    q_ranked = np.minimum(cummin_rev, 1.0)
    # Unsort
    q_finite = np.empty(m, dtype=float)
    q_finite[order] = q_ranked
    q[finite_idx] = q_finite
    reject[finite_idx] = q_finite <= float(alpha)
    return reject, q


def _round_clip(score: float, decimals: int = 6) -> float:
    """Bound an ARI / AMI score to ``[-1, 1]`` then round.

    ``adjusted_rand_score`` and ``adjusted_mutual_info_score`` occasionally
    return values like ``1.0000000003`` due to floating-point arithmetic;
    clipping first guarantees the result lives in the legal interval before
    we round to a stable display precision. (The previous code rounded
    before clipping, which only happens to work because round-to-six rarely
    leaves the value outside ``[-1, 1]`` -- the order is brittle.)
    """
    return float(round(float(np.clip(score, -1.0, 1.0)), decimals))


# ---------------------------------------------------------------------------
# Information-theoretic helpers
# ---------------------------------------------------------------------------


def shannon_entropy(labels: Iterable[int] | np.ndarray) -> float:
    """Shannon entropy (nats) of a discrete label array."""
    arr = np.asarray(labels)
    _, counts = np.unique(arr, return_counts=True)
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def variation_of_information(a: Iterable[int] | np.ndarray, b: Iterable[int] | np.ndarray) -> float:
    """Variation of Information: ``VI = H(a) + H(b) - 2 * MI(a, b)``.

    Symmetric, non-negative, and bounded above by ``log(N)``.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"variation_of_information: a and b must be same length, got {a_arr.shape} vs {b_arr.shape}"
        )
    mi = mutual_info_score(a_arr, b_arr)
    return float(max(0.0, shannon_entropy(a_arr) + shannon_entropy(b_arr) - 2.0 * mi))


# ---------------------------------------------------------------------------
# ARI matrix summaries
# ---------------------------------------------------------------------------


def mean_offdiag_ari_with_counts(
    ari_df: pd.DataFrame | None,
) -> tuple[float | None, int, int]:
    """Return ``(mean_offdiag, n_valid_unique_pairs, n_unique_pairs)``.

    Counts unique pairs only (upper triangle), so each off-diagonal pair is
    visited once. ``n_unique_pairs`` = ``C(n, 2)``; for the symmetric
    matrices produced by :func:`cross_freq_ari_matrix` this equals exactly
    half of the off-diagonal cell count ``n*(n-1)``. This is the canonical
    pair-mean implementation used by the pipeline's per-asset summaries.

    P0-S3: callers reading the returned third value MUST treat it as
    ``n_unique_pairs``, NOT as the total number of off-diagonal cells.
    """
    if ari_df is None or ari_df.empty:
        return None, 0, 0
    vals = ari_df.values.astype(float)
    if vals.shape[0] != vals.shape[1]:
        return None, 0, 0
    if not np.allclose(vals, vals.T, equal_nan=True):
        raise ValueError("mean_offdiag_ari_with_counts: matrix must be symmetric")
    mask = np.triu(np.ones_like(vals, dtype=bool), k=1)
    offdiag = vals[mask]
    if offdiag.size == 0:
        return None, 0, 0
    n_valid_unique_pairs = int(np.isfinite(offdiag).sum())
    # ``n_unique_pairs`` = C(n, 2); for a symmetric matrix this is half the
    # off-diagonal cell count.  Kept as the third positional return for
    # backward compatibility with callers that unpacked it as ``n_total``.
    n_unique_pairs = int(offdiag.size)
    mean_val = float(np.nanmean(offdiag)) if n_valid_unique_pairs > 0 else None
    return mean_val, n_valid_unique_pairs, n_unique_pairs


def mean_offdiag_ari(ari_df: pd.DataFrame | None) -> float | None:
    """Mean of the off-diagonal entries of a square ARI matrix.

    Equivalent to :func:`mean_offdiag_ari_with_counts` for the symmetric
    matrices produced by :func:`cross_freq_ari_matrix` (the upper-triangle
    mean equals the full off-diagonal mean when ``M[i,j] == M[j,i]`` and
    NaNs are mirrored). Delegates to the with-counts variant so the two
    helpers cannot drift in their NaN-handling semantics.

    Returns ``None`` for empty / non-square inputs; ``np.nan`` cells are
    skipped.
    """
    mean_val, _, _ = mean_offdiag_ari_with_counts(ari_df)
    return mean_val


# ---------------------------------------------------------------------------
# Cross-frequency matrices (parameterised by the frequency tuple)
# ---------------------------------------------------------------------------


def cross_freq_ari_matrix(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
    index_subset: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build the symmetric ARI matrix across frequencies on a chosen index.

    Cells with fewer than 10 jointly-non-null pairs are filled with ``NaN``.
    """
    if index_subset is None:
        base = aligned.get(freqs[0])
        index_subset = base.index if base is not None else pd.DatetimeIndex([])

    n = len(freqs)
    ari = np.full((n, n), np.nan, dtype=float)
    idx = {f: i for i, f in enumerate(freqs)}
    for fa in freqs:
        a = aligned[fa].reindex(index_subset)
        valid_a = a.notna()
        i = idx[fa]
        if int(valid_a.sum()) < 10:
            ari[i, i] = np.nan
        else:
            label_a = a.loc[valid_a].round().astype(int)
            ari[i, i] = _round_clip(adjusted_rand_score(label_a, label_a))
    for fa, fb in freq_pairs(freqs):
        a = aligned[fa].reindex(index_subset)
        b = aligned[fb].reindex(index_subset)
        valid = a.notna() & b.notna()
        i, j = idx[fa], idx[fb]
        if int(valid.sum()) < 10:
            ari[i, j] = ari[j, i] = np.nan
        else:
            # P2 / metrics.py:195 -- ``.astype(int)`` truncates toward zero,
            # which silently maps 0.9999... onto 0 and would cause a
            # spurious cluster mismatch for upstream code that emits
            # near-integer floats.  ``.round()`` first guarantees the
            # nearest legal label is picked.
            a_loc = a.loc[valid]
            b_loc = b.loc[valid]
            if not np.allclose(a_loc, a_loc.round(), atol=1e-9):
                raise ValueError(f"cross_freq_ari_matrix: labels for freq {fa!r} are not integer-valued")
            if not np.allclose(b_loc, b_loc.round(), atol=1e-9):
                raise ValueError(f"cross_freq_ari_matrix: labels for freq {fb!r} are not integer-valued")
            a_int = a_loc.round().astype(int)
            b_int = b_loc.round().astype(int)
            score = adjusted_rand_score(a_int, b_int)
            ari[i, j] = ari[j, i] = _round_clip(score)
    return pd.DataFrame(ari, index=list(freqs), columns=list(freqs))


def aggregation_loss(
    p_fine: np.ndarray | list[float],
    p_coarse: np.ndarray | list[float],
) -> float:
    """Symmetric Kullback-Leibler divergence between two label distributions.

    Used to quantify the information lost when downsampling a regime-label
    sequence from a fine resolution to a coarser one. ``aggregation_loss``
    serves as the theoretical baseline against which empirical
    cross-resolution ARI is compared (see Paper 2 §3.5 and the METHODS.md
    "Aggregation loss" recipe).

    Parameters
    ----------
    p_fine : array-like of shape (K,)
        Empirical label-frequency distribution at the fine resolution within
        a single coarse-frequency window. Need not be normalised.
    p_coarse : array-like of shape (K,)
        Distribution at the coarse resolution on the same window.

    Returns
    -------
    float
        ``0.5 * (KL(p_fine || p_coarse) + KL(p_coarse || p_fine))`` in nats.
        Non-negative, equals zero if and only if the two distributions are
        identical after epsilon smoothing.

    Notes
    -----
    Additive epsilon smoothing in the log avoids ``log(0)`` when a label is
    empty in one distribution but present in the other. The smoothing biases
    the result towards zero by ``O(eps)``, which is negligible for the eps
    used here.
    """
    p = np.asarray(p_fine, dtype=float)
    q = np.asarray(p_coarse, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: p={p.shape}, q={q.shape}")
    if p.ndim != 1:
        raise ValueError(f"distributions must be 1-D, got shape {p.shape}")
    if p.sum() == 0 or q.sum() == 0:
        raise ValueError("symmetric_kl: both p and q must have positive mass")
    eps = 1e-9
    p = (p + eps) / (p.sum() + p.size * eps)
    q = (q + eps) / (q.sum() + q.size * eps)
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return 0.5 * (kl_pq + kl_qp)


def cross_freq_extra_metrics(
    aligned: dict[str, pd.Series],
    freqs: tuple[str, ...],
    index_subset: pd.DatetimeIndex | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute symmetric AMI and VI matrices alongside ARI."""
    if index_subset is None:
        base = aligned.get(freqs[0])
        index_subset = base.index if base is not None else pd.DatetimeIndex([])

    n = len(freqs)
    ami_mat = np.full((n, n), np.nan, dtype=float)
    # P2 / metrics.py:264-274 -- initialise to NaN so an unfilled cell can
    # be told from a legitimate zero (which is the value VI takes only when
    # the two label series are identical).  The diagonal is then set
    # explicitly to 0 since VI(a, a) = 0 by definition.
    vi_mat = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(vi_mat, 0.0)
    idx = {f: i for i, f in enumerate(freqs)}
    for fa in freqs:
        a = aligned[fa].reindex(index_subset)
        valid_a = a.notna()
        i = idx[fa]
        if int(valid_a.sum()) < 10:
            ami_mat[i, i] = np.nan
        else:
            label_a = a.loc[valid_a].round().astype(int).values
            ami_mat[i, i] = _round_clip(adjusted_mutual_info_score(label_a, label_a))
    for fa, fb in freq_pairs(freqs):
        a = aligned[fa].reindex(index_subset)
        b = aligned[fb].reindex(index_subset)
        valid = a.notna() & b.notna()
        i, j = idx[fa], idx[fb]
        if int(valid.sum()) < 10:
            ami_mat[i, j] = ami_mat[j, i] = np.nan
            vi_mat[i, j] = vi_mat[j, i] = np.nan
        else:
            # Same .round().astype(int) guard as cross_freq_ari_matrix; see
            # P2 / metrics.py:195 note above.
            a_loc = a.loc[valid]
            b_loc = b.loc[valid]
            if not np.allclose(a_loc, a_loc.round(), atol=1e-9):
                raise ValueError(f"cross_freq_extra_metrics: labels for freq {fa!r} are not integer-valued")
            if not np.allclose(b_loc, b_loc.round(), atol=1e-9):
                raise ValueError(f"cross_freq_extra_metrics: labels for freq {fb!r} are not integer-valued")
            av = a_loc.round().astype(int).values
            bv = b_loc.round().astype(int).values
            ami_mat[i, j] = ami_mat[j, i] = _round_clip(
                adjusted_mutual_info_score(av, bv)
            )
            vi_mat[i, j] = vi_mat[j, i] = float(
                round(variation_of_information(av, bv), 6)
            )
    return {
        "ami": pd.DataFrame(ami_mat, index=list(freqs), columns=list(freqs)),
        "vi": pd.DataFrame(vi_mat, index=list(freqs), columns=list(freqs)),
    }
