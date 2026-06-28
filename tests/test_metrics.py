"""Unit tests for ``src.core.metrics``.

Verifies the canonical home for the partition / agreement metrics used
across the multi-frequency pipeline and the extended-analyses experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.metrics import (
    aggregation_loss,
    cross_freq_ari_matrix,
    cross_freq_extra_metrics,
    mean_offdiag_ari,
    mean_offdiag_ari_with_counts,
    shannon_entropy,
    variation_of_information,
)


# ---------------------------------------------------------------------------
# Information-theoretic helpers
# ---------------------------------------------------------------------------


def test_shannon_entropy_uniform_two_classes():
    labels = np.array([0, 1] * 50)
    h = shannon_entropy(labels)
    assert h == pytest.approx(np.log(2), abs=1e-9)


def test_shannon_entropy_constant_zero():
    assert shannon_entropy(np.zeros(10, dtype=int)) == pytest.approx(0.0, abs=1e-12)


def test_shannon_entropy_handles_empty_array():
    assert shannon_entropy(np.array([], dtype=int)) == 0.0


def test_variation_of_information_identical_partitions_is_zero():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=200)
    assert variation_of_information(labels, labels) == pytest.approx(0.0, abs=1e-12)


def test_variation_of_information_independent_partitions_is_positive():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=500)
    b = rng.integers(0, 2, size=500)
    assert variation_of_information(a, b) > 0.0


def test_variation_of_information_symmetric():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, size=200)
    b = rng.integers(0, 2, size=200)
    assert variation_of_information(a, b) == pytest.approx(variation_of_information(b, a), abs=1e-12)


# ---------------------------------------------------------------------------
# Off-diagonal summaries
# ---------------------------------------------------------------------------


def test_mean_offdiag_ari_known_matrix():
    df = pd.DataFrame(
        [[1.0, 0.4, 0.2],
         [0.4, 1.0, 0.6],
         [0.2, 0.6, 1.0]],
        index=["a", "b", "c"], columns=["a", "b", "c"],
    )
    expected = (0.4 + 0.2 + 0.4 + 0.6 + 0.2 + 0.6) / 6
    assert mean_offdiag_ari(df) == pytest.approx(expected, abs=1e-9)


def test_mean_offdiag_ari_none_or_empty():
    assert mean_offdiag_ari(None) is None
    assert mean_offdiag_ari(pd.DataFrame()) is None


def test_mean_offdiag_ari_1x1_frame_returns_none():
    """A 1x1 ARI matrix has no off-diagonal entries; the helper must return
    ``None`` rather than 0.0 or NaN (P1-T6: previously uncovered).
    """
    assert mean_offdiag_ari(pd.DataFrame([[1.0]])) is None


def test_mean_offdiag_ari_with_counts_unique_pairs():
    df = pd.DataFrame(
        [[1.0, 0.4, 0.2],
         [0.4, 1.0, 0.6],
         [0.2, 0.6, 1.0]],
        index=["a", "b", "c"], columns=["a", "b", "c"],
    )
    mean_val, n_valid, n_total = mean_offdiag_ari_with_counts(df)
    assert n_total == 3  # C(3, 2)
    assert n_valid == 3
    assert mean_val == pytest.approx((0.4 + 0.2 + 0.6) / 3, abs=1e-9)


def test_mean_offdiag_ari_with_counts_skips_nan():
    df = pd.DataFrame(
        [[1.0, np.nan, 0.2],
         [np.nan, 1.0, 0.6],
         [0.2, 0.6, 1.0]],
    )
    mean_val, n_valid, n_total = mean_offdiag_ari_with_counts(df)
    assert n_total == 3
    assert n_valid == 2
    assert mean_val == pytest.approx((0.2 + 0.6) / 2, abs=1e-9)


# ---------------------------------------------------------------------------
# Cross-frequency matrices
# ---------------------------------------------------------------------------


def _aligned_series(freqs: tuple[str, ...], n: int = 100, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return {f: pd.Series(rng.integers(0, 2, size=n), index=idx) for f in freqs}


def test_cross_freq_ari_matrix_shape_and_diagonal():
    """Strengthened (P1-T5): assert (i) shape, (ii) diagonal == 1.0, AND
    (iii) at least one off-diagonal pair is < 1.0 (else an all-ones matrix
    would also pass)."""
    freqs = ("5m", "15m", "1h", "1d")
    aligned = _aligned_series(freqs, n=200)
    mat = cross_freq_ari_matrix(aligned, freqs)
    assert mat.shape == (4, 4)
    assert list(mat.index) == list(freqs)
    assert (np.diag(mat.values) == 1.0).all()
    # Off-diagonal: at least one pair must be < 0.99.  With independently
    # drawn label series the off-diagonal ARIs concentrate near 0, so a
    # function that trivially returned all-ones would fail here.
    off = mat.values[np.triu_indices_from(mat.values, k=1)]
    assert (off < 0.99).any()


def test_cross_freq_ari_matrix_symmetric():
    freqs = ("5m", "15m", "1h")
    aligned = _aligned_series(freqs, n=200, seed=42)
    mat = cross_freq_ari_matrix(aligned, freqs)
    np.testing.assert_allclose(mat.values, mat.values.T, atol=1e-12)


def test_cross_freq_ari_matrix_too_short_returns_nan():
    freqs = ("5m", "15m")
    aligned = _aligned_series(freqs, n=5)  # below the 10-pair threshold
    mat = cross_freq_ari_matrix(aligned, freqs)
    assert np.isnan(mat.iloc[0, 1])


def test_cross_freq_extra_metrics_returns_ami_and_vi():
    freqs = ("5m", "15m", "1h")
    aligned = _aligned_series(freqs, n=200, seed=7)
    out = cross_freq_extra_metrics(aligned, freqs)
    assert set(out.keys()) == {"ami", "vi"}
    assert out["ami"].shape == (3, 3)
    assert out["vi"].shape == (3, 3)
    # VI is non-negative
    np.testing.assert_array_less(-1e-9, out["vi"].values)


# ---------------------------------------------------------------------------
# Aggregation loss (symmetric KL)
# ---------------------------------------------------------------------------


def test_aggregation_loss_identical_distributions_is_zero():
    p = np.array([0.5, 0.5])
    assert aggregation_loss(p, p) == pytest.approx(0.0, abs=1e-9)


def test_aggregation_loss_disjoint_distributions_is_positive():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert aggregation_loss(p, q) > 0.0


def test_aggregation_loss_is_symmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.3, 0.4, 0.3])
    assert aggregation_loss(p, q) == pytest.approx(aggregation_loss(q, p), abs=1e-12)


def test_aggregation_loss_normalises_unnormalised_input():
    p = np.array([2.0, 8.0])    # sums to 10 -> [0.2, 0.8]
    q = np.array([20.0, 80.0])  # sums to 100 -> [0.2, 0.8]
    assert aggregation_loss(p, q) == pytest.approx(0.0, abs=1e-9)


def test_aggregation_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        aggregation_loss(np.array([0.5, 0.5]), np.array([0.3, 0.4, 0.3]))


