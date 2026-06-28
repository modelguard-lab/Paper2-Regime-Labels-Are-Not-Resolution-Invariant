"""Tests for ``src.core.stability.ordering_null_pvalue``."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from src.core.stability import ordering_null_pvalue


def _mean_ari(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))


def test_ordering_null_detects_perfectly_aligned_sequences():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=300)
    b = a.copy()  # perfect agreement => ARI = 1
    result = ordering_null_pvalue(_mean_ari, a, b, n_perm=200, seed=0)
    assert result["observed"] == pytest.approx(1.0, abs=1e-12)
    # Permuting independently destroys agreement; p must be at the floor.
    assert result["p_value"] <= 1 / 201 + 1e-12


def test_ordering_null_random_sequences_null_centred_near_zero():
    """For independent random labels, the null distribution of ARI should be
    centred near zero (chance agreement) and the reported p-value must be a
    valid probability."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, size=300)
    b = rng.integers(0, 2, size=300)
    result = ordering_null_pvalue(_mean_ari, a, b, n_perm=400, seed=1)
    assert abs(result["null_mean"]) < 0.05
    assert 0.0 < result["p_value"] <= 1.0


def test_ordering_null_one_sided_less_direction():
    """Strengthened (P1-T4): with observed=1.0 (perfect agreement) and
    one_sided="less", every null replicate is <= 1.0, so add-one smoothing
    gives exactly ``(n_perm + 1) / (n_perm + 1) = 1.0``.  Tighten from
    ``>= 0.95`` to ``approx(1.0, abs=1/(n_perm+1))``.
    """
    rng = np.random.default_rng(2)
    n_perm = 200
    a = rng.integers(0, 2, size=200)
    b = a.copy()
    result = ordering_null_pvalue(
        _mean_ari, a, b, n_perm=n_perm, seed=2, one_sided="less"
    )
    assert result["p_value"] == pytest.approx(1.0, abs=1 / (n_perm + 1))


def test_ordering_null_invalid_direction_raises():
    a = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        ordering_null_pvalue(_mean_ari, a, a, n_perm=10, one_sided="two-sided")


def test_ordering_null_returns_expected_keys():
    a = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    b = a.copy()
    result = ordering_null_pvalue(_mean_ari, a, b, n_perm=50, seed=0)
    assert set(result.keys()) == {"observed", "p_value", "null_mean", "null_ci"}
    assert isinstance(result["null_ci"], tuple)
    assert len(result["null_ci"]) == 2
