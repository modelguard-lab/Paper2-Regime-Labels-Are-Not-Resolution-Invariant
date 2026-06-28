"""Smoke tests for the GMM / HMM regime fits in ``src.workflows.pipeline``.

These tests do not require any market data. Synthetic two-regime log-volatility
inputs are constructed; the fit functions must (i) return a binary label
sequence aligned to the input index, (ii) flag the high-volatility regime as
``1``, and (iii) honour the fallback path when the GMM/HMM produce a degenerate
posterior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.models import fit_aligned_regimes, fit_regimes_per_frequency
from src.workflows.pipeline import (
    fit_regime,
    fit_regime_hmm,
    fit_regime_model,
    gmm_fit_diagnostics,
)


def _two_regime_features(seed: int = 0, n_per_regime: int = 200) -> pd.DataFrame:
    """Build a tiny feature frame with a clear low / high vol contrast."""
    rng = np.random.default_rng(seed)
    low = np.abs(rng.normal(0.005, 0.001, n_per_regime))
    high = np.abs(rng.normal(0.05, 0.01, n_per_regime))
    vol = np.concatenate([low, high])
    idx = pd.date_range("2026-01-01", periods=len(vol), freq="5min")
    return pd.DataFrame({"vol": vol}, index=idx)


def testfit_regime_returns_binary_labels_aligned_to_index():
    feats = _two_regime_features()
    labels, fallback = fit_regime(feats, n_components=2, freq="5m")
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(feats)
    assert labels.index.equals(feats.index)
    assert set(labels.unique()).issubset({0, 1})


def testfit_regime_separates_low_and_high_vol():
    feats = _two_regime_features()
    labels, _ = fit_regime(feats, n_components=2, freq="5m")
    # Construction: first half is low-vol, second is high-vol.
    n = len(feats) // 2
    low_share = labels.iloc[:n].mean()  # share of "1" labels in the low-vol half
    high_share = labels.iloc[n:].mean()
    assert high_share > low_share


def testfit_regime_constant_input_returns_calm():
    idx = pd.date_range("2026-01-01", periods=50, freq="5min")
    feats = pd.DataFrame({"vol": np.full(50, 0.01)}, index=idx)
    labels, fallback = fit_regime(feats, n_components=2, freq="5m")
    assert (labels == 0).all()
    assert fallback is False


def testfit_regime_too_short_returns_calm():
    idx = pd.date_range("2026-01-01", periods=2, freq="5min")
    feats = pd.DataFrame({"vol": [0.01, 0.02]}, index=idx)
    labels, fallback = fit_regime(feats, n_components=2, freq="5m")
    assert (labels == 0).all()


def testfit_regime_model_dispatches_gmm_vs_hmm(monkeypatch):
    """``fit_regime_model`` must actually route to the matching fitter.

    Strengthened (P0-T1): the previous version only verified that both label
    sets were binary, which a function that ignored the ``model`` arg would
    pass. We monkey-patch the two underlying fitters to record their calls
    and assert the dispatcher hits each exactly once with the right model arg.
    """
    feats = _two_regime_features()

    calls: dict[str, int] = {"gmm": 0, "hmm": 0}

    def _record_gmm(feats_, *, n_components=2, freq="", seed=0):
        calls["gmm"] += 1
        idx = feats_.index
        return pd.Series(np.zeros(len(idx)), index=idx), False

    def _record_hmm(feats_, *, n_components=2, freq="", seed=0):
        calls["hmm"] += 1
        idx = feats_.index
        return pd.Series(np.ones(len(idx)), index=idx), False

    # Patch the symbols imported into the pipeline module so the dispatcher
    # there resolves to our stubs.
    import src.workflows.pipeline as pl
    monkeypatch.setattr(pl, "fit_regime", _record_gmm)
    monkeypatch.setattr(pl, "fit_regime_hmm", _record_hmm)
    # Also patch the source module so the dispatcher (which lives in
    # src.core.models and is imported into pl as fit_regime_model) sees the
    # stubs through its module-level lookups.
    import src.core.models as cm
    monkeypatch.setattr(cm, "fit_regime", _record_gmm)
    monkeypatch.setattr(cm, "fit_regime_hmm", _record_hmm)

    gmm_labels, _ = fit_regime_model(feats, model="gmm", n_components=2)
    hmm_labels, _ = fit_regime_model(feats, model="hmm", n_components=2)
    assert calls == {"gmm": 1, "hmm": 1}
    # Sanity: the stubs return distinct constants so we can verify the
    # dispatcher returned each path's output rather than swapping them.
    assert (gmm_labels == 0).all()
    assert (hmm_labels == 1).all()


def testfit_regime_hmm_handles_no_hmmlearn(monkeypatch):
    """When ``hmmlearn`` is not importable, ``fit_regime_hmm`` must fall back
    to GMM (P0-T2: clear sys.modules cache before patching ``__import__``).
    """
    import builtins
    import sys
    monkeypatch.delitem(sys.modules, "hmmlearn", raising=False)
    monkeypatch.delitem(sys.modules, "hmmlearn.hmm", raising=False)

    real_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if name == "hmmlearn" or name.startswith("hmmlearn."):
            raise ImportError("simulated missing hmmlearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)

    feats = _two_regime_features()
    hmm_labels, _ = fit_regime_hmm(feats, n_components=2)
    # Restore __import__ before invoking GMM so its dependencies import cleanly.
    monkeypatch.setattr(builtins, "__import__", real_import)
    gmm_labels, _ = fit_regime(feats, n_components=2)
    # Semantic equivalence under fallback: HMM falls through to GMM, so the
    # returned label series must match what a direct GMM fit produces.
    assert set(hmm_labels.unique()).issubset({0, 1})
    pd.testing.assert_series_equal(
        hmm_labels.astype(int).reset_index(drop=True),
        gmm_labels.astype(int).reset_index(drop=True),
        check_names=False,
    )


def testgmm_fit_diagnostics_returns_dict():
    """``gmm_fit_diagnostics`` returns BIC / AIC / n_obs / component summary
    keys; values must be finite and AIC < BIC (n_obs > e^2 ~ 7.4).

    Strengthened (P0-T3): previously only ``isinstance(diag, dict)`` and
    ``"n_obs" in diag`` were checked, which a function returning
    ``{"n_obs": 0}`` and skipping the fit would pass.
    """
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    base = 100 + np.cumsum(rng.normal(0, 0.1, n))
    ohlc = pd.DataFrame(
        {
            "Open": base,
            "High": base + 0.05,
            "Low": base - 0.05,
            "Close": base + rng.normal(0, 0.02, n),
            "Volume": rng.integers(1_000, 10_000, n),
        },
        index=idx,
    )
    diag = gmm_fit_diagnostics(ohlc, freq="5m", stem="SPY", n_components=2)
    assert {"bic", "aic", "n_obs", "means", "stds", "weights"}.issubset(diag.keys())
    # n_obs is the count of finite log-vol rows; with default 5m window of
    # ~12 bars and ffill, n_obs equals n minus the leading NaN warm-up. The
    # exact value is deterministic given the seed and window spec.
    assert diag["n_obs"] > 0
    assert diag["n_obs"] <= n
    assert np.isfinite(diag["bic"]) and np.isfinite(diag["aic"])
    # AIC penalty (2k) < BIC penalty (k * log(n)) whenever log(n) > 2,
    # i.e. n > e^2 ~ 7.4. Our n_obs is in the hundreds.
    assert diag["aic"] < diag["bic"]
    assert len(diag["means"]) == 2
    assert len(diag["stds"]) == 2


# ---------------------------------------------------------------------------
# fit_regimes_per_frequency / fit_aligned_regimes helpers
# ---------------------------------------------------------------------------


def _synthetic_5m_ohlc(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        "2026-01-02 09:30", periods=n, freq="5min", tz="America/New_York"
    )
    base = 100 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 0.05,
            "Low": base - 0.05,
            "Close": base + rng.normal(0, 0.02, n),
            "Volume": rng.integers(1_000, 10_000, n),
        },
        index=idx,
    )


def testfit_regimes_per_frequency_returns_label_per_frequency():
    df = _synthetic_5m_ohlc()
    out = fit_regimes_per_frequency(df, "SPY", freqs=("5m", "15m", "1h"))
    assert set(out.keys()) == {"5m", "15m", "1h"}
    for freq, labels in out.items():
        assert isinstance(labels, pd.Series)
        # Originally-NaN bars (insufficient feature data) propagate as NaN;
        # finite labels must lie in {0, 1}.
        finite = labels.dropna()
        assert set(finite.unique()).issubset({0, 1})
        assert len(finite) > 0


def testfit_regimes_per_frequency_native_frequency_lengths():
    df = _synthetic_5m_ohlc(n=600)
    out = fit_regimes_per_frequency(df, "SPY", freqs=("5m", "15m", "1h"))
    # Coarser frequencies must have fewer labels than 5m.
    assert len(out["5m"]) >= len(out["15m"]) >= len(out["1h"])


def test_fit_aligned_regimes_returns_5m_aligned():
    df = _synthetic_5m_ohlc()
    out = fit_aligned_regimes(df, "SPY", freqs=("5m", "15m", "1h"))
    # Every label series shares the 5m timestamp axis.
    base_index = out["5m"].index
    for freq, labels in out.items():
        assert labels.index.equals(base_index), f"{freq} not aligned to 5m"


def test_fit_aligned_regimes_localises_naive_index(monkeypatch):
    df = _synthetic_5m_ohlc()
    df.index = df.index.tz_localize(None)  # make tz-naive
    out = fit_aligned_regimes(df, "SPY", freqs=("5m", "15m"))
    # The function localises in place; the returned labels carry tz-aware index.
    assert out["5m"].index.tz is not None
