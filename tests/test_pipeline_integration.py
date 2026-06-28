"""End-to-end smoke test for ``src.workflows.pipeline.analyze_asset``
(P2-C: pipeline currently has zero integration coverage).

The existing ``test_pipeline.py`` tests exclusively cover private primitives
(``resample_ohlc``, ``window_spec``, ``mean_offdiag_ari``).  Wiring
changes inside the canonical recipe are invisible to that file.  This test
runs the full ``analyze_asset`` call on a deterministic synthetic 5m OHLC
fixture and asserts the returned dict's contract: required keys are
present, the ARI matrix has the expected shape, and the alignment dict
has one Series per frequency on the 5m index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.workflows.pipeline import FREQS, analyze_asset


def _synthetic_5m_ohlc(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Deterministic 5m OHLC over n_days RTH-style trading days."""
    rng = np.random.default_rng(seed)
    bars_per_day = 78
    days = pd.bdate_range("2026-01-02", periods=n_days, tz="America/New_York")
    parts = []
    for d in days:
        parts.append(
            pd.date_range(
                d + pd.Timedelta(hours=9, minutes=30),
                periods=bars_per_day, freq="5min", tz="America/New_York",
            )
        )
    idx = pd.DatetimeIndex(np.concatenate([p.values for p in parts])).tz_localize(
        "UTC"
    ).tz_convert("America/New_York")
    n = len(idx)
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


def test_analyze_asset_produces_expected_keys() -> None:
    """``analyze_asset`` must return a dict with the contracted keys.  Test
    fails LOUDLY if a future refactor renames or drops a key (the existing
    private-primitive tests would not notice this kind of break).
    """
    df_5m = _synthetic_5m_ohlc(n_days=30)
    # Override windows to overlap the synthetic data; the default
    # 2026-02-28..04-15 / 2026-01-01..01-24 windows do not all cover a
    # 30-day fixture, and we want event/calm matrices to be non-empty.
    out = analyze_asset(
        "SPY", df_5m, model="gmm",
        event_window=("2026-02-01", "2026-02-12"),
        calm_window=("2026-01-02", "2026-01-15"),
    )

    expected = {
        "symbol", "model", "n_components", "window_scale", "rolling_days",
        "ari_matrix", "event_ari_matrix", "calm_ari_matrix",
        "regimes_aligned", "daily_df", "rolling_df",
        "fit_diagnostics", "tod_adjusted_ari_matrix", "expanding_ari_matrix",
        "ami_matrix", "vi_matrix", "tod_crisis_distribution",
        "overall_mean_ari_matrix", "overall_mean_ari_perm_stat",
        "overall_mean_ari_pvalue_perm",
        "block_perm_observed_stat", "block_perm_pvalue",
        "event_mean_ari", "calm_mean_ari",
        "rolling_ari_median", "expanding_mean_ari",
    }
    missing = expected - set(out.keys())
    assert not missing, f"analyze_asset missing keys: {sorted(missing)}"


def test_analyze_asset_ari_matrix_shape_and_alignment() -> None:
    """The ARI matrix shape must match ``len(FREQS)`` and the aligned regime
    dict must have one Series per freq on the 5m index."""
    df_5m = _synthetic_5m_ohlc(n_days=20)
    out = analyze_asset(
        "SPY", df_5m, model="gmm",
        event_window=("2026-01-15", "2026-01-23"),
        calm_window=("2026-01-02", "2026-01-10"),
    )
    n = len(FREQS)
    assert out["ari_matrix"].shape == (n, n)
    assert list(out["ari_matrix"].index) == list(FREQS)
    aligned = out["regimes_aligned"]
    assert set(aligned.keys()) == set(FREQS)
    for freq, series in aligned.items():
        assert series.index.equals(df_5m.index), (
            f"{freq} not aligned to 5m index"
        )
        # Labels are 0/1 ints when finite; NaN allowed for warm-up rows.
        finite = series.dropna()
        assert set(finite.unique()).issubset({0.0, 1.0, 0, 1})


def test_calibrated_dgp_endtoend_ari_in_expected_band() -> None:
    """End-to-end smoke test (peer-review H4): a single calibrated MS-Gaussian
    replication routed through ``analyze_asset`` must land within a wide
    band around the calibration mean (n=200 IQR is [0.099, 0.125]; a
    single replication has higher variance, so the band is widened to
    [0.05, 0.25]).

    This is the only test that exercises the full chain
    ``sim_dgp.simulate_ms_returns_5m`` → ``synthetic_ohlc_5m`` →
    ``analyze_asset`` → ``cross_freq_ari_matrix``. A regression that
    breaks any link in that chain (e.g., a bad clip schedule, a broken
    GMM fit, a misaligned 1d resample, a mis-specified calibrated DGP)
    will surface as ARI outside the [0.05, 0.25] band rather than as an
    obscure unit-test failure or, worse, a silent paper-number drift.
    """
    from src.core.calibration import CalibrationAt5m, CalibratedMSParams
    from src.core.sim_dgp import (
        DEFAULT_BARS_PER_DAY_RTH,
        DEFAULT_N_DAYS_5M,
        make_rth_5m_index,
        simulate_ms_returns_5m,
        synthetic_ohlc_5m,
    )
    from src.workflows.pipeline import analyze_asset

    # Hardcoded calibrated 5m DGP parameters (round-2 SPY 1h ML fit;
    # principal 1/12-th matrix root: P_12=0.021, P_21=0.028).
    raw = CalibratedMSParams(
        mu_0=0.0, mu_1=0.0,
        sigma2_0=(0.06 / 100.0) ** 2,
        sigma2_1=(0.18 / 100.0) ** 2,
        P_12=0.197, P_21=0.256,
        data_source="smoke-test", sample_start="", sample_end="",
        n_bars=0, fit_freq="1h", fit_date_utc="",
        log_likelihood=float("nan"),
    )
    cal = CalibrationAt5m(
        mu_0=0.0, mu_1=0.0,
        sigma_0=0.06 / 100.0, sigma_1=0.18 / 100.0,
        P_12=0.021, P_21=0.028,
        raw=raw,
    )

    rng = np.random.default_rng(seed=12345)
    index_5m = make_rth_5m_index(
        n_days=DEFAULT_N_DAYS_5M, bars_per_day=DEFAULT_BARS_PER_DAY_RTH,
    )
    rets, _ = simulate_ms_returns_5m(cal, len(index_5m), rng=rng)
    df_5m = synthetic_ohlc_5m(rets, index_5m)
    out = analyze_asset("SIM", df_5m)
    mean_ari = out["overall_mean_ari_matrix"]
    assert mean_ari is not None
    # n=200 IQR is [0.099, 0.125]. Single-replication variance is much
    # wider. Asserting [0.05, 0.25] catches any broken-pipeline regression
    # without being so tight that one unlucky seed false-positives.
    assert 0.05 <= float(mean_ari) <= 0.25, (
        f"calibrated-DGP ARI = {mean_ari:.4f} outside [0.05, 0.25] -- "
        "either the calibrated reference shifted or the pipeline broke."
    )


def test_analyze_asset_summary_scalars_are_well_typed() -> None:
    """Spot-check: the headline scalars (overall mean ARI, perm p-value, block
    perm p-value) must be floats in [0, 1] (mean ARI can be negative for
    chance independence, but with 20 days of synthetic SPY-like data the
    p-values are bounded probabilities)."""
    df_5m = _synthetic_5m_ohlc(n_days=20)
    out = analyze_asset(
        "SPY", df_5m, model="gmm",
        event_window=("2026-01-15", "2026-01-23"),
        calm_window=("2026-01-02", "2026-01-10"),
    )
    p_perm = out["overall_mean_ari_pvalue_perm"]
    p_block = out["block_perm_pvalue"]
    if p_perm is not None:
        assert 0.0 <= float(p_perm) <= 1.0
    if p_block is not None:
        assert 0.0 <= float(p_block) <= 1.0
    # Mean ARI for an ARI matrix is bounded in [-1, 1].
    overall = out["overall_mean_ari_matrix"]
    if overall is not None:
        assert -1.0 <= float(overall) <= 1.0
