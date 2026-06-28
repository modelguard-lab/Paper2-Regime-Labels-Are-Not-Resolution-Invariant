"""ML-fit Markov-switching calibration of the synthetic-DGP baseline.

Replaces the previously-hardcoded ``mu=(0.0,-0.02), sigma=(0.10,0.25),
P=(0.02,0.02)`` constants in ``exp_04 / exp_11 / exp_13 / exp_16`` with a
real ML fit of a 2-state Markov-switching Gaussian model to SPY 1h log
returns over the empirical sample.

Two public entry points:

- :func:`fit_and_persist_ms_params` ML-fits the DGP on any asset's 1h
  log returns, writes the JSON, and returns the parameter dict.
  Idempotent: re-running on the same data produces identical output
  (modulo statsmodels EM seeding).
- :func:`load_ms_params` reads the persisted JSON and converts the 1h
  parameters to the 5m scale used by the per-replication simulator.

The 1h-to-5m conversion uses the principal 1/12-th matrix power of the
2x2 1h transition matrix via ``scipy.linalg.fractional_matrix_power``
(the Markov-embedding approach): ``P_5m = P_1h^{1/12}``.  Moment
scaling: ``mu_5m = mu_1h / 12``, ``sigma_5m = sigma_1h / sqrt(12)``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random as _random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CALIBRATION_FILENAME: str = "calibrated_ms_params.json"
GARCH_CALIBRATION_FILENAME: str = "calibrated_garch_params.json"


# ---------------------------------------------------------------------------
# Data structure persisted to JSON
# ---------------------------------------------------------------------------


@dataclass
class CalibratedMSParams:
    """Calibrated 2-state Markov-switching Gaussian parameters at 1h scale.

    All parameters are expressed in *raw log-return units* (not percentages).
    The convention is:

    - regime 0 = calm (lower variance), regime 1 = crisis (higher variance);
      this ordering is enforced by sorting on ``sigma2`` after the fit.
    - ``P_12`` = P(regime 1 at t+1 | regime 0 at t) = calm to crisis.
    - ``P_21`` = P(regime 0 at t+1 | regime 1 at t) = crisis to calm.
    """

    mu_0: float
    mu_1: float
    sigma2_0: float
    sigma2_1: float
    P_12: float
    P_21: float

    # Provenance
    data_source: str
    sample_start: str
    sample_end: str
    n_bars: int
    fit_freq: str
    fit_date_utc: str
    log_likelihood: float
    statsmodels_version: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
# ML fit on real SPY 1h returns
# ---------------------------------------------------------------------------


def _fit_ms_gaussian_1h(returns_1h: pd.Series) -> CalibratedMSParams:
    """ML-fit a 2-state Markov-switching Gaussian model on 1h log returns.

    Uses ``statsmodels.tsa.regime_switching.MarkovRegression`` with a
    switching constant and switching variance. Returns are scaled to %
    internally so the EM optimiser sees order-unity values, then the
    fitted parameters are rescaled back to raw log-return units.

    Determinism note: ``statsmodels`` ``MarkovRegression.fit`` relies on
    the process-global ``numpy.random`` and ``random`` modules during its
    EM ``search_reps`` warm starts; it does not expose a ``random_state``
    parameter we can pass through. To get bit-stable persisted JSON
    without contaminating the caller's RNG state, we snapshot the
    incoming ``np.random`` and ``random`` states, seed both with 42
    locally, run the fit, and unconditionally restore the prior state in
    a ``finally`` block. Other Paper 2 random sources already use
    explicit ``np.random.default_rng`` instances; this guarantees they
    are unaffected even if a caller invokes ``fit_and_persist_ms_params``
    interleaved with other ``np.random.*`` global-state calls.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError as e:
        raise ImportError(
            "ML calibration requires statsmodels. Install with: "
            "pip install statsmodels"
        ) from e

    if returns_1h.empty:
        raise ValueError("returns_1h is empty; cannot fit MS model")

    # Scale to % so the EM optimiser sees O(1) magnitudes; rescale at the end.
    y = returns_1h.values * 100.0
    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    # Determinism: snapshot caller RNG state, seed locally, fit, restore.
    # ``search_reps=20`` matches the round-count used elsewhere in the
    # paper for permutation/bootstrap reps.
    _np_state = np.random.get_state()
    _py_state = _random.getstate()
    try:
        np.random.seed(42)
        _random.seed(42)
        res = mod.fit(em_iter=50, search_reps=20, disp=False)
    finally:
        np.random.set_state(_np_state)
        _random.setstate(_py_state)
    if not res.mle_retvals.get("converged", False):
        raise RuntimeError(
            f"MS fit failed to converge: {res.mle_retvals}"
        )

    # MarkovRegression returns a flat ndarray; resolve names via the model.
    name_to_val = dict(zip(mod.param_names, np.asarray(res.params).tolist()))
    const_0 = float(name_to_val["const[0]"])
    const_1 = float(name_to_val["const[1]"])
    sig2_0 = float(name_to_val["sigma2[0]"])
    sig2_1 = float(name_to_val["sigma2[1]"])

    # p[i->i] are the free parameters; off-diagonals are 1 - p[i->i].
    p00 = float(name_to_val["p[0->0]"])
    p10 = float(name_to_val["p[1->0]"])  # = P(regime 0 at t+1 | regime 1 at t)
    p01 = 1.0 - p00
    # p11 = 1.0 - p10  # implicit

    # Validate finiteness and strict variance ordering before sorting.
    # A tie or non-finite variance signals a degenerate fit (one regime
    # absorbs zero mass, or the optimiser landed on a flat saddle point);
    # silently picking either branch would persist garbage to JSON.
    if not (np.isfinite(sig2_0) and np.isfinite(sig2_1)):
        raise RuntimeError(
            f"MS fit returned non-finite variance: sig2_0={sig2_0}, "
            f"sig2_1={sig2_1}"
        )
    if sig2_0 == sig2_1:
        raise RuntimeError(
            "MS fit returned identical regime variances "
            f"(sig2_0=sig2_1={sig2_0}); this indicates a degenerate fit "
            "where the two regimes are unidentified."
        )

    # Order regimes so 0 is calm (lower variance), 1 is crisis (higher).
    # Strict `<` because we have asserted distinctness above.
    if sig2_0 < sig2_1:
        mu_0_pct, mu_1_pct = const_0, const_1
        sig2_0_pct, sig2_1_pct = sig2_0, sig2_1
        P_12_h = p01
        P_21_h = p10
    else:
        mu_0_pct, mu_1_pct = const_1, const_0
        sig2_0_pct, sig2_1_pct = sig2_1, sig2_0
        # After swap, "regime 0" was the original "regime 1" so transition
        # P_12 (new 0 -> new 1) = P(old 1 -> old 0) = p[1->0] = p10.
        P_12_h = p10
        P_21_h = p01

    # Rescale params from % units back to raw log-return units.
    # mu: divide by 100; sigma2: divide by 100^2 = 10000.
    mu_0 = mu_0_pct / 100.0
    mu_1 = mu_1_pct / 100.0
    sigma2_0 = sig2_0_pct / 10000.0
    sigma2_1 = sig2_1_pct / 10000.0

    params = CalibratedMSParams(
        mu_0=float(mu_0),
        mu_1=float(mu_1),
        sigma2_0=float(sigma2_0),
        sigma2_1=float(sigma2_1),
        P_12=float(P_12_h),
        P_21=float(P_21_h),
        data_source="",  # filled in by caller (has the path)
        sample_start="",  # filled in by caller
        sample_end="",    # filled in by caller
        n_bars=int(len(returns_1h)),
        fit_freq="1h",
        fit_date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        log_likelihood=float(res.llf),
    )
    import statsmodels as _sm
    params.statsmodels_version = _sm.__version__
    return params


def fit_and_persist_ms_params(
    spy_5m_path: Path | str,
    out_path: Path | str,
) -> CalibratedMSParams:
    """ML-fit the MS-Gaussian DGP on any asset's 1h log returns and persist to JSON.

    Steps
    -----
    1. Load SPY 5m OHLC via ``data_ib.load_5m_ohlc`` (NY tz).
    2. Resample to 1h via the canonical ``resample_ohlc``.
    3. Compute 1h log returns ``log(C_t / C_{t-1})``.
    4. Fit 2-state MarkovRegression with switching constant and variance.
    5. Persist (mu_0, mu_1, sigma2_0, sigma2_1, P_12, P_21) plus provenance
       to ``out_path``.

    Returns
    -------
    CalibratedMSParams with the fitted values.
    """
    from ..data.data_ib import load_5m_ohlc
    from .features import resample_ohlc

    spy_5m_path = Path(spy_5m_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_5m = load_5m_ohlc(spy_5m_path)
    if df_5m.index.tz is None:
        # ambiguous="NaT" flags fall-back duplicate hours, nonexistent=
        # "shift_forward" advances spring-forward gaps. We then drop NaT
        # rows so DST transitions never silently corrupt the calibration
        # sample. ambiguous="infer" can raise on real-data DST boundaries.
        df_5m.index = df_5m.index.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
        df_5m = df_5m[df_5m.index.notna()]
    else:
        df_5m.index = df_5m.index.tz_convert("America/New_York")

    df_1h = resample_ohlc(df_5m, "1h")
    gaps = df_1h.index.to_series().diff()
    ret_1h = np.log(df_1h["Close"] / df_1h["Close"].shift(1))
    ret_1h = ret_1h.where(gaps <= pd.Timedelta(hours=1, minutes=30)).dropna()
    if len(ret_1h) < 200:
        raise ValueError(
            f"Need at least 200 1h returns for MS fit; got {len(ret_1h)}"
        )

    params = _fit_ms_gaussian_1h(ret_1h)
    # Fill provenance. ``isoformat`` is used (rather than ``str``) so the
    # JSON timestamp string is reliably round-trippable via
    # ``pd.Timestamp.fromisoformat`` regardless of the user's locale.
    params.data_source = str(spy_5m_path.name)
    params.sample_start = df_5m.index.min().isoformat()
    params.sample_end = df_5m.index.max().isoformat()

    out_path.write_text(params.to_json(), encoding="utf-8")
    logger.info(
        "Persisted calibrated MS params to %s | "
        "mu=(%.5f, %.5f), sigma=(%.5f, %.5f), P_12=%.4f, P_21=%.4f, llf=%.2f",
        out_path,
        params.mu_0, params.mu_1,
        np.sqrt(params.sigma2_0), np.sqrt(params.sigma2_1),
        params.P_12, params.P_21, params.log_likelihood,
    )
    return params


# ---------------------------------------------------------------------------
# Loader: 1h params -> 5m simulation params
# ---------------------------------------------------------------------------


@dataclass
class CalibrationAt5m:
    """Calibrated parameters expressed at the 5m simulation grid.

    Used by the canonical-pipeline simulator (``simulate_ms_path_5m``) to
    drive the synthetic 5m return generator. Both ``sigma`` (std) and
    ``sigma2`` (variance) are exposed so downstream consumers (e.g., the
    MS-GARCH simulator) can avoid a sqrt-then-square round-trip when they
    need the variance directly.
    """

    mu_0: float
    mu_1: float
    sigma_0: float
    sigma_1: float
    P_12: float
    P_21: float
    raw: CalibratedMSParams
    sigma2_0: float | None = None
    sigma2_1: float | None = None

    def __post_init__(self) -> None:
        # Backfill sigma2 from sigma if a caller constructed without them
        # (preserves backward-compat for any external constructor calls).
        if self.sigma2_0 is None:
            self.sigma2_0 = float(self.sigma_0) ** 2
        if self.sigma2_1 is None:
            self.sigma2_1 = float(self.sigma_1) ** 2
        if self.sigma_0 < 0 or self.sigma_1 < 0:
            raise ValueError(
                f"Negative sigma in CalibrationAt5m: sigma_0={self.sigma_0}, "
                f"sigma_1={self.sigma_1}"
            )
        if self.sigma2_0 < 0 or self.sigma2_1 < 0:
            raise ValueError(
                f"Negative sigma2 in CalibrationAt5m: sigma2_0={self.sigma2_0}, "
                f"sigma2_1={self.sigma2_1}"
            )


def _convert_1h_to_5m(p: CalibratedMSParams) -> CalibrationAt5m:
    """Convert 1h-scale (mu, sigma2, P) to 5m-scale.

    Scaling law (i.i.d. within regime, slow chain):

    - ``mu_5m = mu_1h / 12``
    - ``sigma_5m = sigma_1h / sqrt(12)`` (since variance is additive)
    - Transition probabilities are obtained from the 1/12-th matrix root of
      the 2x2 1h transition matrix using ``scipy.linalg.fractional_matrix_power``.
      The two-state 1/12-step root is NOT separable across ``P_12`` and
      ``P_21``; the previously-used ``p_5m = 1 - (1 - p_1h)^(1/12)`` formula
      is structurally wrong because it ignores the off-diagonal coupling.
    """
    from scipy.linalg import fractional_matrix_power as _fmp
    BARS_PER_HOUR = 12
    mu_0_5m = p.mu_0 / BARS_PER_HOUR
    mu_1_5m = p.mu_1 / BARS_PER_HOUR
    sigma2_0_5m = p.sigma2_0 / BARS_PER_HOUR
    sigma2_1_5m = p.sigma2_1 / BARS_PER_HOUR
    sigma_0_5m = float(np.sqrt(sigma2_0_5m))
    sigma_1_5m = float(np.sqrt(sigma2_1_5m))
    lam = 1.0 - p.P_12 - p.P_21
    if lam <= 0:
        raise ValueError(
            f"_convert_1h_to_5m: chain not embeddable (P_12+P_21={p.P_12+p.P_21:.4f} >= 1); "
            "1/12-th matrix root has complex entries"
        )
    P_1h = np.array([[1.0 - p.P_12, p.P_12], [p.P_21, 1.0 - p.P_21]])
    P_5m_complex = _fmp(P_1h, 1.0 / BARS_PER_HOUR)
    if np.max(np.abs(P_5m_complex.imag)) > 1e-6:
        raise ValueError(
            f"_convert_1h_to_5m: matrix root has non-negligible imaginary part "
            f"(max={np.max(np.abs(P_5m_complex.imag)):.2e}); chain not Markov-embeddable"
        )
    P_5m_mat = np.real(P_5m_complex)
    # Clip tiny negative numerical noise
    P_5m_mat = np.clip(P_5m_mat, 0.0, 1.0)
    # Renormalise rows so they sum to 1
    row_sums = P_5m_mat.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError(f"_convert_1h_to_5m: row sum non-positive after clip; degenerate matrix root")
    P_5m_mat = P_5m_mat / row_sums
    P_12_5m = float(P_5m_mat[0, 1])
    P_21_5m = float(P_5m_mat[1, 0])
    return CalibrationAt5m(
        mu_0=mu_0_5m, mu_1=mu_1_5m,
        sigma_0=sigma_0_5m, sigma_1=sigma_1_5m,
        sigma2_0=float(sigma2_0_5m), sigma2_1=float(sigma2_1_5m),
        P_12=P_12_5m, P_21=P_21_5m,
        raw=p,
    )


def load_ms_params(json_path: Path | str) -> CalibrationAt5m:
    """Load the persisted calibration JSON and return 5m-scale parameters.

    Pass the resulting :class:`CalibrationAt5m` to
    :func:`src.core.sim_dgp.simulate_ms_path_5m`.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Calibration JSON not found at {json_path}. "
            f"Run fit_and_persist_ms_params first."
        )
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    allowed = {f.name for f in dataclasses.fields(CalibratedMSParams)}
    raw = {k: v for k, v in raw.items() if k in allowed}
    p = CalibratedMSParams(**raw)
    return _convert_1h_to_5m(p)


# ---------------------------------------------------------------------------
# GARCH(1,1) calibration on SPY 5m returns (companion to MS calibration)
# ---------------------------------------------------------------------------


@dataclass
class CalibratedGarchParams:
    """Calibrated GARCH(1,1) parameters at the 5m simulation scale.

    Fitted on SPY 5m log returns scaled to %. ``alpha`` and ``beta`` are the
    universal-regime persistence coefficients; per-regime ``omega`` is set
    so the unconditional within-regime variance matches the MS-fit
    ``sigma2_regime`` after rescaling to raw log-return units. The simulator
    constructs the per-regime omega on the fly given the calibrated MS
    sigma2.
    """

    alpha: float
    beta: float
    omega_uncond_pct: float  # raw GARCH omega from the fit, in %^2 units

    data_source: str
    sample_start: str
    sample_end: str
    n_bars: int
    fit_freq: str
    fit_date_utc: str
    log_likelihood: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _fit_garch11_5m(returns_5m: pd.Series) -> CalibratedGarchParams:
    """Fit a single-regime GARCH(1,1) on 5m log returns scaled to %."""
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError(
            "GARCH calibration requires the 'arch' package. "
            "Install with: pip install arch"
        ) from e
    if returns_5m.empty:
        raise ValueError("returns_5m is empty; cannot fit GARCH model")
    y = returns_5m.values * 100.0
    am = arch_model(y, vol="GARCH", p=1, q=1, dist="normal", mean="zero")
    res = am.fit(disp="off")

    def _get_arch_param(params, *names):
        for name in names:
            if name in params.index:
                return float(params[name])
        raise KeyError(f"None of {names} in arch params")

    alpha = _get_arch_param(res.params, "alpha[1]", "alpha")
    beta = _get_arch_param(res.params, "beta[1]", "beta")
    omega = _get_arch_param(res.params, "omega")
    # Stationarity guard: alpha + beta must be < 1 for the unconditional
    # variance to exist. A degenerate fit on stress data could persist a
    # non-stationary spec to JSON, which would later silently break the
    # MS-GARCH simulator's ``omega = sigma^2 * (1 - alpha - beta)``
    # variance-matching trick.
    if alpha + beta >= 1.0:
        raise ValueError(
            f"GARCH(1,1) non-stationary: alpha+beta = {alpha+beta:.4f} "
            ">= 1.0; refusing to persist a degenerate fit."
        )
    return CalibratedGarchParams(
        alpha=alpha,
        beta=beta,
        omega_uncond_pct=omega,
        data_source="",
        sample_start="",
        sample_end="",
        n_bars=int(len(returns_5m)),
        fit_freq="5m",
        fit_date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        log_likelihood=float(res.loglikelihood),
    )


def fit_and_persist_garch_params(
    spy_5m_path: Path | str,
    out_path: Path | str,
) -> CalibratedGarchParams:
    """ML-fit GARCH(1,1) on SPY 5m log returns and persist to JSON."""
    from ..data.data_ib import load_5m_ohlc

    spy_5m_path = Path(spy_5m_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_5m = load_5m_ohlc(spy_5m_path)
    if df_5m.index.tz is None:
        # ambiguous="NaT" flags fall-back duplicate hours, nonexistent=
        # "shift_forward" advances spring-forward gaps. We then drop NaT
        # rows so DST transitions never silently corrupt the calibration
        # sample. ambiguous="infer" can raise on real-data DST boundaries.
        df_5m.index = df_5m.index.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
        df_5m = df_5m[df_5m.index.notna()]
    else:
        df_5m.index = df_5m.index.tz_convert("America/New_York")
    gaps = df_5m.index.to_series().diff()
    ret_5m = np.log(df_5m["Close"] / df_5m["Close"].shift(1))
    ret_5m = ret_5m.where(gaps <= pd.Timedelta(hours=1, minutes=30)).dropna()
    if len(ret_5m) < 500:
        raise ValueError(
            f"Need at least 500 5m returns for GARCH fit; got {len(ret_5m)}"
        )

    params = _fit_garch11_5m(ret_5m)
    params.data_source = str(spy_5m_path.name)
    params.sample_start = df_5m.index.min().isoformat()
    params.sample_end = df_5m.index.max().isoformat()
    out_path.write_text(params.to_json(), encoding="utf-8")
    logger.info(
        "Persisted calibrated GARCH params to %s | "
        "alpha=%.4f, beta=%.4f, alpha+beta=%.4f, llf=%.2f",
        out_path, params.alpha, params.beta,
        params.alpha + params.beta, params.log_likelihood,
    )
    return params


def load_garch_params(json_path: Path | str) -> CalibratedGarchParams:
    """Load the persisted GARCH-calibration JSON."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"GARCH calibration JSON not found at {json_path}. "
            f"Run fit_and_persist_garch_params first."
        )
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    allowed = {f.name for f in dataclasses.fields(CalibratedGarchParams)}
    raw = {k: v for k, v in raw.items() if k in allowed}
    return CalibratedGarchParams(**raw)
