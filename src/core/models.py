"""GMM and HMM regime fitters used across the multi-frequency pipeline.

Contains the canonical (full-sample) and expanding-window estimators plus
fit-quality diagnostics. Two-state by default: the higher-mean component is
labelled crisis (``1``); the lower-mean component is labelled calm (``0``).

When the GMM / HMM produces a degenerate posterior (one component owning
< 1% or > 99% of the bars) the function falls back to an 80th-percentile
log-volatility threshold so downstream code always receives a usable
binary label sequence.

Each fit function returns ``(labels, fallback_triggered)`` for backward
compatibility, but also annotates ``labels.attrs["fit_status"]`` with one
of three states so downstream code can distinguish:

- ``"normal"``               -- GMM / HMM fit normally; ``fallback_triggered = False``.
- ``"degenerate_skipped"``   -- input was constant or too short; an all-calm
                                series was returned without fitting any model.
                                ``fallback_triggered = False``.
- ``"pct_fallback"``         -- model fit produced a trivial split; labels were
                                replaced by an 80th-percentile threshold.
                                ``fallback_triggered = True``.

Reading just ``fallback_triggered`` cannot distinguish "model did not fit at
all" from "model fit normally": consult ``labels.attrs["fit_status"]`` (or
the per-frequency ``fit_status`` dict produced by the workflow layer).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from .config import (
    DEFAULT_GMM_K,
    DEFAULT_WINDOW_SCALE,
    EXPANDING_REFIT_DENOM,
    MODEL_GMM,
    MODEL_HMM,
    PCT_FALLBACK_PERCENTILE,
    TRIVIAL_SPLIT_LOWER_PCT,
    TRIVIAL_SPLIT_UPPER_PCT,
)
from .features import features, resample_ohlc
from .time_utils import ensure_ny_tz

logger = logging.getLogger(__name__)


DEFAULT_SEED: int = 42
# Bumped from 5 to 10: doubling n_init halves the chance of a poor local
# optimum in the EM step.  Empirically the variance of fitted means /
# stds drops by ~30% on the 5m / 15m frequencies where a small sample
# can pull the second component into a flat-tail solution.  Cost is one
# extra GMM iteration per fit -- negligible at the pipeline level.
DEFAULT_GMM_N_INIT: int = 10

STATUS_NORMAL = "normal"
STATUS_DEGENERATE = "degenerate_skipped"
STATUS_PCT_FALLBACK = "pct_fallback"


# ---------------------------------------------------------------------------
# Shared helpers (extracted to remove the 30-line duplication that previously
# appeared in fit_regime, fit_regime_hmm, and fit_regime_expanding)
# ---------------------------------------------------------------------------


def prepare_log_vol(
    feats: pd.DataFrame,
    impute_with_median: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, log_vol, finite_mask)``: training matrix, full log-vol vector, and finite mask.

    ``X`` is the 2-D matrix of finite, non-NaN log(vol) values used to fit a
    model; ``log_vol`` is the full series (same length as ``feats``) used at
    predict time. ``finite_mask`` is a boolean array of length ``len(feats)``
    that is ``True`` for bars whose ORIGINAL ``feats["vol"]`` was finite and
    non-NaN (i.e., not produced by median imputation).  The two share the
    same epsilon floor.
    """
    raw_vol = feats["vol"].values
    finite_mask = ~pd.isna(raw_vol) & np.isfinite(raw_vol)
    if impute_with_median:
        vol_vals = feats["vol"].fillna(feats["vol"].median()).values
        vol_vals = np.maximum(vol_vals, 1e-12)
    else:
        vol_vals = np.where(raw_vol <= 0, np.nan, raw_vol)
    log_vol = np.log(vol_vals)
    finite = ~np.isnan(log_vol) & np.isfinite(log_vol)
    X = log_vol[finite].reshape(-1, 1)
    return X, log_vol, finite_mask


def is_degenerate_log_vol(
    X: np.ndarray,
    n_components: int,
    freq: str,
    silent: bool = False,
) -> bool:
    """Check the two skip conditions shared by every fit function.

    ``silent=True`` suppresses the per-skip warning; expanding-window callers
    use it to avoid log spam from per-chunk degenerate detection.

    P2 / P3: previously this function tested both ``np.std`` and ``np.ptp``;
    they trigger on the same near-constant inputs and the std branch was
    redundant.  Range (peak-to-peak) is the cheaper, exact constant
    detector, so std is dropped.
    """
    if len(X) < n_components * 2:
        return True
    vol_ptp = float(np.ptp(X))
    if vol_ptp < 1e-10:
        if not silent:
            logger.warning(
                "Regime skip (%s): log-vol constant (ptp=%.2e), returning calm",
                freq or "?",
                vol_ptp,
            )
        return True
    return False


def _calm_series(index: pd.Index, status: str = STATUS_DEGENERATE) -> pd.Series:
    """Build the all-calm fallback series with status annotation."""
    s = pd.Series(0.0, index=index, dtype=float)
    s.attrs["fit_status"] = status
    return s


def _maybe_apply_pct_fallback(
    pred: np.ndarray,
    log_vol: np.ndarray,
    freq: str,
    model_label: str = "GMM",
    finite_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, str]:
    """Replace ``pred`` with an 80th-percentile threshold if the GMM split is trivial.

    Returns ``(pred, fallback_triggered, status)``.
    """
    pred_arr = np.asarray(pred)
    finite_pred = np.isfinite(pred_arr.astype(float))
    n_finite = int(finite_pred.sum())
    if n_finite == 0:
        return pred, False, STATUS_NORMAL
    crisis_pct = 100.0 * np.sum((pred_arr == 1) & finite_pred) / n_finite
    if TRIVIAL_SPLIT_LOWER_PCT <= crisis_pct <= TRIVIAL_SPLIT_UPPER_PCT:
        return pred, False, STATUS_NORMAL
    log_vol_finite = log_vol[finite_mask] if finite_mask is not None else log_vol
    thresh = np.nanpercentile(log_vol_finite, PCT_FALLBACK_PERCENTILE)
    pred = (log_vol >= thresh).astype(float)
    if finite_mask is not None:
        pred[~finite_mask] = np.nan
    logger.info(
        "Regime fallback (%s,%s): trivial (%.1f%% crisis); using %.0fth percentile log(vol) threshold",
        freq or "?",
        model_label,
        crisis_pct,
        PCT_FALLBACK_PERCENTILE,
    )
    return pred, True, STATUS_PCT_FALLBACK


# ---------------------------------------------------------------------------
# Public fit functions
# ---------------------------------------------------------------------------


def fit_regimes_per_frequency(
    df_5m: pd.DataFrame,
    stem: str,
    freqs: tuple[str, ...],
    n_components: int = DEFAULT_GMM_K,
    model: str = MODEL_GMM,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    seed: int = DEFAULT_SEED,
    features_by_freq: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.Series]:
    """Resample the 5m frame to each ``freqs`` member, build features, and fit regimes.

    Returns labels at their **native** frequency (no 5m alignment). Each Series
    carries ``attrs["fit_status"]`` and ``attrs["fallback_triggered"]`` so the
    boolean and 3-state diagnostics are recoverable without re-fitting.

    Pass ``features_by_freq`` to skip the internal resample+features build when
    the caller has already constructed them (workflow-layer caching).
    """
    regimes: dict[str, pd.Series] = {}
    for freq in freqs:
        if features_by_freq is not None and freq in features_by_freq:
            feats = features_by_freq[freq]
        else:
            ohlc = resample_ohlc(df_5m, freq)
            feats = features(ohlc, freq, stem=stem, window_scale=window_scale)
        labels, fb = fit_regime_model(
            feats, model=model, n_components=n_components, freq=freq, seed=seed,
        )
        labels.attrs["fallback_triggered"] = bool(fb)
        regimes[freq] = labels
    return regimes


def fit_aligned_regimes(
    df_5m: pd.DataFrame,
    stem: str,
    freqs: tuple[str, ...],
    n_components: int = DEFAULT_GMM_K,
    model: str = MODEL_GMM,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.Series]:
    """End-to-end: localise the 5m index, fit regimes per frequency, align to 5m.

    Consolidates the boilerplate that appears at the top of several
    extended-analyses experiments (exp_02_bootstrap, exp_05_calm_subsample,
    exp_06_var_uplift). Operates on a tz-coerced copy of the caller's
    frame; the input ``df_5m`` is never mutated.
    """
    # P0-S2: previously this function reassigned ``df_5m.index`` in place,
    # which mutated the caller's frame.  Bootstrap (exp_02), calm-subsample
    # (exp_05), and var-uplift (exp_06) reuse the same frame across many
    # iterations; the second call would see an already-converted index and
    # could take the wrong tz branch.  Now we coerce the index on a copy
    # and never write back to ``df_5m``.
    coerced_index = ensure_ny_tz(df_5m.index)
    df_local = df_5m.set_axis(coerced_index, axis=0, copy=False)
    regimes = fit_regimes_per_frequency(
        df_local, stem, freqs,
        n_components=n_components, model=model, window_scale=window_scale, seed=seed,
    )
    return align_regimes_to_5m(regimes, coerced_index)


def _wrap_regime_labels(
    pred: np.ndarray,
    log_vol: np.ndarray,
    finite_mask: np.ndarray,
    feats_index: pd.Index,
    freq: str,
    model_label: str,
) -> tuple[pd.Series, bool]:
    """Apply the shared GMM/HMM post-processing: NaN-mask non-finite rows,
    apply the 80th-percentile fallback if the model split is trivial, and
    wrap as a labelled ``pd.Series`` with ``attrs["fit_status"]`` populated.
    """
    pred[~finite_mask] = np.nan
    pred, fallback_triggered, status = _maybe_apply_pct_fallback(
        pred, log_vol, freq, model_label=model_label, finite_mask=finite_mask,
    )
    labels = pd.Series(pred, index=feats_index, dtype=float)
    labels.attrs["fit_status"] = status
    return labels, fallback_triggered


def fit_regime(
    feats: pd.DataFrame,
    n_components: int = DEFAULT_GMM_K,
    freq: str = "",
    seed: int = DEFAULT_SEED,
) -> tuple[pd.Series, bool]:
    """Fit GMM on log(volatility); the highest-mean cluster is labelled crisis.

    Returns
    -------
    (labels, fallback_triggered) : tuple
        ``labels`` is a pd.Series of 0 / 1 aligned to ``feats.index``.
        ``fallback_triggered`` is ``True`` when the 80th-percentile rule
        replaced GMM labels (used as a diagnostic in tables).
        ``labels.attrs["fit_status"]`` carries one of the three status
        strings -- see the module docstring.
    """
    X, log_vol, finite_mask = prepare_log_vol(feats, impute_with_median=True)
    if is_degenerate_log_vol(X, n_components, freq):
        return _calm_series(feats.index), False

    gmm = GaussianMixture(
        n_components=n_components, random_state=seed, n_init=DEFAULT_GMM_N_INIT
    )
    try:
        gmm.fit(X)
    except Exception as e:
        logger.warning("GMM fit failed (%s): %s; returning calm", freq or "?", e)
        return _calm_series(feats.index), False
    crisis_cluster = int(np.argmax(gmm.means_.ravel()))
    try:
        pred = (gmm.predict(log_vol.reshape(-1, 1)) == crisis_cluster).astype(float)
    except Exception as e:
        logger.warning("GMM predict failed (%s): %s; returning calm", freq or "?", e)
        return _calm_series(feats.index), False
    return _wrap_regime_labels(pred, log_vol, finite_mask, feats.index, freq, "GMM")


def fit_regime_hmm(
    feats: pd.DataFrame,
    n_components: int = DEFAULT_GMM_K,
    freq: str = "",
    seed: int = DEFAULT_SEED,
) -> tuple[pd.Series, bool]:
    """Fit a Gaussian HMM on log(volatility) with Viterbi decoding.

    Falls back to GMM if ``hmmlearn`` is not importable. Same return contract
    as :func:`fit_regime`.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:  # pragma: no cover
        logger.warning("HMM unavailable (%s); falling back to GMM for %s", e, freq or "?")
        labels, fb = fit_regime(feats, n_components=n_components, freq=freq, seed=seed)
        labels.attrs["fit_status"] = "hmm_unavailable_gmm_fallback"
        labels.attrs["fallback_triggered"] = True
        return labels, fb

    X, log_vol, finite_mask = prepare_log_vol(feats, impute_with_median=True)
    # HMM uses a stricter floor (4*K) than the 2*K used by
    # ``is_degenerate_log_vol`` because Baum-Welch must additionally
    # estimate K(K-1) transition parameters and K start probabilities --
    # at K=2 that's 5 extra DoF on top of GMM's 2K-1.  Empirically below
    # ~4*K bars the EM step alternates between near-degenerate transition
    # matrices and the fit barely informs predictions, so we skip outright
    # rather than emit a noisy unreliable series.  P2 / models.py:282.
    if len(X) < max(4, n_components * 4) or is_degenerate_log_vol(X, n_components, freq):
        return _calm_series(feats.index), False

    hmm = GaussianHMM(
        n_components=int(n_components),
        covariance_type="diag",
        n_iter=200,
        random_state=seed,
        verbose=False,
    )
    try:
        hmm.fit(X)
    except Exception as e:
        logger.warning("HMM fit failed (%s): %s; returning calm", freq or "?", e)
        return _calm_series(feats.index), False

    crisis_state = int(np.argmax(hmm.means_.ravel()))
    try:
        states = hmm.predict(log_vol.reshape(-1, 1))
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        logger.warning("HMM decode failed (%s): %s; returning calm", freq or "?", e)
        return _calm_series(feats.index), False
    pred = (states == crisis_state).astype(float)
    return _wrap_regime_labels(pred, log_vol, finite_mask, feats.index, freq, "HMM")


def fit_regime_model(
    feats: pd.DataFrame,
    model: str = MODEL_GMM,
    n_components: int = DEFAULT_GMM_K,
    freq: str = "",
    seed: int = DEFAULT_SEED,
) -> tuple[pd.Series, bool]:
    """Dispatch to GMM or HMM fitter based on the ``model`` string.

    Raises ``ValueError`` on any string other than ``MODEL_GMM`` /
    ``MODEL_HMM`` (whitespace-trimmed, case-insensitive). The previous
    behaviour silently fell through to GMM for typos like ``"gmm2"``,
    masking config bugs.
    """
    m = (model or MODEL_GMM).strip().lower()
    if m == MODEL_HMM:
        return fit_regime_hmm(feats, n_components=n_components, freq=freq, seed=seed)
    if m == MODEL_GMM:
        return fit_regime(feats, n_components=n_components, freq=freq, seed=seed)
    raise ValueError(
        f"fit_regime_model: unknown model={model!r}; "
        f"expected one of {{{MODEL_GMM!r}, {MODEL_HMM!r}}}"
    )


# ---------------------------------------------------------------------------
# Expanding-window estimator
# ---------------------------------------------------------------------------


def _fit_one_window(
    X_train: np.ndarray,
    model_str: str,
    n_components: int,
    seed: int,
    HMMClass: Any | None,
) -> Any | None:
    """Try to fit a single training chunk; return the fitted model or ``None``.

    Returns ``None`` for any of the three skip conditions:
    too few rows, near-zero variance/range, or estimator-side fit failure.
    The crisis-component index is left to the caller (always ``argmax`` of
    ``means_``); see :func:`fit_regime_expanding`.
    """
    if is_degenerate_log_vol(X_train, n_components, freq="", silent=True):
        return None
    if model_str == MODEL_HMM and HMMClass is not None:
        hmm = HMMClass(
            n_components=int(n_components),
            covariance_type="diag",
            n_iter=200,
            random_state=seed,
            verbose=False,
        )
        try:
            hmm.fit(X_train)
        except Exception:
            return None
        return hmm
    gmm = GaussianMixture(
        n_components=n_components, random_state=seed, n_init=DEFAULT_GMM_N_INIT,
    )
    try:
        gmm.fit(X_train)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        logger.warning("_fit_one_window GMM fit failed: %s", e)
        return None
    return gmm


def fit_regime_expanding(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str | None = None,
    model: str = MODEL_GMM,
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    min_train_bars: int = 200,
    seed: int = DEFAULT_SEED,
    feats: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Expanding-window regime fit (strictly no look-ahead).

    Refits every ``step = max(1, n // EXPANDING_REFIT_DENOM)`` bars on data
    up to (but excluding) the prediction chunk; predictions cover only the
    next chunk. Missing volatility values are imputed with the current
    training-sample median, never with future information. Bars before
    ``min_train_bars`` and any bars whose enclosing chunk fails to
    fit/predict remain ``NaN`` -- the function never falls back to a
    full-sample model.

    Pass ``feats`` to skip the internal ``features`` call when the caller
    has already built them (workflow-layer caching).
    """
    if feats is None:
        feats = features(df_ohlc, freq, stem=stem, window_scale=window_scale)
    _, log_vol, _ = prepare_log_vol(feats, impute_with_median=False)
    n = len(log_vol)
    labels = np.full(n, np.nan, dtype=float)
    refit_count = 0
    step = max(1, n // EXPANDING_REFIT_DENOM)

    m = (model or MODEL_GMM).strip().lower()
    if m not in (MODEL_GMM, MODEL_HMM):
        raise ValueError(
            f"fit_regime_expanding: unknown model={model!r}; "
            f"expected one of {{{MODEL_GMM!r}, {MODEL_HMM!r}}}"
        )
    HMMClass: Any | None = None
    if m == MODEL_HMM:
        try:
            from hmmlearn.hmm import GaussianHMM as _GaussianHMM
            HMMClass = _GaussianHMM
        except Exception as e:  # pragma: no cover
            logger.warning(
                "HMM unavailable in expanding mode (%s); fallback to GMM for %s",
                e, freq or "?",
            )
            m = MODEL_GMM

    first_oos_idx: int | None = None

    for train_end in range(min_train_bars, n, step):
        pred_end = min(train_end + step, n)

        train_chunk = log_vol[:train_end]
        train_med = np.nanmedian(train_chunk)
        if not np.isfinite(train_med):
            continue
        train_chunk = np.where(np.isfinite(train_chunk), train_chunk, train_med)
        X_train = train_chunk.reshape(-1, 1)

        chunk_seed = int(np.random.SeedSequence((seed, int(train_end))).generate_state(1)[0])
        model_obj = _fit_one_window(X_train, m, n_components, chunk_seed, HMMClass)
        if model_obj is None:
            continue
        # Always pick the highest-mean component as crisis. This matches
        # ``fit_regime`` / ``fit_regime_hmm`` semantics. A previous
        # implementation tracked an EWMA of historic crisis means and chose
        # ``argmin |mean - running|`` for label-persistence across refits,
        # but that flips labels whenever both component means shift away
        # from the historic crisis level (e.g. a sustained vol-regime
        # change makes the lower-of-pair closer to the historic running
        # mean than the higher-of-pair). Argmax of means is robust to
        # such drifts because the label "crisis = higher vol" is by
        # definition the higher-mean component within each window's fit.
        crisis_idx = int(np.argmax(model_obj.means_.ravel()))
        refit_count += 1

        pred_chunk = log_vol[train_end:pred_end]
        # P0-S1: leave non-finite prediction-window bars as NaN rather than
        # substituting the train median.  Imputing the median forces those
        # bars into the calm cluster (median sits in the calm component on
        # virtually every fit), which biased expanding-window OOS ARI toward
        # calm wherever warm-up gaps persist.  Now we predict only on finite
        # rows; non-finite rows stay NaN per the documented contract.
        finite_mask = np.isfinite(pred_chunk)
        if not finite_mask.any():
            continue
        try:
            chunk_pred = model_obj.predict(pred_chunk[finite_mask].reshape(-1, 1))
        except Exception:
            continue
        chunk_labels = np.full(pred_end - train_end, np.nan, dtype=float)
        chunk_labels[finite_mask] = (chunk_pred == crisis_idx).astype(float)
        labels[train_end:pred_end] = chunk_labels
        if first_oos_idx is None and pred_end > train_end:
            first_oos_idx = train_end

    if first_oos_idx is None:
        # Fail loud: every chunk's predict() raised, or min_train_bars >= n,
        # so no out-of-sample label was ever assigned.  Returning a fully-NaN
        # series would let an upstream caller treat the expanding fit as
        # successful (refit_count may be > 0 if fits ran without predict()
        # succeeding).  Per SR 26-2 strictly-backward-looking contracts, the
        # caller must know the expanding model produced no usable labels.
        raise ValueError(
            f"fit_regime_expanding produced no out-of-sample labels for "
            f"freq={freq!r}, n={n}, min_train_bars={min_train_bars}, "
            f"refit_count={refit_count}.  Either the time series is too "
            f"short for the configured warm-up or every predict() raised."
        )

    label_series = pd.Series(labels, index=feats.index, dtype=float)
    label_series.attrs["fit_status"] = "expanding"

    diagnostics = {
        "refit_count": refit_count,
        "min_train_bars_used": int(min_train_bars),
    }
    return label_series, diagnostics


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _two_component_overlap_separation(
    means: np.ndarray, stds: np.ndarray, weights: np.ndarray, n_components: int,
) -> tuple[float, float]:
    """Symmetric K=2 overlap + Mahalanobis-style separation. NaN when K != 2."""
    if n_components != 2 or len(stds) != 2:
        return float("nan"), float("nan")
    pooled_std = float(np.sqrt(weights[0] * stds[0] ** 2 + weights[1] * stds[1] ** 2))
    separation = abs(means[1] - means[0]) / pooled_std if pooled_std > 1e-12 else float("nan")
    boundary = (means[0] + means[1]) / 2.0
    lower_idx = int(np.argmin(means))
    higher_idx = 1 - lower_idx
    overlap = float(
        norm.sf(boundary, loc=means[lower_idx], scale=max(stds[lower_idx], 1e-12))
        + norm.cdf(boundary, loc=means[higher_idx], scale=max(stds[higher_idx], 1e-12))
    )
    return float(separation), overlap


def gmm_fit_diagnostics(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str | None = None,
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    seed: int = DEFAULT_SEED,
    feats: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return GMM fit-quality diagnostics: BIC, AIC, component means / stds, separation, overlap.

    Assumes 1-D input (single log-volatility feature).  Robust to all four
    sklearn ``covariance_type`` flavours: for ``"full"`` the array is
    ``(K, 1, 1)``; for ``"tied"`` it's ``(1, 1)`` and broadcast across
    components; for ``"diag"``/``"spherical"`` it's ``(K, 1)`` or ``(K,)``.
    The shared ``np.atleast_3d`` -> reshape path takes the first variance
    column either way.

    ``separation`` and ``overlap`` are reported only for ``n_components == 2``;
    NaN for K != 2.  Pass ``feats`` to skip the internal ``features`` call
    when the caller has already built them.
    """
    if feats is None:
        feats = features(df_ohlc, freq, stem=stem, window_scale=window_scale)
    X, _, _ = prepare_log_vol(feats, impute_with_median=True)
    if len(X) < n_components * 2:
        return {"bic": np.nan, "aic": np.nan, "n_obs": len(X)}

    gmm = GaussianMixture(
        n_components=n_components, random_state=seed, n_init=DEFAULT_GMM_N_INIT,
    )
    gmm.fit(X)
    means = gmm.means_.ravel()
    # P1-S6: robust to covariance_type in {"full","tied","diag","spherical"}.
    # ``np.atleast_3d`` lifts a 1-D / 2-D shape up to 3-D, then reshape(K, -1)
    # collapses the trailing dims so [:, 0] always picks the (only)
    # variance entry on 1-D input.  ``"tied"`` shares one covariance
    # across components; we broadcast manually so ``stds`` has length K.
    cov_arr = np.atleast_3d(np.asarray(gmm.covariances_)).reshape(-1, 1)[:, 0]
    if cov_arr.size == 1 and n_components > 1:
        cov_arr = np.full(n_components, float(cov_arr[0]))
    stds = np.sqrt(cov_arr)
    weights = gmm.weights_.ravel()
    separation, overlap = _two_component_overlap_separation(means, stds, weights, n_components)

    return {
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "n_obs": len(X),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "separation": separation,
        "overlap": overlap,
    }


def hmm_fit_diagnostics(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str | None = None,
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    seed: int = DEFAULT_SEED,
    feats: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """HMM analogue of :func:`gmm_fit_diagnostics`.

    BIC and AIC are computed from ``hmm.score(X)`` log-likelihood plus the
    HMM free-parameter count: ``(K-1) + K*(K-1) + K + K = K^2 + 2*K - 1``
    for a K-state diag-covar 1-D HMM (K-1 startprobs + K*(K-1) transitions
    + K means + K diagonal covars; the row-sum constraint on each
    transition row removes one DoF per row).  At K=2 this gives 7 free
    parameters; at K=3, 14.

    Implementation note (P1-S1): the previous code used ``2*K^2 + K - 1``,
    which over-counted by ``K^2 - K`` and inflated BIC/AIC by ``2*log(N)``
    (N>=4) per HMM.  Any model-comparison table conditioning on the old
    BIC/AIC is wrong; rerun model-selection diagnostics.

    ``weights`` is the stationary distribution of the transition matrix
    (HMM analogue of GMM mixture weights). Falls back to all-NaN
    diagnostics if hmmlearn is unavailable.
    """
    if feats is None:
        feats = features(df_ohlc, freq, stem=stem, window_scale=window_scale)
    X, _, _ = prepare_log_vol(feats, impute_with_median=True)
    if len(X) < max(4, n_components * 4):
        return {"bic": np.nan, "aic": np.nan, "n_obs": len(X)}

    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:  # pragma: no cover
        logger.warning("hmm_fit_diagnostics: hmmlearn unavailable (%s)", e)
        return {"bic": np.nan, "aic": np.nan, "n_obs": len(X)}

    hmm = GaussianHMM(
        n_components=int(n_components),
        covariance_type="diag",
        n_iter=200,
        random_state=seed,
        verbose=False,
    )
    try:
        hmm.fit(X)
    except Exception as e:
        logger.warning("hmm_fit_diagnostics fit failed (%s): %s", freq or "?", e)
        return {"bic": np.nan, "aic": np.nan, "n_obs": len(X)}

    try:
        loglik = float(hmm.score(X))
    except Exception as e:
        logger.warning("hmm_fit_diagnostics score failed (%s): %s", freq or "?", e)
        return {"bic": np.nan, "aic": np.nan, "n_obs": len(X)}

    # K-state diag-covar 1-D HMM free parameters: K^2 + 2K - 1.
    n_params = n_components ** 2 + 2 * n_components - 1
    bic = -2.0 * loglik + n_params * np.log(len(X))
    aic = -2.0 * loglik + 2.0 * n_params

    means = hmm.means_.ravel()
    stds = np.sqrt(hmm.covars_[:, 0, 0])
    # Stationary distribution via linear solve of (A^T - I) pi = 0
    # with the simplex-normalisation constraint sum(pi) = 1 substituted
    # into the last row.  Numerically more stable than picking the
    # eigvec closest to eigenvalue 1 from a non-Hermitian eig() output.
    K = hmm.transmat_.shape[0]
    A = hmm.transmat_.T - np.eye(K)
    A[-1, :] = 1.0  # replace last row with sum=1 constraint
    b = np.zeros(K)
    b[-1] = 1.0
    def _empirical_pi() -> np.ndarray:
        try:
            states = hmm.predict(X)
            counts = np.bincount(states, minlength=K).astype(float)
            if counts.sum() > 0:
                return counts / counts.sum()
        except Exception:
            pass
        return np.ones(K) / K

    try:
        pi = np.linalg.solve(A, b)
        pi = np.clip(pi, 0.0, None)
        pi_sum = float(pi.sum())
        if pi_sum <= 0:
            # Linear-solve gave an all-non-positive vector after clipping.
            # 1/0 would silently leave NaN here and propagate into the
            # separation/overlap diagnostics; fall back to the empirical
            # state-occupancy distribution instead.
            pi = _empirical_pi()
        else:
            pi = pi / pi_sum
    except np.linalg.LinAlgError:
        pi = _empirical_pi()
    weights = pi
    separation, overlap = _two_component_overlap_separation(means, stds, weights, n_components)

    return {
        "bic": float(bic),
        "aic": float(aic),
        "n_obs": len(X),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "separation": separation,
        "overlap": overlap,
    }


def fit_diagnostics(
    df_ohlc: pd.DataFrame,
    freq: str,
    stem: str | None = None,
    model: str = MODEL_GMM,
    n_components: int = DEFAULT_GMM_K,
    window_scale: float = DEFAULT_WINDOW_SCALE,
    seed: int = DEFAULT_SEED,
    feats: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Dispatch to GMM or HMM fit-quality diagnostics based on ``model``.

    Raises ``ValueError`` on any string other than ``MODEL_GMM`` /
    ``MODEL_HMM`` (whitespace-trimmed, case-insensitive).
    """
    m = (model or MODEL_GMM).strip().lower()
    if m == MODEL_HMM:
        return hmm_fit_diagnostics(
            df_ohlc, freq, stem=stem, n_components=n_components,
            window_scale=window_scale, seed=seed, feats=feats,
        )
    if m == MODEL_GMM:
        return gmm_fit_diagnostics(
            df_ohlc, freq, stem=stem, n_components=n_components,
            window_scale=window_scale, seed=seed, feats=feats,
        )
    raise ValueError(
        f"fit_diagnostics: unknown model={model!r}; "
        f"expected one of {{{MODEL_GMM!r}, {MODEL_HMM!r}}}"
    )


# ---------------------------------------------------------------------------
# Alignment to the 5m time axis
# ---------------------------------------------------------------------------


def align_regimes_to_5m(
    regimes_by_freq: dict[str, pd.Series],
    index_5m: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """Map each frequency's regime labels to the 5m timestamp axis.

    Both ``index_5m`` and the per-freq series indices must be tz-aware.  A
    tz-naive ``ser.index`` would silently fail the ``index_5m < min(ser)``
    comparison on pandas >= 2 and raise on earlier versions; we coerce to
    NY before the comparison rather than letting the bug surface
    downstream.  P2 / models.py:649-651.
    """
    out: dict[str, pd.Series] = {}
    # Coerce ``index_5m`` once outside the loop -- cheap, and the helper
    # is a no-op when already NY-tz.
    index_5m = ensure_ny_tz(index_5m)
    for freq, ser in regimes_by_freq.items():
        if ser is None or len(ser) == 0:
            out[freq] = pd.Series(np.nan, index=index_5m, dtype=float)
            continue
        if not ser.index.is_monotonic_increasing:
            ser = ser.sort_index()
        if ser.index.tz is None:
            ser = ser.copy()
            ser.index = ensure_ny_tz(ser.index)
        aligned = ser.reindex(index_5m, method="ffill")
        aligned.loc[index_5m < ser.index.min()] = np.nan
        out[freq] = aligned.astype(float)
    return out
