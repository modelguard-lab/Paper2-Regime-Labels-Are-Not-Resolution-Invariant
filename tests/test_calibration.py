"""Backfill tests for ``src.core.calibration`` (P2-C: missing coverage).

Targets:
- JSON serialise/deserialise round-trip on ``CalibratedMSParams`` -> JSON ->
  ``load_ms_params`` (which returns ``CalibrationAt5m``).
- ``_convert_1h_to_5m`` scaling math: mean and variance scale linearly with
  the 12-bar-per-hour ratio, transition probabilities use the exact
  ``1 - (1 - p_1h)^(1/12)`` formula.
- ``CalibrationAt5m`` post-init: ``sigma2_k`` is backfilled from ``sigma_k``
  when not provided.
- ``load_ms_params`` raises ``FileNotFoundError`` when the JSON is missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json

import numpy as np
import pytest

from src.core.calibration import (
    CalibratedMSParams,
    CalibrationAt5m,
    _convert_1h_to_5m,
    load_ms_params,
)


def _make_synthetic_params() -> CalibratedMSParams:
    """Build a hand-tuned MS params object resembling a SPY 1h calibration."""
    return CalibratedMSParams(
        mu_0=2.5e-4,
        mu_1=-3.1e-5,
        sigma2_0=8e-6,
        sigma2_1=9e-5,
        P_12=0.0197,
        P_21=0.0254,
        data_source="synthetic.csv",
        sample_start="2024-01-02T09:30:00-05:00",
        sample_end="2025-12-31T16:00:00-05:00",
        n_bars=10_000,
        fit_freq="1h",
        fit_date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        log_likelihood=-12345.6,
    )


def test_calibration_json_round_trip(tmp_path: Path) -> None:
    """Serialise CalibratedMSParams -> JSON file -> load_ms_params -> compare.

    ``load_ms_params`` returns the 5m-scaled ``CalibrationAt5m``; the raw
    1h params are exposed via ``.raw``.
    """
    p = _make_synthetic_params()
    out = tmp_path / "ms_params.json"
    out.write_text(p.to_json(), encoding="utf-8")

    loaded = load_ms_params(out)
    # Round-trip: raw 1h fields recovered exactly (JSON float precision).
    raw = loaded.raw
    for field in (
        "mu_0", "mu_1", "sigma2_0", "sigma2_1", "P_12", "P_21",
        "n_bars", "fit_freq", "data_source",
    ):
        assert getattr(raw, field) == getattr(p, field), field
    assert raw.log_likelihood == pytest.approx(p.log_likelihood)


def test_convert_1h_to_5m_scaling_law() -> None:
    """``_convert_1h_to_5m`` must apply (i) ``mu / 12``, (ii) ``sigma2 / 12``,
    (iii) the 1/12-th matrix root of the 2x2 1h transition matrix (the exact
    closed-form 12-step inverse, not the separable per-edge approximation
    ``1 - (1 - p)^(1/12)`` which ignores P_21/P_12 coupling).
    """
    from scipy.linalg import fractional_matrix_power as _fmp

    p = _make_synthetic_params()
    cal5 = _convert_1h_to_5m(p)
    # Mean and variance are the additive-time scaling.
    assert cal5.mu_0 == pytest.approx(p.mu_0 / 12, rel=1e-12)
    assert cal5.mu_1 == pytest.approx(p.mu_1 / 12, rel=1e-12)
    assert cal5.sigma2_0 == pytest.approx(p.sigma2_0 / 12, rel=1e-12)
    assert cal5.sigma2_1 == pytest.approx(p.sigma2_1 / 12, rel=1e-12)
    assert cal5.sigma_0 == pytest.approx(np.sqrt(p.sigma2_0 / 12), rel=1e-12)
    assert cal5.sigma_1 == pytest.approx(np.sqrt(p.sigma2_1 / 12), rel=1e-12)
    # Exact 1/12-th matrix root.
    P_1h = np.array([[1.0 - p.P_12, p.P_12], [p.P_21, 1.0 - p.P_21]])
    P_5m_mat = np.real(_fmp(P_1h, 1.0 / 12.0))
    P_5m_mat = np.clip(P_5m_mat, 0.0, 1.0)
    P_5m_mat = P_5m_mat / P_5m_mat.sum(axis=1, keepdims=True)
    expected_P12_5m = float(P_5m_mat[0, 1])
    expected_P21_5m = float(P_5m_mat[1, 0])
    assert cal5.P_12 == pytest.approx(expected_P12_5m, rel=1e-10)
    assert cal5.P_21 == pytest.approx(expected_P21_5m, rel=1e-10)
    # Round-trip check: ``P_5m^12 == P_1h``.
    P_check = np.linalg.matrix_power(
        np.array([[1.0 - cal5.P_12, cal5.P_12], [cal5.P_21, 1.0 - cal5.P_21]]),
        12,
    )
    assert np.allclose(P_check, P_1h, atol=1e-8)
    # Sanity: the matrix root differs from the per-edge approximation.
    naive = 1.0 - (1.0 - p.P_12) ** (1.0 / 12.0)
    assert abs(expected_P12_5m - naive) > 1e-6


def test_calibration_at_5m_backfills_sigma2() -> None:
    """``CalibrationAt5m`` ``__post_init__`` must compute ``sigma2_k`` from
    ``sigma_k`` when not supplied (preserves backwards-compat for older
    constructor calls)."""
    raw = _make_synthetic_params()
    cal = CalibrationAt5m(
        mu_0=0.0, mu_1=0.0,
        sigma_0=1e-3, sigma_1=5e-3,
        P_12=0.01, P_21=0.02,
        raw=raw,
    )
    assert cal.sigma2_0 == pytest.approx(1e-3 ** 2, rel=1e-12)
    assert cal.sigma2_1 == pytest.approx(5e-3 ** 2, rel=1e-12)


def test_load_ms_params_missing_file_raises(tmp_path: Path) -> None:
    """Bad path -> FileNotFoundError with an actionable message."""
    bad = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Calibration JSON not found"):
        load_ms_params(bad)


def test_load_ms_params_round_trip_via_json_text(tmp_path: Path) -> None:
    """Spot-check that the JSON we write parses back as a plain dict with
    the same keys (defensive against any future ``to_json`` reformat)."""
    p = _make_synthetic_params()
    s = p.to_json()
    payload = json.loads(s)
    expected_keys = {
        "mu_0", "mu_1", "sigma2_0", "sigma2_1", "P_12", "P_21",
        "data_source", "sample_start", "sample_end", "n_bars",
        "fit_freq", "fit_date_utc", "log_likelihood",
        "statsmodels_version",
    }
    assert set(payload.keys()) == expected_keys
