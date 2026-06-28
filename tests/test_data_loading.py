"""Tests for the asset-loading helpers in ``src.data.data_ib``.

Specifically tests ``iter_loaded_assets``, the generator that consolidates the
"loop over symbols, build the canonical 5m CSV path, skip if missing, load"
idiom that previously appeared verbatim in every extended-analyses experiment.

``load_5m_ohlc`` itself is already covered indirectly by ``test_data_ib.py``;
the focus here is on the iteration contract and the on_missing callback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.data_ib import canonical_stem, iter_loaded_assets


def _write_synthetic_5m(path, seed: int = 0, n: int = 100) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="5min")
    base = 100 + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base + 0.05,
            "Low": base - 0.05,
            "Close": base + rng.normal(0, 0.02, n),
            "Volume": rng.integers(1_000, 10_000, n),
        },
        index=idx,
    )
    df.index.name = "Date"
    df.to_csv(path)


def test_iter_loaded_assets_yields_only_existing_files(tmp_path):
    """SPY exists, MISSING does not; only SPY should be yielded."""
    _write_synthetic_5m(tmp_path / "SPY_5m.csv")
    yielded = list(iter_loaded_assets(tmp_path, ["SPY", "MISSING"]))
    assert len(yielded) == 1
    symbol, stem, df = yielded[0]
    assert symbol == "SPY"
    assert stem == "SPY"
    assert isinstance(df, pd.DataFrame)
    assert {"Open", "High", "Low", "Close"}.issubset(df.columns)


def test_iter_loaded_assets_canonical_stem_for_decorated_symbols(tmp_path):
    """Symbols with Yahoo-style suffixes are normalised by stripping "=".

    canonical_stem no longer maps CL=F -> CL (the old special case
    collided with Cleveland-Cliffs); the file is now expected at
    {sym_with_=_stripped}_5m.csv, i.e. CLF_5m.csv.
    """
    _write_synthetic_5m(tmp_path / "CLF_5m.csv")
    yielded = list(iter_loaded_assets(tmp_path, ["CL=F"]))
    assert len(yielded) == 1
    symbol, stem, _ = yielded[0]
    assert symbol == "CL=F"
    assert stem == canonical_stem("CL=F") == "CLF"


def test_iter_loaded_assets_invokes_on_missing_callback(tmp_path):
    """The callback fires for every absent file with (symbol, path)."""
    missing: list[tuple[str, "object"]] = []

    def _record(symbol, path):
        missing.append((symbol, path.name))

    _write_synthetic_5m(tmp_path / "SPY_5m.csv")
    yielded = list(
        iter_loaded_assets(tmp_path, ["SPY", "X1", "X2"], on_missing=_record)
    )
    assert len(yielded) == 1
    assert missing == [("X1", "X1_5m.csv"), ("X2", "X2_5m.csv")]


def test_iter_loaded_assets_silent_when_callback_none(tmp_path):
    """No callback means missing files just get skipped silently."""
    _write_synthetic_5m(tmp_path / "SPY_5m.csv")
    yielded = list(iter_loaded_assets(tmp_path, ["NOPE", "SPY"]))
    assert [s[0] for s in yielded] == ["SPY"]


def test_iter_loaded_assets_empty_assets_list(tmp_path):
    yielded = list(iter_loaded_assets(tmp_path, []))
    assert yielded == []
