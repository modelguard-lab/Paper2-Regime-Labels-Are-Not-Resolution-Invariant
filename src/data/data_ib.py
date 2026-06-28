"""
Download 5-minute (and optional 1d) OHLC data via local Interactive Brokers API.

Requires: IB Gateway or TWS running locally (e.g. Connection Status: API + Historical Data Farm ON).
Default: host=127.0.0.1, port=4002 (Gateway paper). TWS paper port is 7497.
Connects with readonly=True so we only reqHistoricalData; no order/account requests (avoids "API write access" in Gateway).
Output CSV: Date (America/New_York), Open, High, Low, Close, Volume.
Same 5m source can be resampled to 15m/1h/1d per EXPERIMENT_DESIGN_FREQUENCY_2026.md.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from ._ib_contracts import (
    _contract_cont_future,
    _contract_forex,
    _contract_future,
    _contract_index,
    _contract_stock,
    _is_cont_future_marker,
    _require_ib,
)

logger = logging.getLogger(__name__)


def canonical_stem(symbol: str) -> str:
    """
    Map ticker to canonical file stem for output CSV names and future_expiry lookup.
    """
    sym = str(symbol).strip().upper()
    s = sym.replace("^", "").replace("/", "_").replace("=", "").replace(".", "_")
    # Removed: special-case "CLF" -> "CL" and "USDJPYX" -> "USDJPY".
    # "CLF" collides with Cleveland-Cliffs equity ticker. Yahoo-Finance's
    # "=F" futures suffix and "=X" FX suffix are stripped in build_contract;
    # this function should not retrofit additional ticker mappings.
    return s


# Per-future-symbol exchange routing for IB CONTFUT requests. NYMEX hosts
# energy futures (CL=WTI, BZ=Brent, NG=natural gas); COMEX hosts metals
# (GC=gold, SI=silver, HG=copper). Wrong exchange returns "no security
# definition" from IB Gateway. Default fallback NYMEX preserves
# backward-compat for energy-only callers.
_FUTURE_EXCHANGE_BY_SYMBOL: dict[str, str] = {
    "CL": "NYMEX",
    "ES": "CME",
    "BZ": "NYMEX",
    "NG": "NYMEX",
    "GC": "COMEX",
    "SI": "COMEX",
    "HG": "COMEX",
}


def build_contract(
    symbol: str,
    kind: Optional[str] = None,
    exchange: Optional[str] = None,
    future_expiry: Optional[str] = None,
) -> Any:
    """
    Build IB Contract for a symbol.
    symbol: e.g. 'SPY', 'CL=F', 'USDJPY=X'. Config uses SPY for equity and CL=F for WTI.
    kind: 'index' | 'stock' | 'forex' | 'future'; auto-detected if None.
    """
    raw = str(symbol).strip()
    sym_upper = raw.upper().replace("=X", "").replace("/", "").replace(".", "").replace("=F", "").replace("^", "")
    if kind is None:
        if sym_upper in ("SPX", "GSPC"):
            kind = "index"
        elif sym_upper in ("SPY", "USO", "QQQ", "IWM", "IEF", "GLD", "BTC-USD"):
            kind = "stock"
        elif sym_upper in ("USDJPY", "EURUSD", "GBPUSD"):
            kind = "forex"
        elif sym_upper in _FUTURE_EXCHANGE_BY_SYMBOL:
            kind = "future"
        else:
            kind = "stock"

    if kind == "index":
        # GSPC is the Yahoo alias for SPX; IB has no GSPC index, route to SPX.
        ib_index_symbol = "SPX" if sym_upper == "GSPC" else sym_upper
        return _contract_index(ib_index_symbol, exchange or "CBOE", "USD")
    if kind == "stock":
        ib_symbol = raw.replace("^", "").replace("=X", "").split(".")[0]
        return _contract_stock(ib_symbol, exchange or "SMART", "USD")
    if kind == "forex":
        return _contract_forex(symbol)
    if kind == "future":
        fut_symbol = sym_upper
        # Route GC/SI/HG to COMEX, energy futures to NYMEX. Caller can
        # override via the explicit ``exchange`` kwarg.
        fut_exchange = exchange or _FUTURE_EXCHANGE_BY_SYMBOL.get(fut_symbol, "NYMEX")
        if _is_cont_future_marker(future_expiry):
            return _contract_cont_future(fut_symbol, fut_exchange, "USD")
        return _contract_future(fut_symbol, fut_exchange, "USD", future_expiry)
    raise ValueError(f"Unknown kind: {kind}")


# --- Historical 5m bars (chunked; IB limits ~1–2 weeks per request for 5 min) ---

def _parse_dt(s: str) -> datetime:
    if isinstance(s, datetime):
        return s
    return pd.to_datetime(s).to_pydatetime()


def _to_ib_end_datetime(dt: datetime | pd.Timestamp, tz_name: str = "America/New_York") -> str:
    """Format datetime for IB endDateTime. Use UTC with dash (yyyymmdd-HH:mm:ss) to avoid 10314."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_name)
    else:
        ts = ts.tz_convert(tz_name)
    utc = ts.tz_convert("UTC")
    return utc.strftime("%Y%m%d-%H:%M:%S")


def _ib_duration_for_days(days: int) -> str:
    """Build an IB duration string long enough to cover a date span."""
    days = max(1, int(days))
    if days <= 365:
        return f"{days} D"
    years = (days + 364) // 365
    return f"{years} Y"


def _fetch_historical_5m_connected(
    ib: Any,
    contract: Any,
    start_date: str | datetime,
    end_date: str | datetime,
    duration_chunk: str = "1 W",
    use_rth: bool = False,
    tz_name: str = "America/New_York",
) -> pd.DataFrame:
    """Request 5m bars using an already-connected ib. No connect/disconnect.
    Default use_rth=False: Paper 2 (2026 US-Iran) needs 24h data for global event regime shifts."""
    start_dt = _parse_dt(start_date)
    end_dt = _parse_dt(end_date)
    start_ts = pd.Timestamp(start_dt)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(tz_name)
    else:
        start_ts = start_ts.tz_convert(tz_name)
    start_dt = start_ts.to_pydatetime()
    end_ts = pd.Timestamp(end_dt)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(tz_name)
    else:
        end_ts = end_ts.tz_convert(tz_name)
    if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0 and end_ts.microsecond == 0:
        end_ts = end_ts.replace(hour=16, minute=0, second=0, microsecond=0)
    end_dt = end_ts.to_pydatetime()
    all_bars = []
    current_end = end_dt
    last_oldest_ts: Optional[pd.Timestamp] = None
    is_forex = getattr(contract, "secType", None) == "CASH" or contract.__class__.__name__ == "Forex"
    is_contfuture = getattr(contract, "secType", None) == "CONTFUT" or contract.__class__.__name__ == "ContFuture"
    what_to_show = "MIDPOINT" if is_forex else "TRADES"
    use_rth_param = 0 if is_forex else (1 if use_rth else 0)
    if is_contfuture:
        # IB tightened CONTFUT semantics in mid-2026: passing any explicit
        # ``endDateTime`` returns ``Error 10339: Setting end date/time for
        # continuous future security type is not allowed`` (verified
        # against IB Gateway 10.30 on 2026-05-08 with BZ/NG/GC). Empty
        # ``endDateTime=""`` is the only accepted form and yields data
        # ending at "now". To cover a multi-month start->end span we
        # request ``durationStr`` long enough to exceed the desired span,
        # then trim to ``[start_dt, end_dt]`` after fetch. IB's 5-min
        # CONTFUT cap is ~6 months in a single request; for spans up to
        # that length one shot suffices. Longer spans are not currently
        # supported via CONTFUT and would need the dated-future fallback.
        ndays = max(1, int((end_dt - start_dt).days) + 7)
        duration_str = _ib_duration_for_days(ndays)
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting="5 mins",
                whatToShow=what_to_show,
                useRTH=use_rth_param,
                formatDate=1,
                timeout=600,
            )
        except Exception as e:
            logger.warning(
                "reqHistoricalData failed for continuous future durationStr=%s: %s",
                duration_str, e,
            )
            bars = None
        logger.info(
            "ContFut single-shot durationStr=%s -> %d bars",
            duration_str, len(bars) if bars else 0,
        )
        if bars:
            all_bars.extend(bars)
    else:
        while current_end > start_dt:
            end_str = _to_ib_end_datetime(current_end, tz_name)
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_str,
                    durationStr=duration_chunk,
                    barSizeSetting="5 mins",
                    whatToShow=what_to_show,
                    useRTH=use_rth_param,
                    formatDate=1,
                    timeout=120,
                )
            except Exception as e:
                logger.warning("reqHistoricalData failed for end=%s: %s", end_str, e)
                break
            logger.info("Chunk end=%s -> %d bars", end_str, len(bars) if bars else 0)
            if not bars:
                break
            all_bars.extend(bars)
            bar_times = []
            for b in bars:
                # Same fix as the post-loop builder below: use pd.Timestamp(b.date)
                # directly to preserve the tz-aware datetime; the prior
                # fromtimestamp() round-trip discarded the tz and shifted by the
                # system's UTC offset.
                ts = pd.Timestamp(b.date)
                ts = ts.tz_localize(tz_name) if ts.tzinfo is None else ts.tz_convert(tz_name)
                bar_times.append(ts)
            if not bar_times:
                break
            oldest_ts = min(bar_times)
            if last_oldest_ts is not None and oldest_ts >= last_oldest_ts:
                logger.info("Chunk made no progress (oldest=%s); stopping.", oldest_ts)
                break
            last_oldest_ts = oldest_ts
            current_end = (oldest_ts - pd.Timedelta(minutes=1)).to_pydatetime()
            if current_end <= start_dt:
                break
            time.sleep(0.5)
    if not all_bars:
        return pd.DataFrame()
    rows = []
    for b in all_bars:
        # ib_insync delivers bar.date as a tz-aware datetime in exchange-local
        # time (e.g. -04:00 for NY).  Use pd.Timestamp(t) directly to preserve
        # the tz; the previous datetime.fromtimestamp(t.timestamp()) round-trip
        # threw away the tz and re-interpreted the naive value in the system's
        # LOCAL timezone, which shifted every saved timestamp by the system's
        # UTC offset (e.g. +12h on NZ machines, +8h on China machines).
        ts = pd.Timestamp(b.date)
        rows.append({
            "Date": ts,
            "Open": b.open,
            "High": b.high,
            "Low": b.low,
            "Close": b.close,
            "Volume": getattr(b, "volume", 0),
        })
    df = pd.DataFrame(rows).set_index("Date")
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    if df.index.tz is None:
        # Defensive: bar.date is normally tz-aware; only enter this branch if
        # IB returned naive timestamps (different formatDate setting).
        df.index = df.index.tz_localize(tz_name)
    else:
        df.index = df.index.tz_convert(tz_name)
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(tz_name)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(tz_name)
    df = df.loc[df.index >= start_ts]
    df = df.loc[df.index <= end_ts]
    return df


def fetch_historical_5m(
    contract: Any,
    start_date: str | datetime,
    end_date: str | datetime,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 2,
    duration_chunk: str = "1 W",
    use_rth: bool = False,
    tz_name: str = "America/New_York",
) -> pd.DataFrame:
    """
    Request 5-minute bars from IB in chunks (IB limits lookback for 5 min data).
    Connects, fetches, disconnects. For multiple tickers use download_tickers_ib_5m (single connection).
    Default use_rth=False for Paper 2 (2026 global event) 24h regime coverage.
    """
    IB, _, _, _, _, _, _ = _require_ib()
    ib = IB()
    try:
        if ib.isConnected():
            try:
                ib.disconnect()
            except Exception:
                pass
        try:
            ib.connect(host, port, clientId=client_id, readonly=True)
        except Exception:
            try:
                ib.disconnect()
            except Exception:
                pass
            raise
        return _fetch_historical_5m_connected(ib, contract, start_date, end_date, duration_chunk, use_rth, tz_name)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def download_tickers_ib_5m(
    tickers: Sequence[str],
    output_dir: Path | str,
    start_date: str = "2026-01-01",
    end_date: str | None = None,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 2,
    future_expiry_by_symbol: Optional[dict] = None,
) -> None:
    """
    Download 5-minute OHLC for each ticker via IB and save one CSV per ticker.
    output_dir: directory for CSVs (e.g. data/).
    CSV columns: Date (America/New_York), Open, High, Low, Close, Volume.
    """
    IB, _, _, _, _, _, _ = _require_ib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    future_expiry_by_symbol = future_expiry_by_symbol or {}

    ib = IB()
    try:
        if ib.isConnected():
            try:
                ib.disconnect()
            except Exception:
                pass
        try:
            ib.connect(host, port, clientId=client_id, readonly=True)
            logger.info("IB connected host=%s port=%s clientId=%s (readonly)", host, port, client_id)
        except Exception as e:
            logger.error("IB connect failed: %s (host=%s port=%s). Is Gateway/TWS running?", e, host, port)
            try:
                ib.disconnect()
            except Exception:
                pass
            raise
        for ticker in tickers:
            stem = canonical_stem(ticker)
            out_path = output_dir / f"{stem}_5m.csv"
            future_expiry = (
                future_expiry_by_symbol.get(ticker)
                or future_expiry_by_symbol.get(stem)
                or future_expiry_by_symbol.get(ticker.upper().replace("=F", "").replace("=X", ""))
            )
            try:
                contract = build_contract(
                    ticker,
                    future_expiry=future_expiry,
                )
            except Exception as e:
                logger.warning("Skip %s: could not build contract: %s", ticker, e)
                continue

            # Incremental: if CSV exists, fetch from last bar onward.
            # If requested start_date is earlier than the earliest existing bar, do a backfill refresh from start_date.
            effective_start = start_date
            existing_df: Optional[pd.DataFrame] = None
            if out_path.exists() and not _is_cont_future_marker(future_expiry):
                try:
                    existing_df = load_5m_ohlc(out_path)
                    if not existing_df.empty:
                        first_ts = existing_df.index.min()
                        last_ts = existing_df.index.max()
                        requested_start_ts = pd.Timestamp(start_date)
                        if requested_start_ts.tzinfo is None:
                            requested_start_ts = requested_start_ts.tz_localize("America/New_York")
                        else:
                            requested_start_ts = requested_start_ts.tz_convert("America/New_York")
                        if requested_start_ts < first_ts:
                            logger.info(
                                "Backfill %s: requested start %s is earlier than existing first %s; refresh from requested start.",
                                ticker,
                                requested_start_ts,
                                first_ts,
                            )
                            effective_start = start_date
                        else:
                            # Incremental fetch starts at the next expected 5m bar timestamp.
                            tz = getattr(last_ts, "tz", None)
                            next_ts = pd.Timestamp(last_ts) + pd.Timedelta(minutes=5)
                            end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                            if end_ts.tzinfo is None:
                                end_ts = end_ts.tz_localize(tz if tz is not None else "America/New_York")
                            elif tz is not None:
                                end_ts = end_ts.tz_convert(tz)
                            if next_ts >= end_ts:
                                logger.info("Skip %s: already up to date (last=%s)", ticker, last_ts)
                                continue
                            effective_start = next_ts.strftime("%Y-%m-%d %H:%M:%S")
                            logger.info("Incremental %s: from %s (last existing %s) to %s", ticker, effective_start, last_ts, end_date)
                except Exception as e:
                    logger.warning("Could not read existing %s, will full download: %s", out_path, e)
            elif out_path.exists() and _is_cont_future_marker(future_expiry):
                logger.info("Continuous future requested for %s; refreshing full history from %s.", ticker, start_date)

            logger.info("Downloading 5m %s from %s to %s -> %s", ticker, effective_start, end_date, out_path)
            try:
                df = _fetch_historical_5m_connected(ib, contract, effective_start, end_date)
            except Exception as e:
                logger.error("IB fetch failed for %s: %s", ticker, e)
                continue
            if df.empty:
                if existing_df is not None:
                    logger.info("No new bars for %s, keep existing %d rows", ticker, len(existing_df))
                else:
                    logger.warning("No 5m data for %s", ticker)
                continue
            if existing_df is not None and not existing_df.empty:
                df = pd.concat([existing_df, df])
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()
            df.index.name = "Date"
            df.to_csv(out_path)
            logger.info("Saved %s %d rows to %s", ticker, len(df), out_path)
    finally:
        try:
            ib.disconnect()
            logger.info("IB disconnected clientId=%s", client_id)
        except Exception as e:
            logger.warning("IB disconnect failed (clientId=%s): %s", client_id, e)


def iter_loaded_assets(
    raw_dir: Path | str,
    assets: list[str],
    on_missing: Any | None = None,
):
    """Yield ``(symbol, stem, df_5m)`` for every asset whose 5m CSV exists.

    Centralises the "loop over the asset list, build the canonical 5m CSV
    path, skip-if-missing, otherwise load" idiom that appeared verbatim in
    every extended-analyses experiment.

    Parameters
    ----------
    raw_dir : Path | str
        Directory containing the per-asset CSVs.
    assets : list of str
        Asset symbols (e.g., ``["SPY", "USDJPY", "CL=F"]``).
    on_missing : callable, optional
        Called as ``on_missing(symbol, path)`` whenever a file is absent.
        Useful for emitting a warning at the call site without forcing the
        helper to depend on the caller's logger.

    Yields
    ------
    (symbol, stem, df_5m) : tuple
        ``stem`` is the result of :func:`canonical_stem`; ``df_5m`` is the
        loaded OHLC frame.
    """
    raw_dir = Path(raw_dir)
    for symbol in assets:
        stem = canonical_stem(symbol)
        path_5m = raw_dir / f"{stem}_5m.csv"
        if not path_5m.exists():
            if on_missing is not None:
                on_missing(symbol, path_5m)
            continue
        df_5m = load_5m_ohlc(path_5m)
        yield symbol, stem, df_5m


def load_5m_ohlc(path: Path | str) -> pd.DataFrame:
    """Load a 5m CSV produced by download_tickers_ib_5m (Date index, Open/High/Low/Close/Volume).

    Robust to mixed-offset timestamps: when 5m bars span a DST boundary the
    CSV contains both ``-04:00`` and ``-05:00`` offsets in the same column,
    which old pandas paths used to coerce to NaT.  We parse with
    ``utc=True`` to normalise everything to UTC first, then convert to NY.

    Validates that required OHLC columns are present and coerces OHLCV to
    numeric so downstream arithmetic does not silently break on stray string
    cells.  ``encoding="utf-8-sig"`` strips a UTF-8 BOM if present (otherwise
    the first column header would be read as ``Date`` (a leading U+FEFF byte-order mark)).
    """
    # Read with the index column as plain strings so we can parse it ourselves
    # (avoids the silent NaT coercion that ``parse_dates=True`` does on mixed
    # offsets in some pandas versions).
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)

    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    nonnull_close = df["Close"].dropna()
    if nonnull_close.empty:
        raise ValueError("data_ib: Close column is entirely NaN")
    if not (nonnull_close > 0).all():
        raise ValueError("data_ib: Close column contains non-positive values")

    idx = pd.to_datetime(df.index, errors="coerce", utc=True)
    if idx.isna().all():
        raise ValueError("data_ib: failed to parse any timestamps from CSV")

    bad = idx.isna()
    if bad.any():
        n_bad = int(bad.sum())
        df = df.loc[~bad].copy()
        idx = idx[~bad]
        logger.warning("Dropped %d rows with unparseable timestamps in %s", n_bad, path)

    df.index = pd.DatetimeIndex(idx).tz_convert("America/New_York")
    df = df.sort_index()
    return df


# FX symbols whose Volume column is a placeholder from IB MIDPOINT requests.
# Volume on these is meaningless (often -1 or 0); skip the negative-volume
# warning entirely rather than relying on neg_mask.all() which would suppress
# the warning only when EVERY row is negative and miss partial-negative cases.
FX_VOLUME_SKIP_ASSETS: frozenset[str] = frozenset(
    {"USDJPY", "EURUSD", "GBPUSD", "USDJPYX", "EURUSDX", "GBPUSDX"}
)


def validate_5m_ohlc(df: pd.DataFrame, symbol: str) -> list[str]:
    """
    Check 5m OHLC DataFrame for common issues. Returns list of issue messages (empty if OK).
    """
    issues: list[str] = []
    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            issues.append(f"{symbol}: missing column '{col}'")
    if issues:
        return issues
    if df.empty:
        issues.append(f"{symbol}: empty DataFrame")
        return issues
    # Close must be strictly positive (downstream log-returns would propagate
    # -inf otherwise). Report as an issue rather than raising so the function
    # honours its docstring contract of returning a list of issue messages.
    if (df["Close"] <= 0).any():
        issues.append(f"Close has {(df['Close'] <= 0).sum()} non-positive values")
    # NaN
    nan_counts = df[required].isna().sum()
    if nan_counts.any():
        issues.append(f"{symbol}: NaN in {nan_counts[nan_counts > 0].to_dict()}")
    # OHLC consistency: H >= O,L,C; L <= O,H,C
    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl > 0:
        issues.append(f"{symbol}: High < Low in {bad_hl} bars")
    bad_ho = (df["High"] < df["Open"]).sum()
    if bad_ho > 0:
        issues.append(f"{symbol}: High < Open in {bad_ho} bars")
    bad_hc = (df["High"] < df["Close"]).sum()
    if bad_hc > 0:
        issues.append(f"{symbol}: High < Close in {bad_hc} bars")
    bad_lo = (df["Low"] > df["Open"]).sum()
    if bad_lo > 0:
        issues.append(f"{symbol}: Low > Open in {bad_lo} bars")
    bad_lc = (df["Low"] > df["Close"]).sum()
    if bad_lc > 0:
        issues.append(f"{symbol}: Low > Close in {bad_lc} bars")
    if "Volume" in df.columns and (df["Volume"] < 0).any():
        if canonical_stem(symbol) not in FX_VOLUME_SKIP_ASSETS:
            issues.append(f"{symbol}: negative Volume present")
    # Placeholder / forward-fill detection: if a vast majority of bars have
    # O==H==L==C the file is not real 5m OHLC (typical IB CONTFUT pre-2024
    # behaviour, or other vendor placeholder snapshots). Refuse to use it
    # rather than silently produce degenerate GMM fits downstream.
    if len(df) > 0:
        ohlc_flat = (
            (df["Open"] == df["High"])
            & (df["High"] == df["Low"])
            & (df["Low"] == df["Close"])
        )
        flat_share = float(ohlc_flat.mean())
        if flat_share > 0.5:
            issues.append(
                f"{symbol}: {flat_share*100:.1f}% of bars have O=H=L=C "
                f"(placeholder / forward-fill data, not usable for "
                f"cross-frequency regime analysis)"
            )
        # Hour-bucket variant: even when the panel-wide flat share stays
        # under the 50% gate, individual hour-of-day buckets can be
        # entirely forward-fill (typical for ETFs whose extended-hours
        # 5m bars echo the last RTH print). 2022 GLD shows 79-88% flat
        # in 04:00-07:00 and 16:00-19:00 while the daily mean is only
        # 38%, well below the panel gate. That data is unusable for
        # fine-frequency regime work because log-vol collapses to a
        # delta inside those hours -- the exact input that triggers
        # ``pct_fallback`` in ``core.models.fit_regime``, so the 5m /
        # 15m labels stop being GMM-derived. Flag any hour bucket with
        # >70% flat bars provided it has at least one trading day worth
        # of evidence (~78 bars, the minimum BARS_PER_DAY value).
        idx = df.index
        if idx.tz is None:
            hours = idx.hour
        else:
            hours = idx.tz_convert("America/New_York").hour
        flat_by_hour = pd.Series(ohlc_flat.values).groupby(hours).agg(["mean", "count"])
        bad_hours = flat_by_hour[
            (flat_by_hour["mean"] > 0.7) & (flat_by_hour["count"] >= 78)
        ]
        if not bad_hours.empty:
            offenders = ", ".join(
                f"{int(h):02d}:00 ({row['mean']*100:.0f}% flat, n={int(row['count'])})"
                for h, row in bad_hours.iterrows()
            )
            issues.append(
                f"{symbol}: hour-of-day buckets with >70% flat O=H=L=C: "
                f"{offenders} (likely extended-hours forward-fill; fine-freq "
                f"GMM will fall back to percentile threshold)"
            )
    # index
    if not df.index.is_monotonic_increasing:
        issues.append(f"{symbol}: index not sorted ascending")
    dup = df.index.duplicated().sum()
    if dup > 0:
        issues.append(f"{symbol}: {dup} duplicate timestamps")
    return issues


def validate_raw_5m_dir(raw_dir: Path | str, assets: list[str]) -> dict[str, list[str]]:
    """
    Validate all *_5m.csv in raw_dir for given assets. Returns {symbol: list of issue messages}.
    Uses canonical_stem for file names (^GSPC -> GSPC, CL=F -> CLF, USDJPY=X -> USDJPYX);
    pass bare ``CL`` / ``USDJPY`` if you want the file stems to match the bare symbol.
    """
    raw_dir = Path(raw_dir)
    result: dict[str, list[str]] = {}
    for symbol in assets:
        path = raw_dir / f"{canonical_stem(symbol)}_5m.csv"
        if not path.exists():
            result[symbol] = [f"file not found: {path}"]
            continue
        try:
            df = load_5m_ohlc(path)
            result[symbol] = validate_5m_ohlc(df, symbol)
            if not result[symbol]:
                n = len(df)
                start, end = df.index.min(), df.index.max()
                result[symbol] = []  # keep empty; log summary in runner
                logger.info("  %s: OK, %d bars, %s ~ %s", symbol, n, start, end)
        except Exception as e:
            result[symbol] = [f"load/validate error: {e}"]
    return result
