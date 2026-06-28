"""Interactive Brokers contract factories.

Lazy adapter around ``ib_insync`` so :mod:`data_ib` (the I/O orchestrator)
remains importable on machines without the IB SDK. Each factory calls
:func:`_require_ib` which raises a clear ImportError only when an IB
download is actually requested; pure CSV-analysis code paths never trip
it.

Internal to :mod:`src.data`; not part of the public surface.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _require_ib() -> tuple[Any, ...]:
    """Import ``ib_insync`` lazily.

    Keeps the module importable for users who only run analysis on existing
    CSVs, while still providing a clear error when IB download functionality
    is invoked.
    """
    try:
        from ib_insync import IB, Contract, Forex, Stock, Future, Index, ContFuture
        return IB, Contract, Forex, Stock, Future, Index, ContFuture
    except ImportError:
        raise ImportError(
            "IB download requires ib_insync. Install with: pip install ib_insync "
            "(or pip install -r requirements.txt)"
        ) from None


# --- Contract mapping (EXPERIMENT_DESIGN: ^GSPC, CL=F, USD/JPY) ---


def _contract_stock(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Any:
    _, _, _, Stock, _, _, _ = _require_ib()
    return Stock(symbol, exchange, currency)


def _contract_index(symbol: str, exchange: str = "CBOE", currency: str = "USD") -> Any:
    """S&P 500 index: IB uses symbol SPX (not GSPC). Exchange CBOE."""
    _, _, _, _, _, Index, _ = _require_ib()
    return Index(symbol, exchange, currency)


def _contract_forex(pair: str, exchange: str = "IDEALPRO") -> Any:
    """e.g. 'USDJPY' or 'USDJPY=X' -> Forex('USDJPY', 'IDEALPRO'). IDEALPRO required for FX historical."""
    _, _, Forex, _, _, _, _ = _require_ib()
    base = pair.upper().replace("=X", "").replace("/", "").replace(".", "")
    if len(base) != 6:
        raise ValueError(f"Forex pair must be 6 chars (e.g. USDJPY), got {pair!r}")
    return Forex(base, exchange)


def _contract_future(symbol: str, exchange: str = "NYMEX", currency: str = "USD", expiry: Optional[str] = None) -> Any:
    """e.g. CL with expiry '202612' for Dec 2026. If expiry is None, use a placeholder (IB may reject)."""
    _, _, _, _, Future, _, _ = _require_ib()
    if expiry is None:
        # Default: next December for CL (common for oil). Override in config.
        now = datetime.now()
        y = now.year
        expiry = f"{y}12" if now.month < 12 else f"{y+1}12"
        logger.warning("_contract_future: no future_expiry for %s, using default %s", symbol, expiry)
    return Future(symbol, exchange=exchange, currency=currency, lastTradeDateOrContractMonth=expiry)


def _contract_cont_future(symbol: str, exchange: str = "NYMEX", currency: str = "USD") -> Any:
    """e.g. CL continuous front-month futures via IB ContFuture."""
    _, _, _, _, _, _, ContFuture = _require_ib()
    return ContFuture(symbol, exchange=exchange, currency=currency)


def _is_cont_future_marker(value: Optional[str]) -> bool:
    """Whether config asks for IB continuous front-month futures."""
    if value is None:
        return False
    return str(value).strip().upper() in {"CONTFUT", "CONT_FUT", "CONTINUOUS", "CONT"}
