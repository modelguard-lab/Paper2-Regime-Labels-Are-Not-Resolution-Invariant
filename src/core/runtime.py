"""
Logging configuration for Paper 2 (Resolution-Invariant).

Console + optional global file handler, plus Windows-specific UTF-8
console wrapping and PowerShell Tee-Object cleanup.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from pathlib import Path

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def set_thread_env_defaults(n_threads: int = 1) -> None:
    """Set BLAS / OMP thread-count environment variables before numpy import.

    Must be called before any sklearn / numpy / scipy import to suppress the
    MKL memory-leak warning on Windows (K=2 GMM on 1-D features; BLAS
    parallelism beyond 1 thread is not load-bearing).  ``os.environ.setdefault``
    is used so an explicit caller-set value is never overridden.
    """
    for var in _BLAS_THREAD_VARS:
        os.environ.setdefault(var, str(n_threads))


def configure_console_logging(level: int = logging.INFO) -> None:
    """
    Ensure logs are visible in the console.

    Only adds a StreamHandler if the root logger has no handlers yet.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def configure_global_file_logging(log_path: Path, level: int = logging.INFO) -> None:
    """
    Append a single global FileHandler so all logs are also written to run.log.

    Unlike basicConfig, this adds a FileHandler even if logging was already configured.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    existing = []
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                existing.append(Path(h.baseFilename).resolve())
            except Exception:
                logging.getLogger(__name__).warning(
                    "runtime._dedupe: could not resolve handler basefile path"
                )
                continue
    if log_path.resolve() not in existing:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root.addHandler(fh)

    if root.level > level:
        root.setLevel(level)

    logging.captureWarnings(True)


_utf8_done = False
_utf8_lock = threading.Lock()


def ensure_utf8_console() -> None:
    """Force UTF-8 on Windows console streams (idempotent).

    Thread-safety: a module-level ``threading.Lock`` guards the
    double-checked ``_utf8_done`` flag so that multiple workers calling
    ``ensure_utf8_console()`` concurrently (e.g., a ``ProcessPoolExecutor``
    spawned in-process or a ``ThreadPoolExecutor`` worker pool) cannot race
    and wrap ``sys.stdout`` twice. The first-check before acquiring the
    lock keeps the steady-state cost at one boolean read.
    """
    global _utf8_done
    if _utf8_done:
        return
    with _utf8_lock:
        if _utf8_done:
            return
        if sys.platform == "win32":
            for stream in ("stdout", "stderr"):
                current = getattr(sys, stream)
                if hasattr(current, "buffer") and (getattr(current, "encoding", "") or "").lower() not in ("utf-8", "utf-8-sig"):
                    setattr(sys, stream, io.TextIOWrapper(current.buffer, encoding="utf-8", errors="replace"))
        _utf8_done = True
