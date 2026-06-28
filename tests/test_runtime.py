"""Smoke tests for ``src.core.runtime`` logging-setup helpers.

These ensure the three configure-logging helpers run without raising and
produce the expected handler wiring. Idempotency is the load-bearing
property: pipelines may call ``configure_console_logging`` more than once
if a sub-experiment also runs ``logging.basicConfig`` itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.core.runtime import (
    configure_console_logging,
    configure_global_file_logging,
    ensure_utf8_console,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Snapshot root handlers and level; restore after each test so handlers
    added by one test do not leak into the next.
    """
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    yield
    root.handlers = old_handlers
    root.level = old_level


def test_configure_console_logging_adds_stream_handler_once():
    root = logging.getLogger()
    root.handlers = []  # ensure clean
    configure_console_logging()
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    n_after_first = len(root.handlers)
    configure_console_logging()  # idempotent
    assert len(root.handlers) == n_after_first


def test_configure_global_file_logging_creates_log_file(tmp_path: Path):
    log_path = tmp_path / "subdir" / "run.log"
    configure_global_file_logging(log_path)
    logger = logging.getLogger("paper2.test")
    logger.warning("smoke test message")
    # File handler must have been added and the file created.
    assert log_path.exists()
    assert any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)


def test_configure_global_file_logging_dedupes_same_path(tmp_path: Path):
    log_path = tmp_path / "run.log"
    configure_global_file_logging(log_path)
    n_after_first = len([h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)])
    configure_global_file_logging(log_path)
    n_after_second = len([h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)])
    assert n_after_second == n_after_first


def test_ensure_utf8_console_is_idempotent():
    # Just smoke-test that the function is callable without error and that
    # multiple invocations don't raise. Actual stream wrapping behaviour
    # is platform-dependent and tested implicitly by the pipeline runs.
    ensure_utf8_console()
    ensure_utf8_console()
