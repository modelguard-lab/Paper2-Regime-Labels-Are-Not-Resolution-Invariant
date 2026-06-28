"""Matplotlib headless setup helper.

The visualization modules are imported during pipeline runs that may have
no display attached (CI, server, ``--validate`` smoke checks). Each plot
function calls :func:`use_headless_backend` before importing
``matplotlib.pyplot`` to force the non-interactive Agg backend.

Importing matplotlib at module top would lock the backend for the whole
process; deferring the import + backend selection to the plot function is
the established pattern in this package, so the helper preserves it.
"""

from __future__ import annotations


def use_headless_backend() -> None:
    """Force the Agg backend before any pyplot import."""
    import matplotlib
    matplotlib.use("Agg")
