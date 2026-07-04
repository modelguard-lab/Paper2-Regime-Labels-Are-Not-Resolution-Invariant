"""Paper 2 -- Regime Labels Are Not Resolution-Invariant: source code.

Subpackages:

- :mod:`src.commands`     -- CLI dispatcher (``cli_main``, ``cli_registry``).
- :mod:`src.core`         -- algorithmic primitives (features, models, metrics,
                              stability, aggregation, diagnostics, time_utils,
                              runtime, config).
- :mod:`src.data`         -- IO layer (Interactive Brokers download + 5m loader).
- :mod:`src.experiments`  -- one ``exp_NN_<name>`` module per paper experiment;
                              each exposes a ``main()`` CLI entry point.
- :mod:`src.visualization`-- figure builders (timeline, rolling ARI,
                              cross-asset resonance).
- :mod:`src.workflows`    -- orchestrators (``pipeline`` for the main run,
                              ``pipeline_ext`` for the extended sweep).

Nothing is re-exported at the package root; entry points are reached via the
CLI registry in :mod:`src.commands.cli_registry` (see ``run.py`` in the
project root).
"""
