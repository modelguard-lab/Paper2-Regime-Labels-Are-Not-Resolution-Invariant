"""
Paper 2 -- Regime Labels Are Not Resolution-Invariant: unified entry point.

Usage
-----
  python run.py pipeline                    # Full multi-frequency pipeline (default)
  python run.py extended                    # Full extended-analyses sweep (steps 1-6)
  python run.py extended_majority_vote      # Step 1 only
  python run.py extended_bootstrap          # Step 2 only
  python run.py extended_hypothesis_tests   # Step 3 only
  python run.py extended_simulation         # Step 4 only
  python run.py extended_calm_subsample     # Step 5 only
  python run.py extended_var_uplift         # Step 6 only
  python run.py stress_vs_calm              # Formal stress-vs-calm ARI test
  python run.py cross_asset                 # Cross-asset resonance figure
  python run.py summarize                   # Summarize window results
  python run.py all                         # Pipeline + extended + standalone

  # Pipeline options (passed through to pipeline):
  python run.py pipeline --download         # Download data only
  python run.py pipeline --validate         # Validate data only
  python run.py pipeline --raw-dir DIR
"""

from __future__ import annotations

from src.core.runtime import configure_console_logging, configure_global_file_logging, set_thread_env_defaults

# Must be first: set BLAS/OMP thread limits before any numpy/sklearn import.
set_thread_env_defaults()

import argparse
import logging
import sys
from pathlib import Path

from src.commands.cli_main import run
from src.commands.cli_registry import COMMANDS, run_module_command

PROJECT_DIR = Path(__file__).resolve().parent

ALL_EXPERIMENT_COMMANDS: tuple[str, ...] = (
    "pipeline",
    "extended",
    "stress_vs_calm",
    "cross_asset",
    "summarize",
)


def _print_help() -> None:
    text = (__doc__ or "").strip()
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode("ascii", errors="replace").decode("ascii")
        print(safe)


def _parse_pipeline_args() -> dict:
    """Parse pipeline-specific args for backward compatibility."""
    parser = argparse.ArgumentParser(description="Paper 2 pipeline", add_help=False)
    parser.add_argument("config", nargs="?", help="Path to config YAML")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--download", action="store_true", help="Download data only and exit")
    mode_group.add_argument("--validate", action="store_true", help="Validate existing data only and exit")
    parser.add_argument("--raw-dir", dest="raw_dir")
    parser.add_argument("--outputs-dir", dest="outputs_dir")
    # No choices= here so new EPISODES entries become CLI-selectable without
    # editing run.py. pipeline.run() raises a descriptive ValueError on
    # unknown episode keys.
    parser.add_argument("--episode")
    parser.add_argument("--start", dest="start_date")
    parser.add_argument("--end", dest="end_date")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--client-id", dest="client_id", type=int)
    parser.add_argument("--symbols", nargs="+")
    args = parser.parse_args(sys.argv[2:])
    overrides = {}
    if args.download:
        overrides["download_only"] = True
    if args.validate:
        overrides["validate_only"] = True
    for key in ("raw_dir", "outputs_dir", "episode", "start_date", "end_date", "host"):
        val = getattr(args, key, None)
        if val:
            overrides[key] = val
    if args.port is not None:
        overrides["port"] = args.port
    if args.client_id is not None:
        overrides["client_id"] = args.client_id
    if args.symbols:
        overrides["symbols"] = args.symbols
    config = args.config
    return {"config": config, "overrides": overrides or None}


def _run_pipeline() -> None:
    configure_console_logging()
    parsed = _parse_pipeline_args()
    config_path = Path(parsed["config"]) if parsed["config"] else PROJECT_DIR / "config.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    run(config_path, overrides=parsed["overrides"])


def _run_module(name: str) -> None:
    configure_console_logging()
    run_module_command(name)


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "pipeline"

    if cmd in ("-h", "--help", "help"):
        _print_help()
        return

    if cmd == "all":
        configure_console_logging()
        configure_global_file_logging(PROJECT_DIR / "outputs" / "run.log")
        _logger = logging.getLogger("run")
        _run_pipeline()
        for name in ALL_EXPERIMENT_COMMANDS:
            if name == "pipeline":
                continue
            if name not in COMMANDS:
                _logger.warning("Skip missing command: %s", name)
                continue
            _logger.info("=" * 60)
            _logger.info("  %s", name)
            _logger.info("=" * 60)
            _run_module(name)
        return

    if cmd == "pipeline":
        _run_pipeline()
        return

    if cmd in COMMANDS:
        _run_module(cmd)
        return

    raise SystemExit(f"Unknown command: {cmd}. Use --help to list commands.")


if __name__ == "__main__":
    main()
