"""Summarise calm vs event cross-frequency ARI per asset, model, and episode.

Reads the per-asset / per-window ``cross_freq_ari.csv`` outputs produced by
the main pipeline (in both ``outputs/`` for 2026 and ``outputs_2022/`` for
the 2022 OOS replication) and writes a long table of mean off-diagonal ARI
by (episode, asset, model, window). Uses the canonical ``mean_offdiag_ari``
from ``core/metrics.py`` rather than a local re-implementation.

Output:
  - ``outputs/window_ari_summary.csv``
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import project_layout
from ..core.config import FREQS
from ..core.metrics import mean_offdiag_ari
from ..data.data_ib import canonical_stem

logger = logging.getLogger(__name__)


def main(project_dir: Path | None = None) -> None:
    layout = project_layout(project_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Each entry: (episode_label, outputs_dir).
    episodes: list[tuple[str, Path]] = [("2026", layout.outputs_dir)]
    if layout.outputs_dir_2022.exists():
        episodes.append(("2022", layout.outputs_dir_2022))

    rows: list[dict[str, object]] = []
    for episode, outputs_dir in episodes:
        for symbol in layout.assets:
            stem = canonical_stem(symbol)
            for model, prefix in [("GMM", ""), ("HMM", "hmm_")]:
                for window in ["calm", "event"]:
                    path = outputs_dir / f"{stem}_{prefix}{window}_cross_freq_ari.csv"
                    if not path.exists():
                        continue
                    ari = pd.read_csv(path, index_col=0)
                    expected = set(FREQS)
                    got = set(ari.columns)
                    if got != expected:
                        import warnings as _w
                        _w.warn(f"exp_08: skipping {path.name} - columns {got} != FREQS {expected}", RuntimeWarning)
                        continue
                    val = mean_offdiag_ari(ari)
                    rows.append(
                        {
                            "episode": episode,
                            "asset": stem,
                            "model": model,
                            "window": window,
                            "mean_offdiag_ari": float("nan") if val is None else val,
                        }
                    )

    if not rows:
        msg = (
            "No <asset>_{calm,event}_cross_freq_ari.csv files found in any "
            "episode directory. Run the main pipeline first: "
            "`python run.py pipeline` and (optionally) "
            "`python run.py pipeline --episode 2022_ukraine`."
        )
        logger.error(msg)
        raise FileNotFoundError("exp_08: no per-window cross-freq ARI CSVs found in outputs_dir")

    df = pd.DataFrame(rows)
    out_path = layout.outputs_dir / "window_ari_summary.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved window-ARI summary: %s (%d rows)", out_path, len(df))

    per_asset = df.pivot(
        index=["episode", "asset", "model"],
        columns="window",
        values="mean_offdiag_ari",
    ).round(3)
    pooled = (
        df.groupby(["episode", "model", "window"])["mean_offdiag_ari"]
        .mean()
        .unstack()
        .round(3)
    )
    logger.info("Per-asset mean off-diagonal ARI (cross-frequency):\n%s", per_asset)
    logger.info("Pooled mean across assets (per episode):\n%s", pooled)


if __name__ == "__main__":
    main()
