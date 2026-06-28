"""Cross-asset resonance figure: rolling 7-day cross-resolution ARI overlay.

Reads each asset's ``{stem}_rolling_7d_ari.csv`` from the per-episode
outputs directory and overlays the ``mean_offdiag_ari`` traces on a single
axis. Produced for both 2026 and 2022 episodes (separate output dirs).
Headless safe (matplotlib Agg backend).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..experiments import project_layout
from ..core.config import CROSS_ASSET_RESONANCE_FAILSAFE_ARI

logger = logging.getLogger(__name__)


# Per-asset display labels and line colours kept stable across episodes so
# the supplementary figure is comparable. Assets not in this map fall back
# to their canonical stem and a default colour cycle.
_ASSET_STYLE: dict[str, dict[str, str]] = {
    "SPY":    {"label": "SPY (Equities)",    "color": "#1f77b4"},
    "QQQ":    {"label": "QQQ (Tech Equity)", "color": "#ff7f0e"},
    "CL":     {"label": "WTI Crude (Energy)", "color": "#d62728"},
    "USDJPY": {"label": "USD/JPY (FX)",       "color": "#2ca02c"},
    "GLD":    {"label": "GLD (Gold)",          "color": "#9467bd"},
}


def _plot_episode(outputs_dir: Path, episode_label: str, assets: list[str]) -> None:
    from ._mpl import use_headless_backend

    use_headless_backend()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns

    from ..data.data_ib import canonical_stem

    if not outputs_dir.exists():
        logger.info("Skip cross-asset resonance %s: outputs dir %s missing",
                    episode_label, outputs_dir)
        return

    series: list[tuple[str, pd.DataFrame]] = []
    for sym in assets:
        stem = canonical_stem(sym)
        path = outputs_dir / f"{stem}_rolling_7d_ari.csv"
        if not path.exists():
            logger.info("Skip %s for %s: %s not found", stem, episode_label, path)
            continue
        df = pd.read_csv(path)
        if df.empty or "window_end" not in df.columns or "mean_offdiag_ari" not in df.columns:
            logger.info("Skip %s for %s: empty or missing columns", stem, episode_label)
            continue
        ts_col = pd.to_datetime(df["window_end"])
        if ts_col.dt.tz is not None:
            ts_col = ts_col.dt.tz_convert("America/New_York").dt.tz_localize(None)
        df["window_end"] = ts_col
        series.append((stem, df))

    if not series:
        logger.warning("No rolling-7d ARI series found for %s; skipping figure",
                       episode_label)
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        for stem, df in series:
            style = _ASSET_STYLE.get(stem, {})
            ax.plot(
                df["window_end"],
                df["mean_offdiag_ari"],
                label=style.get("label", stem),
                color=style.get("color"),
                linewidth=2,
            )

        ax.axhline(
            y=CROSS_ASSET_RESONANCE_FAILSAFE_ARI,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Indicative monitoring threshold (ARI={CROSS_ASSET_RESONANCE_FAILSAFE_ARI}; see Suppl S.3)",
        )
        ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        ax.set_title(
            f"Cross-Asset Resonance: Rolling 7-Day Cross-Resolution ARI ({episode_label})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Window End Date", fontsize=12)
        ax.set_ylabel("Mean Off-Diagonal ARI (Cross-Frequency Agreement)", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()

        out_path = outputs_dir / "cross_asset_resonance.png"
        fig.savefig(out_path, dpi=300)
        logger.info("Saved cross-asset resonance figure to %s", out_path)
    finally:
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    layout = project_layout()
    episodes = [
        (layout.outputs_dir, "Jan-Mar 2026", layout.assets),
        (layout.outputs_dir_2022, "Jan-Feb 2022", layout.assets_2022),
    ]
    for outputs_dir, episode_label, assets in episodes:
        _plot_episode(outputs_dir, episode_label, list(assets))


if __name__ == "__main__":
    main()
