"""Rolling ARI line plot.

Plots the mean off-diagonal ARI per rolling window plus each pairwise ARI
series across the same windows. Used as a per-asset robustness figure for
the paper. Headless safe (Agg backend).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_rolling_ari(
    rolling_df: pd.DataFrame,
    rolling_pair_df: pd.DataFrame,
    asset_name: str,
    out_path: Path,
) -> None:
    """Render the rolling-ARI line plot to ``out_path``.

    ``rolling_df`` must carry columns ``window_end`` and ``mean_offdiag_ari``.
    ``rolling_pair_df`` must carry ``window_end``, ``freq_a``, ``freq_b``,
    ``ari`` so that one line per frequency pair can be drawn alongside the
    headline mean.
    """
    from ._mpl import use_headless_backend

    use_headless_backend()
    import matplotlib.pyplot as plt

    if rolling_df.empty:
        return

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = pd.to_datetime(rolling_df["window_end"])
        ax.plot(x, rolling_df["mean_offdiag_ari"], color="black", linewidth=2, label="mean off-diag")

        if "freq_a" in rolling_pair_df.columns and "freq_b" in rolling_pair_df.columns:
            for (fa, fb), grp in rolling_pair_df.groupby(["freq_a", "freq_b"]):
                ax.plot(
                    pd.to_datetime(grp["window_end"]),
                    grp["ari"],
                    alpha=0.6,
                    linewidth=1.2,
                    label=f"{fa}-{fb}",
                )

        ax.set_title(f"{asset_name}: Rolling ARI")
        ax.set_ylabel("ARI")
        ax.set_xlabel("Window end date")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    finally:
        if fig is not None:
            plt.close(fig)
