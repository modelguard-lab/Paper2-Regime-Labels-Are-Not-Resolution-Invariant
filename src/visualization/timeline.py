"""Gantt-style timeline plot of regime labels by frequency.

Renders four stacked rows (1d / 1h / 15m / 5m) where each timestamp is
shaded red for crisis labels and sky-blue for calm. Used as the per-asset
headline figure in the paper. Headless safe (Agg backend).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_timeline(
    aligned: dict[str, pd.Series],
    asset_name: str,
    out_path: Path,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Render the per-asset crisis-vs-calm timeline to ``out_path``.

    Parameters
    ----------
    aligned : dict[str, pd.Series]
        Per-frequency regime label series (0=calm, 1=crisis).
    asset_name : str
        Title prefix.
    out_path : Path
        Output PNG path.
    figsize : tuple[float, float], optional
        Figure size. Default (12, 6) gives ~1.5 inches per row, more
        readable than the previous (12, 4) default which crammed four
        rows into a strip.
    """
    from ._mpl import use_headless_backend

    use_headless_backend()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(
        4, 1, figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1]},
    )
    order = ("1d", "1h", "15m", "5m")

    # Build the common axis as the union of all four frequency indices so
    # short coarser series do not silently truncate the x-range when the
    # 5m series is missing or short.
    indices = [s.index for s in aligned.values() if s is not None and not s.empty]
    if indices:
        t_common = indices[0]
        for idx in indices[1:]:
            t_common = t_common.union(idx)
    else:
        t_common = pd.DatetimeIndex([])

    for i, (ax, freq) in enumerate(zip(axes, order)):
        s = aligned.get(freq)
        if s is None or s.empty or not len(t_common):
            ax.set_ylabel(freq)
            ax.set_yticks([])
            ax.annotate(
                "no data",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            continue
        s = s.reindex(t_common).ffill()
        if not s.empty:
            first_valid = s.first_valid_index()
            if first_valid is not None:
                s.loc[s.index < first_valid] = np.nan
        if not s.notna().any():
            ax.set_ylabel(freq)
            ax.set_yticks([])
            ax.annotate(
                "no data",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            continue
        t = s.index
        crisis = (s == 1).values
        calm = (s == 0).values
        ax.fill_between(t, 0, 1, where=crisis, color="darkred", alpha=0.8, step="post")
        ax.fill_between(t, 0, 1, where=calm, color="skyblue", alpha=0.6, step="post")
        ax.set_ylabel(freq, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.yaxis.set_label_position("right")

    axes[0].set_title(f"{asset_name}: regime by frequency (red = crisis)")
    legend_handles = [
        Patch(facecolor="darkred", alpha=0.8, label="crisis"),
        Patch(facecolor="skyblue", alpha=0.6, label="calm"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)
    try:
        tz = t_common.tz
    except Exception:
        tz = None
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=tz))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
