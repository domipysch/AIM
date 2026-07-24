"""Plot for the per-spot mapping-confidence distribution."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_confidence_distribution(
    confidence: np.ndarray, summary: dict, output_path: Path
) -> None:
    """Histogram of per-spot assignment confidence in [0, 1], annotated with mean/median.

    ``summary`` must supply "n_spots", "mean", and "median".
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(confidence, bins=40, range=(0.0, 1.0), color="seagreen", alpha=0.85)
    ax.axvline(
        summary["mean"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"mean={summary['mean']:.3f}",
    )
    ax.axvline(
        summary["median"],
        color="darkorange",
        linestyle=":",
        linewidth=1,
        label=f"median={summary['median']:.3f}",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Assignment confidence per spot (1.0 = most confident)")
    ax.set_ylabel("Number of spots")
    ax.set_title(f"Mapping confidence — {summary['n_spots']} spots")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confidence distribution → %s", output_path)
