"""Boxplot of cosine-similarity reconstruction results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from metrics.cossim import CossimResult

logger = logging.getLogger(__name__)


def plot_cossim_boxplots(
    cossim_results: dict[str, CossimResult], output_path: Path
) -> None:
    """Two-panel (gene-wise, spot-wise) boxplot of cosine-similarity distributions per soft/hard x raw/norm combo; saves to output_path.

    cossim_results maps each label ("soft-raw"/"hard-raw"/"soft-norm"/"hard-norm") to a CossimResult.
    """
    order = ["soft-raw", "hard-raw", "soft-norm", "hard-norm"]
    labels = [lbl for lbl in order if lbl in cossim_results]
    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, attr, title in [
        (axes[0], "per_gene", "Gene-wise cosine similarity"),
        (axes[1], "per_spot", "Spot-wise cosine similarity"),
    ]:
        data = [list(getattr(cossim_results[lbl], attr).values()) for lbl in labels]
        colors = [
            "steelblue" if lbl.startswith("soft") else "darkorange" for lbl in labels
        ]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Cosine similarity", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="grey", linewidth=0.6)

    fig.suptitle("Reconstruction cosine-similarity distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cossim boxplots → %s", output_path)
