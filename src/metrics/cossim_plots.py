"""Boxplot of cosine-similarity reconstruction results — shared between
reference_aligners/mapping_analysis and analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from .cossim import CossimResult

logger = logging.getLogger(__name__)


def plot_cossim_boxplots(
    cossim_results: dict[str, CossimResult], output_path: Path
) -> None:
    """
    Two-panel boxplot (gene-wise, spot-wise) of the full cosine-similarity
    distributions for the soft/hard x raw/norm reconstruction combos.

    Boxes are ordered raw-soft, raw-hard, norm-soft, norm-hard within each
    panel, so the soft/hard pair for each (raw|norm) group sits side by side.

    Args:
        cossim_results: label ("soft-raw"/"hard-raw"/"soft-norm"/"hard-norm")
                        -> CossimResult, as returned by compute_and_save_cossim.
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
