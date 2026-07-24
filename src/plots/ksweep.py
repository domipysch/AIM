"""Cross-K comparison figure for the AIM sweep: one 2x2 panel of key analysis
metrics as a function of K (number of states)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)

_SELF_Z_COLOR = "#3b6ea5"
_PURITY_Z_COLOR = "#c0504d"
_SPOT_COLOR = "steelblue"
_GENE_COLOR = "seagreen"
_MODULARITY_COLOR = "darkorange"
_CONFIDENCE_COLOR = "#7b5ea7"


def plot_ksweep_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Render the 4-panel cross-K comparison, all panels sharing K on the x-axis.

    ``df`` has one row per K (ascending) with columns ``k``,
    ``cossim_hard_norm_spot``, ``cossim_hard_norm_gene``, ``cossim_hard_raw_spot``,
    ``cossim_hard_raw_gene``, ``nhood_mean_self_zscore``, ``local_purity_zscore``,
    ``modularity_shared``, ``mean_confidence``. NaNs (a metric missing for some K)
    simply leave a gap in the line.
    """
    k = df["k"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("K-sweep comparison", fontsize=15, weight="bold")

    # Reconstruction cosine similarity: per-spot and per-gene share one y-axis.
    ax = axes[0, 0]
    ax.plot(
        k, df["cossim_hard_norm_spot"], marker="o", color=_SPOT_COLOR, label="per spot"
    )
    ax.plot(
        k, df["cossim_hard_norm_gene"], marker="o", color=_GENE_COLOR, label="per gene"
    )
    ax.set_title("Reconstruction cosine similarity — normalized (hard)")
    ax.set_ylabel("median cosine similarity")
    ax.legend(fontsize=9, loc="best")

    ax = axes[0, 1]
    ax.plot(
        k, df["cossim_hard_raw_spot"], marker="o", color=_SPOT_COLOR, label="per spot"
    )
    ax.plot(
        k, df["cossim_hard_raw_gene"], marker="o", color=_GENE_COLOR, label="per gene"
    )
    ax.set_title("Reconstruction cosine similarity — raw (hard)")
    ax.set_ylabel("median cosine similarity")
    ax.legend(fontsize=9, loc="best")

    # Spatial organisation: two metrics on different scales -> twin y-axes.
    ax = axes[1, 0]
    (l_self,) = ax.plot(
        k,
        df["nhood_mean_self_zscore"],
        marker="o",
        color=_SELF_Z_COLOR,
        label="nhood enrichment mean self z-score",
    )
    ax.set_ylabel("nhood enrichment mean self z-score", color=_SELF_Z_COLOR)
    ax.tick_params(axis="y", labelcolor=_SELF_Z_COLOR)
    ax_purity = ax.twinx()
    (l_purity,) = ax_purity.plot(
        k,
        df["local_purity_zscore"],
        marker="s",
        color=_PURITY_Z_COLOR,
        label="local spatial purity z-score",
    )
    ax_purity.set_ylabel("local spatial purity z-score", color=_PURITY_Z_COLOR)
    ax_purity.tick_params(axis="y", labelcolor=_PURITY_Z_COLOR)
    ax.set_title("Spatial organisation of mapped spots")
    ax.legend(handles=[l_self, l_purity], fontsize=9, loc="best")

    # Mapping modularity + mean confidence on different scales -> twin y-axes. The
    # confidence axis is added only when the mapper defined a confidence.
    ax = axes[1, 1]
    (l_mod,) = ax.plot(
        k,
        df["modularity_shared"],
        marker="o",
        color=_MODULARITY_COLOR,
        label="modularity_shared",
    )
    ax.set_ylabel("modularity_shared (shared-gene graph)", color=_MODULARITY_COLOR)
    ax.tick_params(axis="y", labelcolor=_MODULARITY_COLOR)
    ax.set_title("Mapping modularity & confidence")
    handles = [l_mod]
    if df["mean_confidence"].notna().any():
        ax_conf = ax.twinx()
        (l_conf,) = ax_conf.plot(
            k,
            df["mean_confidence"],
            marker="s",
            color=_CONFIDENCE_COLOR,
            label="mean mapping confidence",
        )
        ax_conf.set_ylabel("mean mapping confidence", color=_CONFIDENCE_COLOR)
        ax_conf.tick_params(axis="y", labelcolor=_CONFIDENCE_COLOR)
        handles.append(l_conf)
    ax.legend(handles=handles, fontsize=9, loc="best")

    for ax in axes.ravel():
        ax.set_xlabel("K (number of states)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("K-sweep comparison figure → %s", output_path)
