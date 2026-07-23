"""Plots for metrics.onehot — shared between reference_aligners/mapping_analysis
and analysis so both pipelines' "how one-hot is this mapping" figures match."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics.onehot import DOMINANCE_THRESHOLDS

logger = logging.getLogger(__name__)


def plot_onehot_distribution(
    metrics: dict, output_path: Path, row_label: str = "spot"
) -> None:
    """Histogram of per-row max-probability ("how one-hot"), annotated with
    mean/median and the Gini-impurity / entropy summary stats."""
    max_prob = metrics["max_prob"]
    summary = metrics["summary"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(max_prob, bins=40, color="steelblue", alpha=0.85)
    ax.axvline(
        summary["max_prob"]["mean"],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"mean={summary['max_prob']['mean']:.3f}",
    )
    ax.axvline(
        summary["max_prob"]["median"],
        color="darkorange",
        linestyle=":",
        linewidth=1,
        label=f"median={summary['max_prob']['median']:.3f}",
    )
    ax.set_xlabel(f"Max probability per {row_label} (1.0 = fully one-hot)")
    ax.set_ylabel(f"Number of {row_label}s")
    ax.set_title(
        f"Mapping sharpness — {metrics['n_rows']} {row_label}s, {metrics['n_cols']} columns\n"
        f"Gini impurity mean={summary['gini_impurity']['mean']:.3f}, "
        f"entropy mean={summary['entropy']['mean']:.3f}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("One-hot distribution → %s", output_path)


def plot_dominance_thresholds(
    metrics: dict, output_path: Path, row_label: str = "spot"
) -> None:
    """
    Bar chart of the fraction of rows whose max-probability ("dominance") is
    >= each of a fixed set of thresholds.

    Mirrors src/00_Playground/assess_one_hotness.py's plot_report(), reusing
    its threshold set and styling.
    """
    max_prob = metrics["max_prob"]
    fracs = [(t, float(np.mean(max_prob >= t))) for t in DOMINANCE_THRESHOLDS]
    labels = [f"≥ {t:g}" for t, _ in fracs]
    vals = [f for _, f in fracs]

    ink = "#2b2b2b"
    accent = "#3b6ea5"
    fill = "#9ecae1"

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Fraction of {row_label}s with max value ≥ threshold  ({metrics['n_rows']} {row_label}s)",
        fontsize=12,
        color=ink,
    )
    bars = ax.bar(labels, vals, color=fill, edgecolor=accent, linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.annotate(
            f"{v:.1%}",
            (bar.get_x() + bar.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
            color=ink,
        )
    ax.set_xlabel("max-probability threshold")
    ax.set_ylabel(f"fraction of {row_label}s")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Dominance threshold fractions → %s", output_path)
