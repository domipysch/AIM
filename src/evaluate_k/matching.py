"""Contingency matrix argmax matching."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Contingency-matrix argmax matching
# ──────────────────────────────────────────────────────────────────────────────


def compute_contingency_matching(
    cell_states: np.ndarray,
    leiden_labels: np.ndarray,
) -> dict:
    """
    Argmax matching on the contingency matrix.

    Builds a (n_computed × n_leiden) count matrix.  For the finer clustering
    (more clusters) each cluster is assigned to the argmax of the coarser
    clustering — many-to-one on the coarse side is allowed.

    Score = matched cells / total cells  (0–1, higher is better).
    """
    unique_computed = [int(x) for x in sorted(np.unique(cell_states))]
    unique_leiden = [int(x) for x in sorted(np.unique(leiden_labels))]
    K = len(unique_computed)
    L = len(unique_leiden)

    ct = np.zeros((K, L), dtype=np.int64)
    for i, k in enumerate(unique_computed):
        for j, l in enumerate(unique_leiden):
            ct[i, j] = int(((cell_states == k) & (leiden_labels == l)).sum())

    total = int(ct.sum())

    if L >= K:
        best_computed_per_leiden = ct.argmax(axis=0)  # (L,)
        matched = int(sum(ct[best_computed_per_leiden[j], j] for j in range(L)))
        best_leiden_per_computed = None
    else:
        best_leiden_per_computed = ct.argmax(axis=1)  # (K,)
        matched = int(sum(ct[i, best_leiden_per_computed[i]] for i in range(K)))
        best_computed_per_leiden = None

    score = matched / total if total > 0 else float("nan")
    logger.info(
        "Contingency matching: K=%d computed, L=%d leiden | "
        "matched=%d / %d cells | score=%.3f",
        K,
        L,
        matched,
        total,
        score,
    )
    return {
        "contingency_matrix": ct,
        "score": score,
        "matched_cells": matched,
        "total_cells": total,
        "n_computed": K,
        "n_leiden": L,
        "unique_computed": unique_computed,
        "unique_leiden": unique_leiden,
        "best_computed_per_leiden": best_computed_per_leiden,
        "best_leiden_per_computed": best_leiden_per_computed,
    }


def plot_contingency_heatmap(
    match_results: dict,
    output_path: Path,
    spot_fractions: dict[int, float] | None = None,
    cell_fractions: dict[int, float] | None = None,
) -> None:
    """
    Heatmap of the contingency matrix (counts, log-scaled colour), with the
    argmax assignment for each fine-side cluster marked by a star.

    When `cell_fractions`/`spot_fractions` are supplied, only computed states
    with cell- OR spot-fraction above the display threshold are shown as rows
    (consistent with the other report plots); the score in the title is still
    the global value over all states.
    """
    from .plots import select_displayed_states

    ct = match_results["contingency_matrix"]
    score = match_results["score"]
    K, L = ct.shape
    unique_computed = match_results.get("unique_computed", list(range(K)))
    unique_leiden = match_results.get("unique_leiden", list(range(L)))

    # Per-computed-state fractions over the *global* total (before any row drop)
    total = ct.sum()
    cell_fracs_computed = ct.sum(axis=1) / total if total > 0 else np.zeros(K)
    cell_fracs_leiden = ct.sum(axis=0) / total if total > 0 else np.zeros(L)

    if match_results["best_computed_per_leiden"] is not None:
        marker_rows = np.asarray(match_results["best_computed_per_leiden"])  # (L,)
        marker_cols = np.arange(L)
    else:
        marker_rows = np.arange(K)
        marker_cols = np.asarray(match_results["best_leiden_per_computed"])  # (K,)

    # ── Restrict displayed rows to the OR-active set of computed states ───────
    keep = list(range(K))
    if cell_fractions is not None or spot_fractions is not None:
        active = set(
            select_displayed_states(unique_computed, cell_fractions, spot_fractions)
        )
        keep = [i for i, s in enumerate(unique_computed) if s in active] or list(
            range(K)
        )

    if len(keep) != K:
        pos_map = {orig: new for new, orig in enumerate(keep)}
        ct = ct[keep, :]
        unique_computed = [unique_computed[i] for i in keep]
        cell_fracs_computed = cell_fracs_computed[keep]
        # Remap argmax markers to the surviving rows; drop markers on hidden rows
        new_rows, new_cols = [], []
        for r, c in zip(marker_rows.tolist(), marker_cols.tolist()):
            if r in pos_map:
                new_rows.append(pos_map[r])
                new_cols.append(c)
        marker_rows = np.asarray(new_rows, dtype=int)
        marker_cols = np.asarray(new_cols, dtype=int)
        K = len(keep)

    fig, ax = plt.subplots(figsize=(max(6, L * 0.6 + 1), max(4, K * 0.6 + 1)))

    pos_vals = ct[ct > 0]
    vmin = float(pos_vals.min()) if len(pos_vals) else 1.0
    norm = mcolors.LogNorm(vmin=vmin, vmax=max(float(ct.max()), vmin))
    im = ax.imshow(ct, aspect="auto", cmap="YlOrRd", norm=norm)
    fig.colorbar(im, ax=ax, label="Cell count (log scale)")

    threshold = ct.max() * 0.5
    for i in range(K):
        for j in range(L):
            val = int(ct[i, j])
            if val == 0:
                continue
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if ct[i, j] > threshold else "black",
            )

    ax.scatter(
        marker_cols,
        marker_rows,
        marker="*",
        s=70,
        color="royalblue",
        zorder=5,
        label=f"Argmax match  (score={score:.3f})",
    )
    ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("Leiden cluster", fontsize=11)
    ax.set_ylabel("Computed state", fontsize=11)
    ax.set_title(
        f"Contingency matrix: Computed states × Leiden clusters\n"
        f"Score = {score:.3f}  "
        f"({match_results['matched_cells']} / {match_results['total_cells']} cells matched)",
        fontsize=11,
    )

    ax.set_xticks(range(L))
    ax.set_xticklabels(
        [f"L{l}\n{cell_fracs_leiden[j]:.1%}" for j, l in enumerate(unique_leiden)],
        fontsize=7,
    )

    y_labels = []
    for i, state_id in enumerate(unique_computed):
        parts = [f"S{state_id}", f"c:{cell_fracs_computed[i]:.1%}"]
        if spot_fractions is not None:
            parts.append(f"s:{spot_fractions.get(state_id, 0):.1%}")
        y_labels.append("\n".join(parts))
    ax.set_yticks(range(K))
    ax.set_yticklabels(y_labels, fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Contingency heatmap → %s", output_path)
