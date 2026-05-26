"""
Supervised matching methods: centroid cosine (Hungarian + greedy),
contingency matrix argmax, and AUC-based iterative matching.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import _dense_X
from ....pre_check.metric import zscored_centroids as _zscored_centroids

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Centroid cosine matching (Hungarian + greedy)
# ──────────────────────────────────────────────────────────────────────────────


def compute_leiden_vs_computed_matching(
    adata_sc: AnnData,
    cell_states: np.ndarray,
    leiden_labels: np.ndarray,
    shared_genes: list[str],
) -> dict:
    """
    Match computed cell-state centroids against Leiden cluster centroids,
    both computed on shared marker genes only.

    Returns a dict with:
        sim_matrix               ndarray (n_leiden × K)
        hungarian_score          float
        hungarian_row_idx        ndarray  — Leiden cluster indices
        hungarian_col_idx        ndarray  — matched computed state indices
        greedy_score             float
        best_computed_per_leiden ndarray  — argmax computed state per Leiden cluster
        n_leiden / n_computed    int
    """
    available = [g for g in shared_genes if g in adata_sc.var_names]
    if len(available) < 2:
        raise ValueError("Too few shared genes for centroid matching.")

    X = _dense_X(adata_sc[:, available])

    C_computed = _zscored_centroids(X, cell_states)  # (K, G)
    C_leiden = _zscored_centroids(X, leiden_labels)  # (L, G)

    sim = cosine_similarity(C_leiden, C_computed)  # (L, K)

    row_idx, col_idx = linear_sum_assignment(-sim)
    hungarian_score = float(sim[row_idx, col_idx].mean())

    # Each computed state picks its best Leiden cluster independently
    n_computed_states = C_computed.shape[0]
    best_leiden_per_computed = sim.argmax(axis=0)  # (n_computed,)
    greedy_score = float(
        sim[best_leiden_per_computed, np.arange(n_computed_states)].mean()
    )

    logger.info(
        "Centroid matching: %d Leiden vs %d computed states | "
        "Hungarian=%.3f | Greedy=%.3f",
        C_leiden.shape[0],
        C_computed.shape[0],
        hungarian_score,
        greedy_score,
    )
    unique_computed = [int(x) for x in sorted(np.unique(cell_states))]
    unique_leiden = [int(x) for x in sorted(np.unique(leiden_labels))]
    return {
        "sim_matrix": sim,
        "hungarian_score": hungarian_score,
        "hungarian_row_idx": row_idx,
        "hungarian_col_idx": col_idx,
        "greedy_score": greedy_score,
        "best_leiden_per_computed": best_leiden_per_computed,
        "n_leiden": C_leiden.shape[0],
        "n_computed": C_computed.shape[0],
        "unique_computed": unique_computed,
        "unique_leiden": unique_leiden,
    }


def _plot_matching_heatmap(
    sim: np.ndarray,
    marker_row: np.ndarray,
    marker_col: np.ndarray,
    legend_label: str,
    title: str,
    output_path: Path,
    unique_computed: list[int] | None = None,
    unique_leiden: list[int] | None = None,
    cell_fractions: dict[int, float] | None = None,
    spot_fractions: dict[int, float] | None = None,
) -> None:
    n_leiden, n_computed = sim.shape
    # Fall back to positional indices only when actual IDs are unavailable
    if unique_computed is None:
        unique_computed = list(range(n_computed))
    if unique_leiden is None:
        unique_leiden = list(range(n_leiden))

    fig, ax = plt.subplots(
        figsize=(max(5, n_computed * 0.7 + 1), max(4, n_leiden * 0.6 + 1))
    )
    im = ax.imshow(sim, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Cosine similarity")

    ax.scatter(
        marker_col,
        marker_row,
        marker="*",
        s=80,
        color="black",
        zorder=5,
        label=legend_label,
    )
    ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel("Computed state", fontsize=11)
    ax.set_ylabel("Leiden cluster", fontsize=11)
    ax.set_title(title, fontsize=12)

    # x-axis: actual computed-state IDs with optional fractions
    x_labels = []
    for state_id in unique_computed:
        parts = [f"S{state_id}"]
        if cell_fractions is not None:
            parts.append(f"c:{cell_fractions.get(state_id, 0):.1%}")
        if spot_fractions is not None:
            parts.append(f"s:{spot_fractions.get(state_id, 0):.1%}")
        x_labels.append("\n".join(parts))

    ax.set_xticks(range(n_computed))
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_yticks(range(n_leiden))
    ax.set_yticklabels([f"L{l}" for l in unique_leiden], fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Centroid heatmap → %s", output_path)


def plot_centroid_matching_heatmap(
    match_results: dict,
    output_path: Path,
    cell_fractions: dict[int, float] | None = None,
    spot_fractions: dict[int, float] | None = None,
) -> None:
    """Hungarian-matched cosine similarity heatmap: Leiden clusters × computed states."""
    _plot_matching_heatmap(
        sim=match_results["sim_matrix"],
        marker_row=match_results["hungarian_row_idx"],
        marker_col=match_results["hungarian_col_idx"],
        legend_label=f"Hungarian match  (mean={match_results['hungarian_score']:.3f})",
        title="Centroid cosine similarity (Hungarian):\nLeiden (shared genes) vs Computed states",
        output_path=output_path,
        unique_computed=match_results.get("unique_computed"),
        unique_leiden=match_results.get("unique_leiden"),
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
    )


def plot_centroid_matching_greedy(
    match_results: dict,
    output_path: Path,
    cell_fractions: dict[int, float] | None = None,
    spot_fractions: dict[int, float] | None = None,
) -> None:
    """Greedy best-match heatmap: each computed state maps to its best Leiden cluster."""
    n_computed = match_results["n_computed"]
    _plot_matching_heatmap(
        sim=match_results["sim_matrix"],
        marker_row=match_results["best_leiden_per_computed"],
        marker_col=np.arange(n_computed),
        legend_label=f"Greedy best match  (mean={match_results['greedy_score']:.3f})",
        title="Centroid cosine similarity (Greedy):\nLeiden (shared genes) vs Computed states",
        output_path=output_path,
        unique_computed=match_results.get("unique_computed"),
        unique_leiden=match_results.get("unique_leiden"),
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
    )


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
) -> None:
    """
    Heatmap of the contingency matrix (counts, log-scaled colour), with the
    argmax assignment for each fine-side cluster marked by a star.

    Axis tick labels include cell fractions (derived from the contingency
    matrix) for both computed states and Leiden clusters.  Spot fractions for
    the computed states are shown when *spot_fractions* is provided.
    """
    ct = match_results["contingency_matrix"]
    score = match_results["score"]
    K, L = ct.shape
    unique_computed = match_results.get("unique_computed", list(range(K)))
    unique_leiden = match_results.get("unique_leiden", list(range(L)))

    if match_results["best_computed_per_leiden"] is not None:
        marker_rows = match_results["best_computed_per_leiden"]  # (L,)
        marker_cols = np.arange(L)
    else:
        marker_rows = np.arange(K)
        marker_cols = match_results["best_leiden_per_computed"]  # (K,)

    total = ct.sum()
    cell_fracs_computed = ct.sum(axis=1) / total if total > 0 else np.zeros(K)
    cell_fracs_leiden = ct.sum(axis=0) / total if total > 0 else np.zeros(L)

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

    # x-axis: actual Leiden IDs with cell fraction
    ax.set_xticks(range(L))
    ax.set_xticklabels(
        [f"L{l}\n{cell_fracs_leiden[j]:.1%}" for j, l in enumerate(unique_leiden)],
        fontsize=7,
    )

    # y-axis: actual computed-state IDs with cell fraction (+ spot fraction if provided)
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


# ──────────────────────────────────────────────────────────────────────────────
# AUC-based matching
# ──────────────────────────────────────────────────────────────────────────────


def compute_auc_matching(
    adata_sc: AnnData,
    cell_states: np.ndarray,
    leiden_labels: np.ndarray,
    gene_list: list[str],
    max_iter: int = 50,
) -> dict:
    """
    Match predicted clusters to Leiden clusters via AUC scores.

    For each Leiden cluster m, the "score" of cell i to m is the cosine
    similarity between cell i's expression and the centroid of m.  For each
    pair (predicted cluster k, Leiden cluster m) the AUC is the area under
    the ROC curve when using these scores to predict membership in cluster k.

    Algorithm
    ---------
    1. Compute (n_cells × n_leiden) cosine-similarity score matrix.
    2. Compute initial AUC matrix (K × L).
    3. Assign each predicted cluster k to the Leiden cluster with max AUC.
    4. Iterate until convergence:
       - Recompute AUC excluding cells from other predicted clusters already
         assigned to the same Leiden cluster.
       - Update assignment to new argmax.
    5. Merge: for each Leiden cluster, combine all predicted clusters assigned
       to it and recompute a merged AUC.

    Returns
    -------
    dict with:
        auc_matrix_initial   ndarray (K × L)
        auc_matrix_final     ndarray (K × L)
        assignment           ndarray (K,)  predicted cluster → leiden index
        mean_auc             float         mean AUC of assigned pairs
        merged_aucs          dict          leiden index → merged AUC
        mean_merged_auc      float
        n_iterations         int
        n_computed / n_leiden int
    """
    available = [g for g in gene_list if g in adata_sc.var_names]
    if len(available) < 2:
        raise ValueError("Too few genes for AUC matching.")

    X = _dense_X(adata_sc[:, available])  # (n_cells, G)

    unique_computed = sorted(np.unique(cell_states))
    unique_leiden = sorted(np.unique(leiden_labels))
    K = len(unique_computed)
    L = len(unique_leiden)

    centroids = np.stack([X[leiden_labels == l].mean(axis=0) for l in unique_leiden])
    scores = cosine_similarity(X, centroids)  # (n_cells, L)

    membership = np.stack(
        [(cell_states == k).astype(np.float32) for k in unique_computed], axis=1
    )  # (n_cells, K)

    def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        pos = int(y_true.sum())
        if pos == 0 or pos == len(y_true):
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    def _build_auc_matrix(assignment: np.ndarray | None) -> np.ndarray:
        auc = np.zeros((K, L))
        for ki in range(K):
            for li in range(L):
                if assignment is not None:
                    conflict = np.zeros(len(cell_states), dtype=bool)
                    for ki2 in range(K):
                        if ki2 != ki and assignment[ki2] == li:
                            conflict |= membership[:, ki2].astype(bool)
                    keep = ~conflict
                    auc[ki, li] = _auc(membership[keep, ki], scores[keep, li])
                else:
                    auc[ki, li] = _auc(membership[:, ki], scores[:, li])
        return auc

    auc_matrix = _build_auc_matrix(None)
    auc_matrix_initial = auc_matrix.copy()
    assignment = auc_matrix.argmax(axis=1).copy()

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        auc_matrix = _build_auc_matrix(assignment)
        new_assignment = auc_matrix.argmax(axis=1)
        if np.array_equal(new_assignment, assignment):
            logger.info("AUC matching converged after %d iteration(s).", n_iter)
            break
        assignment = new_assignment
    else:
        logger.warning("AUC matching did not converge after %d iterations.", max_iter)

    mean_auc = float(auc_matrix[np.arange(K), assignment].mean())

    merged_aucs: dict[int, float] = {}
    for li, l in enumerate(unique_leiden):
        assigned_ki = [ki for ki in range(K) if assignment[ki] == li]
        if not assigned_ki:
            continue
        merged_pos = membership[:, assigned_ki].max(axis=1)
        merged_aucs[l] = _auc(merged_pos, scores[:, li])

    mean_merged_auc = (
        float(np.mean(list(merged_aucs.values()))) if merged_aucs else float("nan")
    )

    logger.info(
        "AUC matching: K=%d computed, L=%d leiden | "
        "mean_auc=%.3f | mean_merged_auc=%.3f | iterations=%d",
        K,
        L,
        mean_auc,
        mean_merged_auc,
        n_iter,
    )
    return {
        "auc_matrix_initial": auc_matrix_initial,
        "auc_matrix_final": auc_matrix,
        "assignment": assignment,
        "mean_auc": mean_auc,
        "merged_aucs": merged_aucs,
        "mean_merged_auc": mean_merged_auc,
        "n_iterations": n_iter,
        "n_computed": K,
        "n_leiden": L,
        "unique_computed": [int(x) for x in unique_computed],
        "unique_leiden": [int(x) for x in unique_leiden],
    }


def plot_auc_matching_heatmap(
    match_results: dict,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """
    Heatmap of the final AUC matrix (K × L) with the assigned Leiden cluster
    for each predicted state marked by a star.  A second panel shows the
    initial AUC matrix for comparison.
    """
    auc_final = match_results["auc_matrix_final"]
    auc_initial = match_results["auc_matrix_initial"]
    assignment = match_results["assignment"]
    mean_auc = match_results["mean_auc"]
    n_iter = match_results["n_iterations"]
    unique_computed = match_results.get(
        "unique_computed", list(range(auc_final.shape[0]))
    )
    unique_leiden = match_results.get("unique_leiden", list(range(auc_final.shape[1])))
    K, L = auc_final.shape

    fig, axes = plt.subplots(
        1, 2, figsize=(max(10, L * 0.55 + 2) * 2, max(4, K * 0.55 + 2))
    )

    for ax, auc, label in zip(
        axes,
        [auc_initial, auc_final],
        ["Initial AUC", f"Final AUC  ({n_iter} iter)"],
    ):
        im = ax.imshow(auc, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
        fig.colorbar(im, ax=ax, label="AUC")

        for ki in range(K):
            for li in range(L):
                ax.text(
                    li,
                    ki,
                    f"{auc[ki, li]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

        ax.set_xlabel("Leiden cluster", fontsize=11)
        ax.set_ylabel("Computed state", fontsize=11)
        ax.set_xticks(range(L))
        ax.set_xticklabels([f"L{l}" for l in unique_leiden], fontsize=7)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"S{k}" for k in unique_computed], fontsize=7)
        ax.set_title(label, fontsize=11)

    axes[1].scatter(
        assignment,
        np.arange(K),
        marker="*",
        s=120,
        color="black",
        zorder=5,
        label=f"Assignment  (mean AUC={mean_auc:.3f})",
    )
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"AUC matching: Computed states × Leiden clusters{title_suffix}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("AUC matching heatmap → %s", output_path)
