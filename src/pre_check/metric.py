"""
Centroid cosine similarity metric and permutation test.

All operations are scale-invariant: centroids are z-scored within each
modality before comparison, so cross-modality scale differences never matter.
"""

import logging

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (also used by summary.py and plots.py)
# ---------------------------------------------------------------------------


def to_dense(X) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)


def shared_expression(
    sc_adata: AnnData, st_adata: AnnData
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_sc, X_st, shared_gene_names) restricted to shared genes."""
    shared = sorted(set(sc_adata.var_names) & set(st_adata.var_names))
    if len(shared) == 0:
        raise ValueError("No shared genes between SC and ST datasets.")
    X_sc = to_dense(sc_adata[:, shared].X)
    X_st = to_dense(st_adata[:, shared].X)
    return X_sc, X_st, shared


def leiden_labels(X: np.ndarray, resolution: float) -> np.ndarray:
    """Run Leiden on expression matrix X; return integer label array."""
    adata = sc.AnnData(X=X.copy())
    old_verbosity = sc.settings.verbosity
    sc.settings.verbosity = 0
    try:
        sc.pp.neighbors(adata, use_rep="X", random_state=42)
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=42,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    finally:
        sc.settings.verbosity = old_verbosity
    return adata.obs["leiden"].astype(int).values


def zscored_centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-cluster mean centroids, z-scored within each centroid."""
    unique = sorted(np.unique(labels))
    C = np.stack([X[labels == k].mean(axis=0) for k in unique])
    std = C.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (C - C.mean(axis=1, keepdims=True)) / std


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def centroid_cosine_sim(C_sc: np.ndarray, C_st: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Hungarian-matched mean cosine similarity between z-scored centroids.

    Returns
    -------
    score       : float — mean cosine sim of optimally matched cluster pairs
    sim_matrix  : ndarray (n_clusters_sc × n_clusters_st) — full similarity matrix
    """
    sim = cosine_similarity(C_sc, C_st)
    row_idx, col_idx = linear_sum_assignment(-sim)
    score = float(sim[row_idx, col_idx].mean())
    return score, sim


def variance_rank_spearman(X_sc: np.ndarray, X_st: np.ndarray) -> float:
    """
    Spearman correlation of per-gene variance ranks between SC and ST.
    High value → the same genes carry signal in both modalities.
    Note: can be inflated by HVG pre-selection; use as a diagnostic, not a primary metric.
    """
    rho, _ = spearmanr(X_sc.var(axis=0), X_st.var(axis=0))
    return round(float(rho), 4)


def greedy_cosine_sim(
    C_sc: np.ndarray, C_st: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    For each ST cluster, pick the single best-matching SC cluster (argmax cosine sim).
    SC clusters can be matched by multiple ST clusters — no exclusivity constraint.

    Returns
    -------
    score       : mean cosine similarity across all ST clusters' best matches
    sim_matrix  : full (n_sc × n_st) cosine similarity matrix
    best_sc_per_st : index array of length n_st — best SC cluster for each ST cluster
    """
    sim = cosine_similarity(C_sc, C_st)  # (n_sc, n_st)
    best_sc_per_st = sim.argmax(axis=0)  # (n_st,)
    score = float(sim[best_sc_per_st, np.arange(sim.shape[1])].mean())
    return round(score, 4), sim, best_sc_per_st


def permutation_test(
    C_sc: np.ndarray,
    C_st: np.ndarray,
    X_st: np.ndarray,
    labels_st: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict:
    """
    Validate centroid_cosine_sim against a gene-shuffle null distribution.

    Gene labels of X_st are shuffled n_permutations times, destroying any
    real SC-ST correspondence while preserving internal structure. Leiden
    clustering is reused from the real data.

    Returns
    -------
    dict with keys: real_score, null_mean, null_std, z_score, p_value,
                    n_permutations, null_scores
    """
    real_score, _ = centroid_cosine_sim(C_sc, C_st)
    logger.info(f"Permutation test — real score: {real_score:.3f}")

    rng = np.random.default_rng(seed)
    null_scores = []
    for i in range(n_permutations):
        perm = rng.permutation(X_st.shape[1])
        C_st_perm = zscored_centroids(X_st[:, perm], labels_st)
        score, _ = centroid_cosine_sim(C_sc, C_st_perm)
        null_scores.append(score)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{n_permutations} permutations done")

    null_arr = np.array(null_scores)
    null_std = float(null_arr.std())
    z_score = (
        float((real_score - null_arr.mean()) / null_std)
        if null_std > 0
        else float("nan")
    )
    p_value = float((null_arr >= real_score).mean())

    logger.info(
        f"  null: {null_arr.mean():.3f} ± {null_std:.3f} | "
        f"z={z_score:.2f} | p={p_value:.3f}"
    )

    return {
        "real_score": real_score,
        "null_mean": float(null_arr.mean()),
        "null_std": null_std,
        "z_score": z_score,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "null_scores": null_scores,
    }


def top_gene_jaccard(
    cluster_df_sc: pd.DataFrame,
    cluster_df_st: pd.DataFrame,
    top_k: int,
) -> float:
    """
    Jaccard overlap of the union of top-k marker genes across all SC clusters
    vs the union of top-k marker genes across all ST clusters.

    A high value means both modalities identify the same genes as the most
    distinctive markers, regardless of which clusters they come from.
    """
    genes_sc = set(cluster_df_sc[cluster_df_sc["rank"] <= top_k]["gene"])
    genes_st = set(cluster_df_st[cluster_df_st["rank"] <= top_k]["gene"])
    intersection = genes_sc & genes_st
    union = genes_sc | genes_st
    score = round(len(intersection) / len(union), 4) if union else float("nan")
    logger.info(
        f"Top-gene Jaccard (top {top_k:2d}): {score:.3f}  "
        f"(|SC|={len(genes_sc)}, |ST|={len(genes_st)}, |∩|={len(intersection)}, |∪|={len(union)})"
    )
    return score
