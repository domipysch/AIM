"""
Mapping-level metrics for AIM's decoupled post-mapping analysis: loading the
raw P (mapping_prob.h5ad) and G (leiden_merge_prob.h5ad) matrices, hardening
them (argmax), validating the hard mapping, and assembling per-state gene
expression centroids for the cosine-similarity reconstruction.

This is AIM-specific (unlike metrics.onehot / metrics.cossim, which are
generic and shared with reference_aligners/mapping_analysis) because AIM has
a two-level structure — Leiden subclusters merged into computed states via G,
spots mapped onto those states via P — that reference aligners don't have.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from metrics.onehot import hard_mapping

logger = logging.getLogger(__name__)


def load_mapping_matrices(
    output_folder: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], np.ndarray]:
    """
    Load the raw P (spot -> state) and G (Leiden -> state) matrices, plus the
    per-cell Leiden cluster index, from an AIM run's saved outputs.

    Args:
        output_folder: Folder containing mapping_prob.h5ad, leiden_merge_prob.h5ad,
                        and leiden_overclustering.h5ad, as written by main.py.

    Returns:
        P: spot -> state matrix (S x L).
        G: Leiden -> state matrix (L x L).
        spot_names, leiden_names, state_names: axis labels.
        leiden_idx: per-cell Leiden cluster index (C,), parsed from
                    leiden_overclustering.h5ad's obs["leiden_cluster"].
    """
    output_folder = Path(output_folder)
    mapping_path = output_folder / "mapping_prob.h5ad"
    leiden_merge_path = output_folder / "leiden_merge_prob.h5ad"
    clusters_path = output_folder / "leiden_overclustering.h5ad"
    for path in (mapping_path, leiden_merge_path, clusters_path):
        if not path.exists():
            raise FileNotFoundError(f"Required mapping output missing: {path}")

    mapping_ad = ad.read_h5ad(mapping_path)
    leiden_merge_ad = ad.read_h5ad(leiden_merge_path)

    P = np.asarray(mapping_ad.X, dtype=np.float64)  # (S x L)
    G = np.asarray(leiden_merge_ad.X, dtype=np.float64)  # (L x L)
    spot_names = mapping_ad.obs_names.tolist()
    leiden_names = leiden_merge_ad.obs_names.tolist()
    state_names = leiden_merge_ad.var_names.tolist()

    leiden_cluster_names = ad.read_h5ad(clusters_path).obs["leiden_cluster"].to_numpy()
    leiden_idx = np.array(
        [int(name.rsplit("_", 1)[-1]) for name in leiden_cluster_names]
    )

    return P, G, spot_names, leiden_names, state_names, leiden_idx


def compute_hard_mapping_validated(
    P: np.ndarray, G: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Hard (argmax one-hot) versions of P and G, with a consistency check.

    Args:
        P: soft spot -> state matrix (S x L).
        G: soft Leiden -> state matrix (L x L).

    Returns:
        (P_hard, G_hard, n_active_states, n_mapped_states):
        - P_hard, G_hard: one-hot (argmax) versions of P and G.
        - n_active_states: number of AIM states actually aggregated out of the
          Leiden clusters — columns of G_hard with >=1 one, i.e. at least one
          Leiden cluster hard-maps there.
        - n_mapped_states: number of AIM states actually used by spots —
          columns of P_hard with >=1 one, i.e. at least one spot hard-maps
          there.
        Both are counts of "columns with a surviving 1 after argmax", applied
        to G and P respectively — as opposed to L, the total number of state
        slots (see model.py), most of which may go unused.

    Raises:
        ValueError: if any spot is hard-assigned (via P_hard) to a state whose
                    column in G_hard is entirely zero — i.e. no Leiden cluster
                    hard-maps to that state, so it has no cells and its
                    centroid cannot be computed.
    """
    P_hard = hard_mapping(P)
    G_hard = hard_mapping(G)

    states_with_leiden_support = set(np.where(G_hard.sum(axis=0) > 0)[0].tolist())
    states_used_by_spots = set(np.where(P_hard.sum(axis=0) > 0)[0].tolist())
    orphaned = sorted(states_used_by_spots - states_with_leiden_support)
    if orphaned:
        raise ValueError(
            f"Hard mapping is inconsistent: state(s) {orphaned} are hard-assigned "
            "to by at least one spot (mapping_prob) but no Leiden cluster hard-maps "
            "to them in leiden_merge_prob — these states have no cells, so their "
            "centroid is undefined."
        )
    return P_hard, G_hard, len(states_with_leiden_support), len(states_used_by_spots)


def compute_leiden_expression_sums(
    adata_sc: AnnData, leiden_idx: np.ndarray, n_leiden: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-Leiden-cluster summed expression and cluster sizes, from an sc AnnData
    and the per-cell Leiden index (mirrors main.py's old leiden_aggregates,
    recomputed here from disk-loaded data instead of live tensors).

    Returns:
        expr_sums: summed expression per Leiden cluster (L x G).
        sizes: number of cells per Leiden cluster (L,).
    """
    X = (
        adata_sc.X.toarray()
        if hasattr(adata_sc.X, "toarray")
        else np.asarray(adata_sc.X)
    )
    expr_sums = np.zeros((n_leiden, X.shape[1]), dtype=np.float64)
    sizes = np.zeros(n_leiden, dtype=np.float64)
    for l in range(n_leiden):
        mask = leiden_idx == l
        sizes[l] = mask.sum()
        if mask.any():
            expr_sums[l] = X[mask].sum(axis=0)
    return expr_sums, sizes


def assemble_state_centroids(
    G: np.ndarray, expr_sums: np.ndarray, sizes: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """
    Assemble per-state gene expression centroids from the merge matrix G and
    the fixed Leiden-cluster expression sums/sizes (mirrors main.py's old
    assemble_state_gep):

        M[k] = (sum_l G[l,k] * expr_sums[l]) / (sum_l G[l,k] * sizes[l])

    Works for both soft G (weighted average) and hard G (a clean mean over all
    cells whose Leiden cluster hard-maps to state k).

    Args:
        G: Leiden -> state matrix (L x L).
        expr_sums: summed expression per Leiden cluster (L x G_genes).
        sizes: number of cells per Leiden cluster (L,).

    Returns:
        M: state gene expression profiles (L x G_genes).
    """
    weighted_sum = G.T @ expr_sums  # (L x G_genes)
    state_sizes = G.T @ sizes  # (L,)
    return weighted_sum / (state_sizes[:, None] + eps)


def predict_expression(mapping: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Predicted spot expression Z' = mapping @ centroids (S x G_genes)."""
    return np.asarray(mapping) @ np.asarray(centroids)


def save_matrix_h5ad(
    matrix: np.ndarray, obs_names: list[str], var_names: list[str], path: Path
) -> None:
    """Save a plain matrix as an h5ad with the given obs/var labels."""
    AnnData(
        X=np.asarray(matrix, dtype=np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    ).write_h5ad(path)
