"""Reference aggregation: per-Leiden-cluster expression sums, sizes, and centroids,
and the size-weighted per-state expression profiles M for a given K cut."""

import numpy as np
import torch
from anndata import AnnData

from adata_schema import (
    OBS_LEIDEN_ALL_GENES,
    OBSM_LOGNORM_SHARED_GENES,
    UNS_LEIDEN_CENTROIDS_SHARED_GENES,
    UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM,
    UNS_LEIDEN_EXPR_SUMS_SHARED_GENES,
    UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM,
    UNS_LEIDEN_SIZES,
    UNS_LEIDEN_UNIT_SUMS_SHARED_GENES,
    UNS_SHARED_GENES,
)


def _sum_by_cluster(
    expr: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Sum rows of ``expr`` into ``n_clusters`` buckets given per-row cluster labels."""
    sums = np.zeros((n_clusters, expr.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, expr)
    return sums


def compute_leiden_aggregates(adata_sc: AnnData) -> None:
    """
    Aggregate the reference over its Leiden clusters on the shared genes, raw and
    normalized, storing the per-cluster sums, sizes, and centroids (all L x G_shared,
    column-aligned to UNS_SHARED_GENES).

    Requires: adata_sc.uns[UNS_SHARED_GENES], adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES],
    adata_sc.obs[OBS_LEIDEN_ALL_GENES].
    Adds: adata_sc.uns[UNS_LEIDEN_SIZES], adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES],
    adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES],
    adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM],
    adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM].

    All-zero raw centroid rows are nudged to a tiny uniform value so their cosine
    distance stays defined for the agglomeration tree; the normalized centroids are not.

    Also stores the per-cluster sum of L2-normalized raw cell vectors
    (UNS_LEIDEN_UNIT_SUMS_SHARED_GENES), from which any K cut's cell-level cosine
    dispersion is recovered by additive pooling (see
    ``assemble_state_dispersion_shared_genes``).
    """
    adata_shared = adata_sc[:, adata_sc.uns[UNS_SHARED_GENES]]

    raw = adata_shared.X
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype=np.float32)

    # Sum of unit-normalized raw cells per cluster: mean cosine of a cluster's
    # cells to any centroid m is (unit_sum . m/||m||) / size, so pooling these
    # over a K cut gives the state's cell-level mean cosine (hence dispersion).
    unit = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)

    norm = adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES]
    if hasattr(norm, "toarray"):
        norm = norm.toarray()
    norm = np.asarray(norm, dtype=np.float32)

    labels = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    n_clusters = int(labels.max()) + 1

    sizes = np.bincount(labels, minlength=n_clusters).astype(np.float32)

    expr_sums = _sum_by_cluster(raw, labels, n_clusters)
    centroids = expr_sums / (sizes[:, None] + 1e-8)
    zero_rows = np.where(~centroids.any(axis=1))[0]
    if zero_rows.size:
        centroids[zero_rows] = 1e-6

    expr_sums_norm = _sum_by_cluster(norm, labels, n_clusters)
    centroids_norm = expr_sums_norm / (sizes[:, None] + 1e-8)

    unit_sums = _sum_by_cluster(unit, labels, n_clusters)

    adata_sc.uns[UNS_LEIDEN_SIZES] = sizes
    adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES] = expr_sums
    adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES] = centroids
    adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM] = expr_sums_norm
    adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM] = centroids_norm
    adata_sc.uns[UNS_LEIDEN_UNIT_SUMS_SHARED_GENES] = unit_sums


def assemble_state_profiles_shared_genes(
    labels_k: np.ndarray,
    k: int,
    adata_sc: AnnData,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Size-weighted per-state expression profiles M (k x G_shared) for a subcluster->state
    label array: M[s] = sum of expr_sums over its subclusters / sum of their sizes.

    Requires: adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES], adata_sc.uns[UNS_LEIDEN_SIZES].
    """
    expr_sums = adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES]
    sizes = adata_sc.uns[UNS_LEIDEN_SIZES]

    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=np.float32)
    np.add.at(state_sums, labels_k, expr_sums)
    state_sizes = np.zeros(k, dtype=np.float32)
    np.add.at(state_sizes, labels_k, sizes)

    m = state_sums / (state_sizes[:, None] + eps)
    return torch.as_tensor(m, dtype=torch.float32)


def assemble_state_dispersion_shared_genes(
    labels_k: np.ndarray,
    k: int,
    adata_sc: AnnData,
    m_shared: torch.Tensor,
    shrinkage: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-state cell-level cosine dispersion sigma (k,) for a subcluster->state cut:
    sigma[s] = 1 - mean cosine of every cell in state s to the state centroid m[s].

    Recovered by pooling the per-Leiden unit-vector sums: for state s with member
    subclusters, mean cosine = (sum_c unit_sums[c] . m_hat[s]) / (sum_c sizes[c]),
    where m_hat[s] = m[s]/||m[s]||. Tight/singleton states (sigma ~ 0) are shrunk
    toward the size-weighted global dispersion with pseudocount ``shrinkage`` and
    floored at ``eps`` so downstream division stays finite.

    Requires: adata_sc.uns[UNS_LEIDEN_UNIT_SUMS_SHARED_GENES], adata_sc.uns[UNS_LEIDEN_SIZES].
    """
    unit_sums = adata_sc.uns[UNS_LEIDEN_UNIT_SUMS_SHARED_GENES]
    sizes = adata_sc.uns[UNS_LEIDEN_SIZES]

    state_unit_sums = np.zeros((k, unit_sums.shape[1]), dtype=np.float32)
    np.add.at(state_unit_sums, labels_k, unit_sums)
    state_sizes = np.zeros(k, dtype=np.float32)
    np.add.at(state_sizes, labels_k, sizes)

    m = np.asarray(m_shared, dtype=np.float32)
    m_hat = m / (np.linalg.norm(m, axis=1, keepdims=True) + eps)
    cos_bar = np.einsum("sg,sg->s", state_unit_sums, m_hat) / (state_sizes + eps)
    sigma = 1.0 - cos_bar

    global_sigma = float(np.average(sigma, weights=state_sizes + eps))
    sigma = (state_sizes * sigma + shrinkage * global_sigma) / (state_sizes + shrinkage)
    sigma = np.maximum(sigma, eps)

    return torch.as_tensor(sigma, dtype=torch.float32)
