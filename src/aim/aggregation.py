"""Reference aggregation: per-start-cluster expression sums, sizes, and centroids,
and the size-weighted per-state expression profiles M for a given K cut."""

import numpy as np
from anndata import AnnData

from aim.adata_schema import (
    OBS_START_CLUSTER,
    OBSM_LOGNORM_SHARED_GENES,
    UNS_SHARED_GENES,
    UNS_START_CLUSTER_CENTROIDS_SHARED_GENES,
    UNS_START_CLUSTER_CENTROIDS_SHARED_GENES_NORM,
    UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES,
    UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM,
    UNS_START_CLUSTER_SIZES,
)


def _sum_by_cluster(
    expr: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Sum rows of ``expr`` into ``n_clusters`` buckets given per-row cluster labels."""
    sums = np.zeros((n_clusters, expr.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, expr)
    return sums


def compute_start_cluster_aggregates(adata_sc: AnnData) -> None:
    """
    Aggregate the reference over its start clusters on the shared genes, raw and
    normalized, storing the per-cluster sums, sizes, and centroids (all L x G_shared,
    column-aligned to UNS_SHARED_GENES).

    Mapper-facing and mode-independent: the start clusters are the Leiden
    over-clusters or the annotated types, and this reads them the same way either way.

    Requires: adata_sc.uns[UNS_SHARED_GENES], adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES],
    adata_sc.obs[OBS_START_CLUSTER].
    Adds: adata_sc.uns[UNS_START_CLUSTER_SIZES],
    adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES],
    adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES],
    adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM],
    adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES_NORM].

    All-zero raw centroid rows are nudged to a tiny uniform value so their cosine
    distance stays defined for the agglomeration tree; the normalized centroids are not.
    """
    adata_shared = adata_sc[:, adata_sc.uns[UNS_SHARED_GENES]]

    raw = adata_shared.X
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    raw = np.asarray(raw, dtype=np.float32)

    norm = adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES]
    if hasattr(norm, "toarray"):
        norm = norm.toarray()
    norm = np.asarray(norm, dtype=np.float32)

    labels = adata_sc.obs[OBS_START_CLUSTER].astype(int).to_numpy()
    n_clusters = int(labels.max()) + 1

    sizes = np.bincount(labels, minlength=n_clusters).astype(np.float32)

    expr_sums = _sum_by_cluster(raw, labels, n_clusters)
    centroids = expr_sums / (sizes[:, None] + 1e-8)
    zero_rows = np.where(~centroids.any(axis=1))[0]
    if zero_rows.size:
        centroids[zero_rows] = 1e-6

    expr_sums_norm = _sum_by_cluster(norm, labels, n_clusters)
    centroids_norm = expr_sums_norm / (sizes[:, None] + 1e-8)

    adata_sc.uns[UNS_START_CLUSTER_SIZES] = sizes
    adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES] = expr_sums
    adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES] = centroids
    adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES_NORM] = expr_sums_norm
    adata_sc.uns[UNS_START_CLUSTER_CENTROIDS_SHARED_GENES_NORM] = centroids_norm


def _size_weighted_state_profiles(
    labels_k: np.ndarray,
    k: int,
    expr_sums: np.ndarray,
    sizes: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Pool per-start-cluster expression sums into size-weighted per-state means M
    (k x G_shared): M[s] = sum of expr_sums over its start clusters / sum of sizes."""
    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=np.float32)
    np.add.at(state_sums, labels_k, expr_sums)
    state_sizes = np.zeros(k, dtype=np.float32)
    np.add.at(state_sizes, labels_k, sizes)

    m = state_sums / (state_sizes[:, None] + eps)
    return m.astype(np.float32, copy=False)


def assemble_state_profiles_shared_genes(
    labels_k: np.ndarray,
    k: int,
    adata_sc: AnnData,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Size-weighted per-state raw expression profiles M (k x G_shared) for a
    start-cluster->state label array (mean raw counts per state).

    Requires: adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES],
    adata_sc.uns[UNS_START_CLUSTER_SIZES].
    """
    return _size_weighted_state_profiles(
        labels_k,
        k,
        adata_sc.uns[UNS_START_CLUSTER_EXPR_SUMS_SHARED_GENES],
        adata_sc.uns[UNS_START_CLUSTER_SIZES],
        eps,
    )
