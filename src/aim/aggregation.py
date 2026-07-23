"""
Reference aggregation for the AIM method (clustering half, part 1).

    compute_leiden_aggregates  adata + shared genes -> per-cluster expression
                               sums, sizes, and centroids (raw + normalized),
                               stored on adata_sc.uns
    assemble_state_profiles    labels_k              -> size-weighted per-state
                               profiles M (per K)

`assemble_state_profiles` is the torch mirror of
``metrics.reconstruction.assemble_state_centroids`` (which does the same on
numpy for the disk-based analysis).
"""

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
    UNS_SHARED_GENES,
)


def _sum_by_cluster(
    expr: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    sums = np.zeros((n_clusters, expr.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, expr)
    return sums


def compute_leiden_aggregates(adata_sc: AnnData) -> None:
    """
    Aggregate the reference over its Leiden clusters (on the shared genes) and
    store the results on ``adata_sc.uns``, raw (``.X``) and normalized
    (``obsm[OBSM_LOGNORM_SHARED_GENES]``) alike:

        uns["shared_genes"]                   list[str]      the exact gene order the
                                                               arrays below are aligned to
        uns["leiden_expr_sums_shared"]        (L x G_shared)  summed raw expression per cluster
        uns["leiden_sizes"]                   (L,)            number of cells per cluster
        uns["leiden_centroids_shared"]        (L x G_shared)  per-cluster mean raw expression
        uns["leiden_expr_sums_shared_norm"]   (L x G_shared)  summed normalized expression per cluster
        uns["leiden_centroids_shared_norm"]   (L x G_shared)  per-cluster mean normalized expression

    Reads raw counts from ``adata_sc[:, shared_genes].X``, the normalized
    variant from ``adata_sc.obsm[OBSM_LOGNORM_SHARED_GENES]`` (as written by
    ``aim.sweep.pre_processing`` — already column-aligned to ``shared_genes``,
    so unlike the raw side it needs no further gene slicing here), and cluster
    labels from ``adata_sc.obs[OBS_LEIDEN_ALL_GENES]`` (as written by
    ``aim.clustering.run_leiden_clustering``). Genes are column-aligned to
    ``shared_genes``, which is itself stored so downstream consumers (e.g. the
    post-mapping analysis) reuse this exact gene order instead of recomputing
    (and potentially reordering) the sc/st gene intersection. All-zero raw
    centroid rows are nudged to a tiny uniform value so their cosine distance
    stays defined for ``tree.build_agglomeration_tree`` (the normalized variant
    isn't used there, so it isn't nudged).
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

    adata_sc.uns[UNS_LEIDEN_SIZES] = sizes
    adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES] = expr_sums
    adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES] = centroids
    adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES_NORM] = expr_sums_norm
    adata_sc.uns[UNS_LEIDEN_CENTROIDS_SHARED_GENES_NORM] = centroids_norm


def assemble_state_profiles_shared_genes(
    labels_k: np.ndarray,
    k: int,
    adata_sc: AnnData,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Size-weighted per-state gene expression profiles from the subcluster->state
    label array (as returned by ``tree.labels_at_k``) and the per-cluster
    expression sums (L x G) / sizes (L,) stored on ``adata_sc.uns`` by
    ``compute_leiden_aggregates``:

        M[s] = (sum_{l: labels_k[l]=s} expr_sums[l]) / (sum_{l: labels_k[l]=s} sizes[l])
    """
    expr_sums = adata_sc.uns[UNS_LEIDEN_EXPR_SUMS_SHARED_GENES]
    sizes = adata_sc.uns[UNS_LEIDEN_SIZES]

    state_sums = np.zeros((k, expr_sums.shape[1]), dtype=np.float32)
    np.add.at(state_sums, labels_k, expr_sums)
    state_sizes = np.zeros(k, dtype=np.float32)
    np.add.at(state_sizes, labels_k, sizes)

    m = state_sums / (state_sizes[:, None] + eps)
    return torch.as_tensor(m, dtype=torch.float32)
