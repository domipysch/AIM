"""
Agglomeration tree for the agglomerative method (clustering half, part 2).

The tree is built ONCE over the Leiden-subcluster centroids with average-linkage
on the shared-gene cosine distance (scipy does the "merge closest, recompute,
repeat" recursion). Every K then reuses the same tree:

    build_agglomeration_tree   centroids   -> scipy linkage matrix Z
    labels_at_k                Z, k        -> subcluster label array (0..k-1)
    merge_height_for_k         Z, k        -> linkage distance of the K+1 -> K merge

Downstream steps (aggregation, disk I/O) consume the ``labels_k`` array
directly rather than a materialized one-hot subcluster->state matrix.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def build_agglomeration_tree(centroids: np.ndarray) -> np.ndarray:
    """
    Average-linkage agglomeration tree over centroids (shared-gene cosine).#
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html.
    """
    return linkage(centroids, method="average", metric="cosine")


def labels_at_k(linkage_z: np.ndarray, k: int, n_leiden: int) -> np.ndarray:
    """Cut the agglomeration tree at k states -> subcluster label array (0..k-1)."""
    if k >= n_leiden:
        return np.arange(n_leiden, dtype=int)
    raw = fcluster(linkage_z, t=k, criterion="maxclust")  # 1..k
    # remap to contiguous 0..k-1
    _, remapped = np.unique(raw, return_inverse=True)
    return remapped.astype(int)


def merge_height_for_k(linkage_z: np.ndarray, k: int, n_leiden: int) -> float:
    """Linkage distance of the merge that produced k clusters (0.0 at k = L)."""
    if k >= n_leiden:
        return 0.0
    # merge row r reduces cluster count to (n_leiden - r - 1); solve for k
    return float(linkage_z[n_leiden - k - 1, 2])
